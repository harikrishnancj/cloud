import os
import urllib.parse as urlparse
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# --- Persistent Storage Paths ---
# Models are cached here; they download once and reuse on every restart.
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")
# Each processed video gets its own Chroma collection stored on disk.
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_stores")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

current_video_id = None
vector_store = None

# Initialize embedding model ONCE with a local cache folder.
# On first run it downloads the model (~90MB). Subsequent starts load from disk instantly.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=MODEL_CACHE_DIR
)

# Check for token in env
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use InferenceClient with chat_completion — works with all modern instruct models
# via the conversational (chat) API, bypassing the text-generation task issue.
# featherless-ai is confirmed active on the user's HF account.
chat_client = InferenceClient(
    provider="featherless-ai",
    model="Qwen/Qwen2.5-7B-Instruct",
    token=hf_token
)

SYSTEM_PROMPT = """You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know."""

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def _extract_video_id(url: str) -> str:
    """Extracts video ID from various YouTube URL formats."""
    # Check for short urls (youtu.be)
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    # Check for standard youtube URLs
    parsed_url = urlparse.urlparse(url)
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        # Check query params for v
        query_params = urlparse.parse_qs(parsed_url.query)
        if 'v' in query_params:
            return query_params['v'][0]
    # Fallback to general splits
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    raise ValueError("Invalid YouTube URL format. Could not extract Video ID.")

def _get_persist_path(video_id: str) -> str:
    """Returns the disk path for a given video's Chroma store."""
    return os.path.join(VECTOR_STORE_DIR, video_id)

def process_video(url: str):
    global vector_store, current_video_id
    try:
        video_id = _extract_video_id(url)
        persist_path = _get_persist_path(video_id)

        # ✅ If we've already processed this video, load from disk instantly.
        if os.path.exists(persist_path):
            vector_store = Chroma(
                persist_directory=persist_path,
                embedding_function=embedding_model
            )
            current_video_id = video_id
            return {"status": "success", "message": "Video loaded from cache", "video_id": video_id}

        # Otherwise, fetch transcript, embed, and persist to disk.
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript_text = " ".join(snippet.text for snippet in transcript.snippets)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        docs = splitter.split_text(transcript_text)

        # Persist Chroma to disk — survives server restarts.
        vector_store = Chroma.from_texts(
            embedding=embedding_model,
            texts=docs,
            persist_directory=persist_path
        )
        current_video_id = video_id
        return {"status": "success", "message": "Video processed successfully", "video_id": video_id}
    except TranscriptsDisabled:
        raise Exception("There is no transcript available for this video.")
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def ask_question(question: str):
    global vector_store
    if not vector_store:
        raise Exception("Please process a video first.")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    # Call HuggingFace chat completion API directly (works with all modern instruct models)
    response = chat_client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=512,
        temperature=0.1
    )
    return response.choices[0].message.content
