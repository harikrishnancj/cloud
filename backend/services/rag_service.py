import os
import urllib.parse as urlparse
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# In-memory state
# -----------------------------
current_video_id = None
vector_store = None
embedding_model = None

# -----------------------------
# Lazy load embedding model
# -----------------------------
def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")

    return embedding_model

# -----------------------------
# Hugging Face client
# -----------------------------
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

chat_client = InferenceClient(
    provider="featherless-ai",
    model="Qwen/Qwen2.5-7B-Instruct",
    token=hf_token
)

SYSTEM_PROMPT = """
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, simply say you don't know.
"""


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID."""

    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    parsed_url = urlparse.urlparse(url)

    if parsed_url.hostname in ("youtube.com", "www.youtube.com"):
        query_params = urlparse.parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]

    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]

    raise ValueError("Invalid YouTube URL.")


def process_video(url: str):
    global current_video_id
    global vector_store

    try:
        video_id = _extract_video_id(url)

        # Already processed
        if current_video_id == video_id and vector_store is not None:
            return {
                "status": "success",
                "message": "Video already loaded",
                "video_id": video_id,
            }

        # Fetch transcript
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        transcript_text = " ".join(
            snippet.text for snippet in transcript.snippets
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
        )

        docs = splitter.split_text(transcript_text)

        # Load embedding model only when needed
        embeddings = get_embedding_model()

        vector_store = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
        )

        current_video_id = video_id

        return {
            "status": "success",
            "message": "Video processed successfully",
            "video_id": video_id,
        }

    except TranscriptsDisabled:
        raise Exception("No transcript available for this video.")

    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")


def ask_question(question: str):
    global vector_store

    if vector_store is None:
        raise Exception("Please process a video first.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    response = chat_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=512,
        temperature=0.1,
    )

    return response.choices[0].message.content