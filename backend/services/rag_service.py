import os
import urllib.parse as urlparse
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

# -----------------------------
# In-memory state
# -----------------------------
current_video_id = None
vector_store = None
embedding_model = None


# -----------------------------
# Lazy load Gemini Embeddings
# -----------------------------
def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading Gemini Embedding Client...")

        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        print("Gemini Embedding Client Ready.")

    return embedding_model


# -----------------------------
# Hugging Face LLM
# -----------------------------
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

chat_client = InferenceClient(
    provider="featherless-ai",
    model="Qwen/Qwen2.5-7B-Instruct",
    token=hf_token,
)

SYSTEM_PROMPT = """
You are a helpful assistant.

Answer ONLY using the provided transcript context.

If the answer cannot be found in the context, simply say:
"I don't know based on the transcript."
"""


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def _extract_video_id(url: str) ->str:
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    parsed_url = urlparse.urlparse(url)

    if parsed_url.hostname in ("youtube.com", "www.youtube.com"):
        query = urlparse.parse_qs(parsed_url.query)
        if "v" in query:
            return query["v"][0]

    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]

    raise ValueError("Invalid YouTube URL.")


def process_video(url: str):
    global current_video_id
    global vector_store

    try:
        video_id = _extract_video_id(url)

        if current_video_id == video_id and vector_store is not None:
            return {
                "status": "success",
                "message": "Video already processed.",
                "video_id": video_id,
            }

        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"],
        )

        transcript_text = " ".join(
            snippet.text for snippet in transcript.snippets
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
        )

        docs = splitter.split_text(transcript_text)

        embeddings = get_embedding_model()

        vector_store = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
        )

        current_video_id = video_id

        return {
            "status": "success",
            "message": "Video processed successfully.",
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
        search_kwargs={"k":4},
    )

    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    response = chat_client.chat_completion(
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT,
            },
            {
                "role":"user",
                "content":f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )

    return response.choices[0].message.content