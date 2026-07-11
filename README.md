# YouTube RAG Application

This project is a standardized web application built with a **FastAPI backend** and a **React (Vite) frontend**. It allows users to paste a YouTube link, generate embeddings from the video transcript, and ask questions to an LLM using a Retrieval-Augmented Generation (RAG) approach.

## Prerequisites
- Python 3.8+
- Node.js 16+
- A Hugging Face account and API token (for LLM and embeddings).

## 1. Setup the Backend (FastAPI)

1. Navigate to the root folder:
   ```bash
   cd eresumrag
   ```
2. Activate your virtual environment (you have already done this):
   ```bash
   env\Scripts\activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Set up Environment Variables:
   Create a `.env` file in the root folder with your Hugging Face token:
   ```
   HF_TOKEN=your_hugging_face_token_here
   ```
5. Run the Backend Server:
   ```bash
   python backend/main.py
   ```
   The backend will be running at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## 2. Setup the Frontend (React Vite)

1. Open a new terminal window.
2. Navigate to the frontend directory:
   ```bash
   cd eresumrag/frontend
   ```
3. Run the development server (dependencies are already installed):
   ```bash
   npm run dev
   ```
4. Open the frontend in your browser (usually `http://localhost:5173`).

## 3. Deployment

**Backend (e.g., Render, Heroku):**
- You can deploy the backend using Docker or directly using a `Procfile` specifying: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Ensure to set `HF_TOKEN` in the deployment environment variables.

**Frontend (e.g., Vercel, Netlify):**
- In the frontend directory, change the API URL in `App.jsx` from `http://localhost:8000` to your deployed backend URL.
- Build the project using `npm run build`.
- Connect your GitHub repository to Vercel/Netlify for automatic deployment.
