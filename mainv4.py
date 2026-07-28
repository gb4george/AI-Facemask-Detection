from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

import cv2
import numpy as np
from ultralytics import YOLO
import os

from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# ---------- CONFIG ----------

MODEL_PATH = "fmdv2.pt"
DOC_PATH = "facemaskrprojectdocument.docx"
BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# ---------- SERVE FRONTEND HTML ----------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    Serve the main HTML page (indexv5.html) for Face Mask Detection + AI.
    """
    html_path = BASE_DIR / "static/indexv5.html"   # make sure your file is named indexv5.html
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>indexv5.html not found</h1>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# ---------- AI LOGIC: YOLO MODEL + RAG LOADING ----------

def load_model():
    """
    Load YOLO model once.

    Same training assumptions as your Streamlit app:
    - 3 classes: 0=Mask, 1=MaskIncorrect, 2=NoMask
    - Trained for low-quality webcam conditions.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found.")
    return YOLO(MODEL_PATH)

model = load_model()

def load_project_doc():
    """
    Load the Face Mask Detection project document and build a TF-IDF indexv5.
    """
    if not os.path.exists(DOC_PATH):
        return [], None

    doc = Document(DOC_PATH)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    chunks = paragraphs

    if not chunks:
        return [], None

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(chunks)

    return chunks, (vectorizer, doc_vectors)

doc_chunks, doc_index = load_project_doc()

def retrieve_relevant_chunk(query: str) -> str:
    """
    Same RAG retrieval as in your Streamlit code:
    - Compute cosine similarity and pick best paragraph.
    """
    if not doc_chunks or doc_index is None:
        return ""

    vectorizer, doc_vectors = doc_index
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors)[0]
    best_idx = sims.argmax()
    return doc_chunks[best_idx]

def call_llm(system_prompt: str, context: str, question: str) -> str:
    """
    Call Groq LLM with system prompt + context + question.
    """
    if not GROQ_API_KEY:
        return (
            "Groq API key not found. Please set GROQ_API_KEY in your .env file."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": f"Here is relevant project documentation:\n\n{context}",
            },
            {"role": "user", "content": question},
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling Groq API: {e}"

# ---------- DETECTION ENDPOINT ----------

@app.post("/detect")
async def detect_face_mask(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.45),
    enhance: bool = Form(False),
):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data."})

    # Optional Enhancement (CLAHE) [Source 20]
    if enhance:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Run YOLO Inference
    results = model(img, conf=conf, iou=iou, verbose=False)
    boxes = results.boxes

    detections = []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy.tolist()) # Convert to int for drawing
        cls_id = int(b.cls)
        score = float(b.conf)

        # Define Colors and Labels [Source 17]
        if cls_id == 0: 
            label, color = "Mask", (0, 255, 0) # Green
        elif cls_id == 1: 
            label, color = "MaskIncorrect", (0, 165, 255) # Orange
        else: 
            label, color = "NoMask", (0, 0, 255) # Red

        # --- DRAWING LOGIC (Missing in your current code) ---
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({"bbox": [x1, y1, x2, y2], "label": label, "score": score})

    # Convert the processed image to Base64 for the HTML frontend
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{img_base64}" # This matches your JS line
    }

# ---------- CHAT ENDPOINT ----------

@app.post("/chat")
async def chat(question: str = Form(...)):
    """
    Input:
    - question text

    Output:
    - answer string from Groq LLM, using RAG context.
    """
    context = retrieve_relevant_chunk(question)

    system_prompt = (
        "You are a helpful tutor for a Face Mask Detection project "
        "using YOLOv8, OpenCV, and FastAPI. "
        "Use only the provided project documentation to answer, "
        "and explain in simple language."
    )

    answer = call_llm(system_prompt, context, question)

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("mainv3:app", host="0.0.0.0", port=8000, reload=True)