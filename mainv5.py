# ==============================
# IMPORTS
# ==============================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse

import uvicorn
import os
import base64
import cv2
import numpy as np

from pathlib import Path

from ultralytics import YOLO

from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from groq import Groq


# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "fmdv2.pt"

DOC_PATH = BASE_DIR / "facemaskrprojectdocument.docx"

UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR = BASE_DIR / "outputs"


UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


app = FastAPI(
    title="AI Face Mask Detection System",
    version="1.0"
)


# ==============================
# FRONTEND SERVER
# ==============================

@app.get("/", response_class=HTMLResponse)
async def serve_index():

    html_path = BASE_DIR / "static" / "indexv6.html"

    if not html_path.exists():

        return HTMLResponse(
            content="<h1>indexv6.html not found</h1>",
            status_code=404
        )


    return HTMLResponse(
        html_path.read_text(
            encoding="utf-8"
        )
    )



# ==============================
# YOLO MODEL LOADING
# ==============================

def load_model():

    if not MODEL_PATH.exists():

        raise RuntimeError(
            f"YOLO model not found: {MODEL_PATH}"
        )


    return YOLO(str(MODEL_PATH))



model = load_model()



# ==============================
# DOCUMENT RAG LOADING
# ==============================

def load_project_doc():

    if not DOC_PATH.exists():

        return [], None


    doc = Document(
        str(DOC_PATH)
    )


    paragraphs = [

        p.text.strip()

        for p in doc.paragraphs

        if p.text.strip()

    ]


    if not paragraphs:

        return [], None



    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    vectors = vectorizer.fit_transform(
        paragraphs
    )


    return paragraphs, (
        vectorizer,
        vectors
    )



doc_chunks, doc_index = load_project_doc()
# ==============================
# RAG RETRIEVAL FUNCTIONS
# ==============================


def retrieve_relevant_chunk(query: str) -> str:

    """
    Find the most relevant paragraph
    from project documentation.
    """

    if not doc_chunks or doc_index is None:

        return ""


    vectorizer, doc_vectors = doc_index


    query_vector = vectorizer.transform(
        [query]
    )


    similarities = cosine_similarity(
        query_vector,
        doc_vectors
    )[0]


    best_index = similarities.argmax()


    return doc_chunks[best_index]



# ==============================
# GROQ LLM FUNCTION
# ==============================


def call_llm(
    system_prompt: str,
    context: str,
    question: str
):

    """
    Sends question + retrieved documentation
    to Groq LLM.
    """


    if not GROQ_API_KEY:

        return (
            "Groq API key missing. "
            "Please add GROQ_API_KEY to .env file."
        )



    try:

        client = Groq(
            api_key=GROQ_API_KEY
        )


        messages = [

            {
                "role": "system",
                "content": system_prompt
            },


            {
                "role": "assistant",
                "content":
                (
                    "Relevant project documentation:\n\n"
                    +
                    context
                )
            },


            {
                "role": "user",
                "content": question
            }

        ]



        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=messages

        )



        return (
            response
            .choices[0]
            .message
            .content
        )



    except Exception as e:


        return (
            f"Groq API error: {str(e)}"
        )



# ==============================
# IMAGE UTILITY FUNCTIONS
# ==============================


def image_to_base64(image):

    """
    Convert OpenCV image
    into browser display format.
    """


    success, buffer = cv2.imencode(
        ".jpg",
        image
    )


    if not success:

        return None



    encoded = base64.b64encode(
        buffer
    ).decode(
        "utf-8"
    )


    return (
        "data:image/jpeg;base64,"
        +
        encoded
    )
# ==============================
# YOLO DETECTION ENDPOINT
# ==============================


@app.post("/detect")
async def detect_face_mask(

    file: UploadFile = File(...),

    conf: float = Form(0.5),

    iou: float = Form(0.45),

    enhance: bool = Form(False)

):


    try:

        # ------------------------------
        # Read uploaded image
        # ------------------------------

        content = await file.read()


        if not content:

            return JSONResponse(

                status_code=400,

                content={
                    "error": "Empty file received"
                }

            )


        # ------------------------------
        # Save original image
        # ------------------------------

        upload_path = (
            UPLOAD_DIR /
            file.filename
        )


        with open(upload_path, "wb") as f:

            f.write(content)



        # ------------------------------
        # Decode image
        # ------------------------------

        nparr = np.frombuffer(

            content,

            np.uint8

        )


        img = cv2.imdecode(

            nparr,

            cv2.IMREAD_COLOR

        )



        if img is None:

            return JSONResponse(

                status_code=400,

                content={
                    "error":
                    "Invalid image format"
                }

            )



        # ------------------------------
        # Optional enhancement
        # ------------------------------

        if enhance:


            lab = cv2.cvtColor(

                img,

                cv2.COLOR_BGR2LAB

            )


            l, a, b = cv2.split(lab)


            clahe = cv2.createCLAHE(

                clipLimit=2.0,

                tileGridSize=(8,8)

            )


            l = clahe.apply(l)


            img = cv2.cvtColor(

                cv2.merge(
                    (
                        l,
                        a,
                        b
                    )
                ),

                cv2.COLOR_LAB2BGR

            )



        # ------------------------------
        # YOLO inference
        # ------------------------------

        results = model(

            img,

            conf=conf,

            iou=iou,

            verbose=False

        )


        result = results[0]


        detections = []



        # ------------------------------
        # Draw detections
        # ------------------------------

        for box in result.boxes:


            x1, y1, x2, y2 = (

                map(

                    int,

                    box.xyxy[0].tolist()

                )

            )


            cls_id = int(

                box.cls[0]

            )


            score = float(

                box.conf[0]

            )



            if cls_id == 0:

                label = "Mask"

                color = (
                    0,
                    255,
                    0
                )


            elif cls_id == 1:

                label = "MaskIncorrect"

                color = (
                    0,
                    165,
                    255
                )


            else:

                label = "NoMask"

                color = (
                    0,
                    0,
                    255
                )



            cv2.rectangle(

                img,

                (x1,y1),

                (x2,y2),

                color,

                2

            )


            cv2.putText(

                img,

                f"{label} {score:.2f}",

                (x1, y1-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                color,

                2

            )



            detections.append(

                {

                    "bbox":
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "label":
                    label,

                    "confidence":
                    score

                }

            )



        # ------------------------------
        # Save detected output image
        # ------------------------------

        output_path = (

            OUTPUT_DIR /

            file.filename

        )


        cv2.imwrite(

            str(output_path),

            img

        )



        # ------------------------------
        # Convert to Base64
        # ------------------------------

        encoded_image = image_to_base64(

            img

        )



        return {

            "success": True,

            "detections":
            detections,

            "image_base64":
            encoded_image,

            "saved_original":
            str(upload_path),

            "saved_detection":
            str(output_path)

        }



    except Exception as e:


        return JSONResponse(

            status_code=500,

            content={

                "error":
                str(e)

            }

        )
# ==============================
# CHAT ENDPOINT
# ==============================


@app.post("/chat")
async def chat(

    question: str = Form(...)

):

    try:

        # Retrieve relevant documentation
        context = retrieve_relevant_chunk(
            question
        )


        system_prompt = (

            "You are a helpful AI tutor for a "
            "Face Mask Detection project using "
            "YOLO, OpenCV, FastAPI, and RAG. "

            "Answer using the provided project "
            "documentation whenever possible. "

            "Explain concepts in simple language."

        )


        answer = call_llm(

            system_prompt,

            context,

            question

        )


        return {

            "answer": answer

        }



    except Exception as e:


        return JSONResponse(

            status_code=500,

            content={

                "error":
                str(e)

            }

        )



# ==============================
# APPLICATION START
# ==============================


if __name__ == "__main__":

    uvicorn.run(

        "mainv5:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )