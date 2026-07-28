# ==========================================
# IMPORTS
# ==========================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse
)

import uvicorn
import os
import cv2
import numpy as np
import base64

from pathlib import Path

from ultralytics import YOLO

from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

from groq import Groq



# ==========================================
# CONFIGURATION
# ==========================================


BASE_DIR = Path(__file__).resolve().parent



# YOLO MODEL

MODEL_PATH = BASE_DIR / "fmdv2.pt"



# DOCUMENT FOR RAG CHAT

DOC_PATH = (
    BASE_DIR /
    "facemaskrprojectdocument.docx"
)



# FRONTEND

STATIC_DIR = (
    BASE_DIR /
    "static"
)



# ==========================================
# LIVE WEBCAM CONFIGURATION
# ==========================================


CAMERA_INDEX = 0


FRAME_WIDTH = 640

FRAME_HEIGHT = 480



# YOLO FILTERS

CONF_THRESHOLD = 0.50

IOU_THRESHOLD = 0.45



# ==========================================
# LOAD ENVIRONMENT
# ==========================================


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)



# ==========================================
# FASTAPI APPLICATION
# ==========================================


app = FastAPI(

    title="Face Mask Detection + AI Chat",

    version="2.0"

)



# ==========================================
# LOAD YOLO MODEL
# ==========================================


def load_model():


    if not MODEL_PATH.exists():

        raise RuntimeError(

            f"YOLO model not found: {MODEL_PATH}"

        )


    return YOLO(

        str(MODEL_PATH)

    )



model = load_model()



# ==========================================
# INITIALIZE WEBCAM
# ==========================================


camera = cv2.VideoCapture(
    CAMERA_INDEX
)



camera.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    FRAME_WIDTH

)



camera.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    FRAME_HEIGHT

)



if not camera.isOpened():

    raise RuntimeError(

        "Cannot access webcam"

    )



print("Webcam started successfully")

print(
    f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}"
)

print(
    f"Confidence threshold: {CONF_THRESHOLD}"
)

print(
    f"IoU threshold: {IOU_THRESHOLD}"
)
# ==========================================
# LOW LIGHT IMAGE ENHANCEMENT
# ==========================================


def enhance_low_light(frame):

    """
    CLAHE enhancement for
    dim / grainy webcam conditions.
    """


    lab = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2LAB

    )


    l, a, b = cv2.split(lab)



    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)

    )



    enhanced_l = clahe.apply(l)



    enhanced = cv2.merge(

        (

            enhanced_l,

            a,

            b

        )

    )



    enhanced = cv2.cvtColor(

        enhanced,

        cv2.COLOR_LAB2BGR

    )


    return enhanced




# ==========================================
# YOLO LIVE FRAME PROCESSING
# ==========================================


def detect_live_frame(frame):

    """
    Process one webcam frame.

    Pipeline:
    Webcam
       |
    Resize 640x480
       |
    CLAHE enhancement
       |
    YOLO inference
       |
    Draw detections
    """



    # --------------------------------------
    # Resize for lower latency
    # --------------------------------------


    frame = cv2.resize(

        frame,

        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )

    )



    # --------------------------------------
    # Improve low-light webcam quality
    # --------------------------------------


    frame = enhance_low_light(

        frame

    )



    # --------------------------------------
    # YOLO inference
    # --------------------------------------


    results = model(

        frame,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        verbose=False

    )



    result = results[0]



    # --------------------------------------
    # Process detections
    # --------------------------------------


    for box in result.boxes:



        x1, y1, x2, y2 = map(

            int,

            box.xyxy[0].tolist()

        )



        class_id = int(

            box.cls.item()

        )



        confidence = float(

            box.conf.item()

        )



        # ------------------------------
        # Class labels
        # ------------------------------


        if class_id == 0:


            label = "Mask"

            color = (

                0,

                255,

                0

            )



        elif class_id == 1:


            label = "Incorrect Mask"

            color = (

                0,

                165,

                255

            )



        else:


            label = "No Mask"

            color = (

                0,

                0,

                255

            )



        text = (

            f"{label} "

            f"{confidence:.2f}"

        )



        # ------------------------------
        # Draw bounding box
        # ------------------------------


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            2

        )



        # ------------------------------
        # Draw label
        # ------------------------------


        cv2.putText(

            frame,

            text,

            (

                x1,

                y1 - 10

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )



    return frame
# ==========================================
# LIVE VIDEO FRAME GENERATOR
# ==========================================


def generate_frames():

    """
    Continuously captures webcam frames,
    runs YOLO detection,
    and streams processed frames.
    """


    while True:


        success, frame = camera.read()



        if not success:

            print(
                "Failed to read webcam frame"
            )

            break



        # Run detection

        processed_frame = detect_live_frame(

            frame

        )



        # Encode frame as JPEG


        ret, buffer = cv2.imencode(

            ".jpg",

            processed_frame

        )



        if not ret:

            continue



        frame_bytes = buffer.tobytes()



        # MJPEG streaming format


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +

            frame_bytes

            +

            b"\r\n"

        )




# ==========================================
# LIVE CAMERA STREAM ENDPOINT
# ==========================================


@app.get("/video_feed")
async def video_feed():


    return StreamingResponse(

        generate_frames(),

        media_type=

        "multipart/x-mixed-replace; boundary=frame"

    )
# ==========================================
# LOAD PROJECT DOCUMENT FOR RAG
# ==========================================


def load_project_doc():

    """
    Reads project documentation
    and creates TF-IDF vectors.
    """


    if not DOC_PATH.exists():

        print(
            "Documentation file not found"
        )

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




# ==========================================
# DOCUMENT RETRIEVAL
# ==========================================


def retrieve_relevant_chunk(query: str):

    """
    Finds the most relevant
    paragraph from project document.
    """


    if not doc_chunks or doc_index is None:

        return ""



    vectorizer, doc_vectors = doc_index



    query_vector = vectorizer.transform(

        [query]

    )



    similarity = cosine_similarity(

        query_vector,

        doc_vectors

    )[0]



    best_index = similarity.argmax()



    return doc_chunks[best_index]





# ==========================================
# GROQ AI FUNCTION
# ==========================================


def call_llm(

    system_prompt,

    context,

    question

):


    if not GROQ_API_KEY:


        return (

            "Groq API key missing. "

            "Add GROQ_API_KEY in .env"

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

                    "Project documentation:\n\n"

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

            f"AI Error: {str(e)}"

        )





# ==========================================
# AI CHAT ENDPOINT
# ==========================================


@app.post("/chat")
async def chat(

    question: str = Form(...)

):


    try:


        context = retrieve_relevant_chunk(

            question

        )



        system_prompt = (

            "You are an AI tutor for a "

            "Face Mask Detection project "

            "using YOLO, OpenCV, FastAPI, "

            "and RAG. "

            "Explain answers clearly "

            "with proper formatting."

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
    # ==========================================
# FRONTEND SERVER
# ==========================================


@app.get("/", response_class=HTMLResponse)
async def serve_index():

    """
    Serves the live webcam HTML interface.
    """


    html_path = (

        STATIC_DIR /

        "indexvdv1.html"

    )



    if not html_path.exists():


        return HTMLResponse(

            content=

            "<h1>indexvdv1.html not found</h1>",

            status_code=404

        )



    return HTMLResponse(

        html_path.read_text(

            encoding="utf-8"

        )

    )




# ==========================================
# CLEANUP WEBCAM
# ==========================================


@app.on_event("shutdown")
def shutdown_event():

    """
    Release webcam when server stops.
    """


    if camera.isOpened():

        camera.release()



    cv2.destroyAllWindows()




# ==========================================
# APPLICATION START
# ==========================================


if __name__ == "__main__":


    uvicorn.run(

        "mainvdv1:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )