import streamlit as st
import cv2
from ultralytics import YOLO
import os
import time

# RAG + Groq imports
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq

MODEL_PATH = "fmdv2.pt"
DOC_PATH = "facemaskrprojectdocument.docx"

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_model():
    """
    Load YOLO model once.

    Optimized for low-quality/low-light laptop webcams using tight bounds
    (conf around 0.50, iou around 0.45).

    Training-time strategy (not executed here):
    - Use Roboflow or custom augmentations:
      * Brightness: -25% to +25%
      * Blur: up to 2.5 px Gaussian blur
      * Noise: up to 5% white noise
      * Cutout / occlusion: random patches

    - Train with YOLOv8m / YOLOv8l using tuned hyperparameters
      (documented in your report, not in this app).
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_project_doc():
    """
    Load the Face Mask Detection project document and build a TF-IDF index.
    Used by the AI chatbot on the right side.
    """
    if not os.path.exists(DOC_PATH):
        st.warning(
            f"Project document '{DOC_PATH}' not found. "
            "Chatbot will have limited knowledge."
        )
        return [], None

    doc = Document(DOC_PATH)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    chunks = paragraphs

    if not chunks:
        return [], None

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(chunks)

    return chunks, (vectorizer, doc_vectors)


def retrieve_relevant_chunk(query, chunks, vectorizer_and_matrix):
    """Return the most relevant chunk for a given query using cosine similarity."""
    if not chunks or vectorizer_and_matrix is None:
        return ""

    vectorizer, doc_vectors = vectorizer_and_matrix
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors)[0]
    best_idx = sims.argmax()
    return chunks[best_idx]


def call_llm(system_prompt, context, question):
    """
    Call Groq's chat completion API.
    Uses GROQ_API_KEY from environment.
    """
    if not GROQ_API_KEY:
        return (
            "Groq API key not found. Please set GROQ_API_KEY in your .env file "
            "before using the chatbot."
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


def main():
    st.set_page_config(
        page_title="Face Mask Detection (Low-Quality Webcam Optimizer) + AI Chat",
        page_icon="📷",
        layout="wide",
    )

    # Two main columns: left = webcam, right = AI chat
    col_left, col_right = st.columns([2, 1])

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Settings & Enhancements")

    enable_enhancement = st.sidebar.checkbox(
        "Enhance Low-Light/Grainy Video",
        value=True,
        help=(
            "Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) "
            "to brighten dark indoor rooms and improve contrast on low-quality webcams."
        ),
    )

    conf_thresh = st.sidebar.slider(
        "Base confidence threshold (conf)",
        min_value=0.3,
        max_value=0.9,
        value=0.50,
        step=0.05,
        help=(
            "Higher values filter out false positives from grainy sensors "
            "(recommended ≥ 0.50 for high precision)."
        ),
    )

    iou_thresh = st.sidebar.slider(
        "IOU threshold (NMS)",
        min_value=0.3,
        max_value=0.7,
        value=0.45,
        step=0.05,
        help=(
            "Intersection-over-Union threshold for Non-Maximum Suppression. "
            "Helps prevent overlapping duplicate boxes on low-res frames."
        ),
    )

    run_camera = st.sidebar.checkbox(
        "Start webcam",
        help="Turn on the built-in laptop camera and run live facemask detection.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hardware & Optimization Note")
    st.sidebar.markdown(
        """
        **Target Hardware:** Built-in HP / Laptop 720p Webcam  
        **Countermeasures Active:**  
         - Optional CLAHE contrast correction for dim rooms  
         - Frame resizing to 640×480 to reduce latency  
         - Strict `conf` and `iou` filtering for ≥ 0.90 precision  
        """
    )

    # Load YOLO model
    model = load_model()
    if model is None:
        return

    # ---------------- LEFT COLUMN: WEBCAM FEED ----------------
    with col_left:
        st.title("Face Mask Detection (Laptop Webcam Optimizer)")
        st.caption(
            "Optimized for standard 720p built-in laptop webcams. "
            "Applies real-time frame scaling, optional contrast enhancement, "
            "and strict confidence thresholds."
        )

        webcam_container = st.container(border=True)
        with webcam_container:
            st.subheader("Live Webcam Feed")
            st.write(
                "Running real-time inference on your built-in camera stream. "
                "Ensure you are in a reasonably lit area or toggle enhancement on."
            )

            frame_placeholder = st.empty()
            summary_placeholder = st.empty()

            if run_camera:
                cap = cv2.VideoCapture(0)  # Default laptop camera index

                if not cap.isOpened():
                    st.error(
                        "Could not open webcam. Check camera permissions or if another app is using it."
                    )
                    return

                # Force lower resolution to stabilize frame rate on integrated graphics/CPUs
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                st.warning(
                    "Webcam running. Uncheck 'Start webcam' in the sidebar to stop."
                )

                with st.status(
                    "Processing low-res stream...", expanded=False
                ) as status:
                    status.update(label="Detecting...", state="running")

                    # Continuous live loop
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to read from camera.")
                            break

                        # 1. Optional CLAHE for low light/grainy sensors
                        if enable_enhancement:
                            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(
                                clipLimit=2.0, tileGridSize=(8, 8)
                            )
                            cl = clahe.apply(l)
                            limg = cv2.merge((cl, a, b))
                            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                        # 2. LIVE INFERENCE with strict metrics
                        results = model(
                            frame,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            verbose=False,
                        )
                        annotated_frame = results[0].plot()

                        # BGR -> RGB for Streamlit display
                        annotated_rgb = cv2.cvtColor(
                            annotated_frame, cv2.COLOR_BGR2RGB
                        )

                        # Show current frame
                        frame_placeholder.image(
                            annotated_rgb,
                            caption=(
                                f"Live Detections "
                                f"(conf ≥ {conf_thresh:.2f}, iou = {iou_thresh:.2f})"
                            ),
                        )

                        # Detection summary for this frame
                        boxes = results[0].boxes
                        num_detections = len(boxes)

                        # Class mapping: 0 -> Mask, 1 -> Mask Incorrect, 2 -> NoMask
                        num_mask = sum(1 for b in boxes if int(b.cls[0]) == 0)
                        num_incorrect = sum(1 for b in boxes if int(b.cls[0]) == 1)
                        num_no_mask = sum(1 for b in boxes if int(b.cls[0]) == 2)

                        summary_placeholder.info(
                            f"Detections in current frame: {num_detections} faces "
                            f"({num_mask} Mask, {num_incorrect} Incorrect, {num_no_mask} No Mask)."
                        )

                        # Small sleep to stabilize UI rendering loop
                        time.sleep(0.03)

                        # Stop condition: user unchecks the sidebar checkbox
                        if not st.session_state.get("Start webcam", run_camera):
                            break

                    # Release resources
                    cap.release()
                    cv2.destroyAllWindows()

                    status.update(label="Detection stopped.", state="complete")
            else:
                st.info("To start live detection, check 'Start webcam' in the sidebar.")

    # ---------------- RIGHT COLUMN: AI CHATBOT ----------------
    with col_right:
        bot_icon = "🤖"

        st.title("Project AI Chatbot")
        st.caption(
            "Ask questions about this Face Mask Detection project while "
            "the webcam runs on the left."
        )

        # Load document chunks and index
        chunks, vectorizer_and_matrix = load_project_doc()

        if "chat_messages_webcam" not in st.session_state:
            st.session_state.chat_messages_webcam = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi! I can answer questions about this Face Mask Detection project "
                        "(webcam mode)."
                    ),
                }
            ]

        # Show chat history
        for msg in st.session_state.chat_messages_webcam:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant":
                with st.chat_message(role, avatar=bot_icon):
                    st.write(content)
            else:
                with st.chat_message(role):
                    st.write(content)

        # Chat input widget
        user_question = st.chat_input("Ask a question about the project (webcam mode)...")

        if user_question:
            # Save and display user message
            st.session_state.chat_messages_webcam.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.write(user_question)

            # Retrieve relevant context from document
            context = retrieve_relevant_chunk(
                user_question, chunks, vectorizer_and_matrix
            )

            system_prompt = (
                "You are a helpful tutor for a Face Mask Detection project "
                "using YOLOv8, OpenCV, and Streamlit. "
                "The user is now in webcam mode. "
                "Use only the provided project documentation to answer, "
                "and explain in simple language."
            )

            # Status indicator while calling Groq
            with st.status("Processing answer...", expanded=False) as status:
                status.update(label="Processing...", state="running")

                assistant_reply = call_llm(
                    system_prompt=system_prompt,
                    context=context,
                    question=user_question,
                )

                status.update(label="Answer ready.", state="complete")

            # Save and display assistant reply
            st.session_state.chat_messages_webcam.append(
                {"role": "assistant", "content": assistant_reply}
            )
            with st.chat_message("assistant", avatar=bot_icon):
                st.write(assistant_reply)


if __name__ == "__main__":
    main()