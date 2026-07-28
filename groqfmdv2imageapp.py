import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- IMPORTS FOR CHAT/RAG ---
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- ENV + GROQ ---
from dotenv import load_dotenv
from groq import Groq

MODEL_PATH = "fmdv2.pt"
DOC_PATH = "facemaskrprojectdocument.docx"

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_model():
    """Load YOLO model once."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None
    return YOLO(MODEL_PATH)


# --- LOAD PROJECT DOCUMENT AND BUILD TF-IDF INDEX ---
@st.cache_resource
def load_project_doc():
    """Load the facemask project document and return list of text chunks + TF-IDF index."""
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


# --- REAL LLM CALL USING GROQ ---
def call_llm(system_prompt, context, question):
    """Call Groq's chat completion API."""
    if not GROQ_API_KEY:
        return (
            "Groq API key not found. Please set GROQ_API_KEY in your .env file "
            "before using the chatbot."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"Here is relevant project documentation:\n\n{context}"},
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
        page_title="Face Mask Detection + AI Chat",
        page_icon="😷",
        layout="wide"
    )

    # Initialize session state variables for persistence
    if "last_uploaded_file_name" not in st.session_state:
        st.session_state.last_uploaded_file_name = None
    if "annotated_rgb" not in st.session_state:
        st.session_state.annotated_rgb = None
    if "original_img" not in st.session_state:
        st.session_state.original_img = None
    if "detection_stats" not in st.session_state:
        st.session_state.detection_stats = None

    # --- MAIN LAYOUT ---
    col_left, col_right = st.columns([2, 1])

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Settings")

    conf_thresh = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    end_session = st.sidebar.checkbox(
        "End image session",
        help="Stop using the image upload mode for this session."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("About this app")
    st.sidebar.markdown(
        """
        **Model:** YOLO-based facemask detector  
        **Classes:** mask, no mask  
        **Input:** JPG / PNG images  
        **Output:** Bounding boxes with labels and confidence scores.
        """
    )

    # Load YOLO model
    model = load_model()
    if model is None:
        return

    # ---------------- LEFT COLUMN: IMAGE DETECTION ----------------
    with col_left:
        st.title("Face Mask Detection")
        st.caption(
            "Upload an image to detect masks. "
            "You can chat about the project on the right while viewing results."
        )

        if end_session:
            st.info(
                "Session ended. Uncheck 'End image session' in the sidebar "
                "to use the app again."
            )
            # Clear stored state if session ends
            st.session_state.last_uploaded_file_name = None
            st.session_state.annotated_rgb = None
            st.session_state.original_img = None
        else:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                # Check if a new file is uploaded
                if st.session_state.last_uploaded_file_name != uploaded_file.name:
                    st.session_state.last_uploaded_file_name = uploaded_file.name
                    
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.session_state.original_img = img

                    with st.status("Detecting masks...", expanded=False) as status:
                        try:
                            results = model(img, conf=conf_thresh)
                            annotated_frame = results[0].plot()
                            st.session_state.annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                            boxes = results[0].boxes
                            num_detections = len(boxes)
                            num_mask = sum(1 for b in boxes if int(b.cls[0]) == 0)
                            num_no_mask = sum(1 for b in boxes if int(b.cls[0]) == 1)

                            st.session_state.detection_stats = {
                                "total": num_detections,
                                "mask": num_mask,
                                "no_mask": num_no_mask
                            }
                            status.update(label="Detection complete.", state="complete")
                        except Exception as e:
                            status.update(label="Error during detection.", state="error")
                            st.error(f"Error processing image: {e}")

            # Display results if present in session state (survives chat reruns)
            if st.session_state.original_img is not None and st.session_state.annotated_rgb is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original")
                    st.image(
                        cv2.cvtColor(st.session_state.original_img, cv2.COLOR_BGR2RGB),
                        caption="Original image"
                    )

                with col2:
                    st.subheader("Detections")
                    st.image(
                        st.session_state.annotated_rgb,
                        caption=f"Detections (conf ≥ {conf_thresh:.2f})"
                    )

                stats = st.session_state.detection_stats
                if stats:
                    st.info(
                        f"Detections: {stats['total']} faces "
                        f"({stats['mask']} with mask, {stats['no_mask']} without mask)."
                    )
                st.success(f"Detection completed (confidence ≥ {conf_thresh:.2f}).")

    # ---------------- RIGHT COLUMN: ALWAYS-VISIBLE CHATBOT ----------------
    with col_right:
        bot_icon = "🤖"
        chunks, vectorizer_and_matrix = load_project_doc()

        if "chat_messages_image" not in st.session_state:
            st.session_state.chat_messages_image = [
                {
                    "role": "assistant",
                    "content": "Hi! I can answer questions about this Face Mask Detection project.",
                }
            ]

        for msg in st.session_state.chat_messages_image:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                with st.chat_message(role, avatar=bot_icon):
                    st.write(content)
            else:
                with st.chat_message(role):
                    st.write(content)

        user_question = st.chat_input("Ask a question about the project (image mode)...")

        if user_question:
            st.session_state.chat_messages_image.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.write(user_question)

            context = retrieve_relevant_chunk(
                user_question, chunks, vectorizer_and_matrix
            )

            system_prompt = (
                "You are a helpful tutor for a Face Mask Detection project "
                "using YOLOv8, OpenCV, and Streamlit. "
                "The user is now in image mode. "
                "Use only the provided project documentation to answer, "
                "and explain in simple language."
            )

            with st.status("Processing answer...", expanded=False) as status:
                status.update(label="Processing...", state="running")
                assistant_reply = call_llm(
                    system_prompt=system_prompt,
                    context=context,
                    question=user_question,
                )
                status.update(label="Answer ready.", state="complete")

            st.session_state.chat_messages_image.append(
                {"role": "assistant", "content": assistant_reply}
            )
            with st.chat_message("assistant", avatar=bot_icon):
                st.write(assistant_reply)


if __name__ == "__main__":
    main()