import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Page Config (UX)
# -----------------------------
st.set_page_config(
    page_title="Language Translator · Groq",
    page_icon="🌍",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🌍 Language Translator</h1>
    <p style="text-align:center; color: gray;">
        Translate text into any language using LLaMA-3.1 (Groq)
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a language translation assistant. "
            "Translate the given text into {language}. "
            "Return ONLY the translated text."
        ),
        ("user", "{input}")
    ]
)

# -----------------------------
# LLM & Chain
# -----------------------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

chain = prompt | model | StrOutputParser()

# -----------------------------
# Session State (ONLY for output)
# -----------------------------
if "response" not in st.session_state:
    st.session_state.response = None

# -----------------------------
# Input UI (LET STREAMLIT CONTROL IT)
# -----------------------------
with st.form("translator_form"):
    user_text = st.text_area(
        "✍️ Enter text",
        placeholder="Type text to translate...",
        height=120,
        key="user_text"
    )

    language = st.text_input(
        "🌐 Translate to",
        placeholder="e.g., Hindi, Japanese, Arabic, Klingon",
        key="target_language"
    )

    submitted = st.form_submit_button("Translate")

# -----------------------------
# On Submit (USE CURRENT VALUES)
# -----------------------------
if submitted and user_text.strip() and language.strip():
    with st.spinner("🔄 Translating..."):
        st.session_state.response = chain.invoke(
            {
                "input": user_text,
                "language": language
            }
        )

# -----------------------------
# Output UI
# -----------------------------
if st.session_state.response:
    st.markdown("---")
    st.markdown("### Translated Text")
    st.markdown(
        f"""
        <div>
            {st.session_state.response}
        </div>
        """,
        unsafe_allow_html=True
    )
