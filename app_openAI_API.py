import os
from pathlib import Path
import streamlit as st

from knowledge_base import build_index, search_index, load_documents
from openai import OpenAI

st.set_page_config(
    page_title="Principal Desk AI — TCEA",
    page_icon="🏫",
    layout="wide",
)

BASE = Path(__file__).parent
DOCS = BASE / "documents"

st.title("🏫 Principal Desk AI")
st.caption("TCEA — Prototype Knowledge Assistant")

with st.sidebar:
    st.header("Knowledge Base")
    st.write("This prototype answers from the documents placed in the `documents/` folder.")
    st.info(
        "Sensitive student lists, phone numbers, and other personal-data documents "
        "are intentionally excluded from this first prototype."
    )
    model = st.text_input(
        "OpenAI model",
        value=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        help="Change this if your OpenAI account uses a different model.",
    )

@st.cache_resource
def get_index():
    return build_index(DOCS)

try:
    index = get_index()
except Exception as e:
    st.error(f"Could not build the knowledge base: {e}")
    st.stop()

documents = load_documents(DOCS)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Ask the Principal Desk")
with col2:
    st.metric("Indexed documents", len(documents))

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What is the minimum attendance required to appear in the SEE?\n"
        "Example: What was the approved budget mentioned in the Swachhata Abhiyan memo?"
    ),
    height=110,
)

if st.button("Ask", type="primary") and question.strip():
    hits = search_index(index, question, k=6)

    st.subheader("Answer")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "OPENAI_API_KEY is not configured. Showing the most relevant source excerpts "
            "instead of generating an AI answer."
        )
        for h in hits:
            with st.expander(h["source"]):
                st.write(h["text"])
    else:
        context = "\n\n".join(
            f"[SOURCE: {h['source']}]\n{h['text']}" for h in hits
        )
        prompt = f"""
You are Principal Desk AI for Techno College of Engineering Agartala (TCEA).

Answer the user's question using ONLY the supplied source excerpts.
Do not invent facts. If the sources do not contain enough information, say so.
Preserve the terminology and authority notes in the source documents.
If sources conflict, explicitly report the conflict rather than silently resolving it.
For dates, rules, fees, attendance, examination decisions, or other administrative
matters, distinguish the source document's statement from any inference.

USER QUESTION:
{question}

SOURCE EXCERPTS:
{context}
"""
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            st.write(response.output_text)
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            st.write("Relevant source excerpts:")
            for h in hits:
                with st.expander(h["source"]):
                    st.write(h["text"])

    st.subheader("Sources used")
    for h in hits:
        st.write(f"• {h['source']}")

with st.expander("Documents currently included"):
    for d in documents:
        st.write(f"• {d}")
