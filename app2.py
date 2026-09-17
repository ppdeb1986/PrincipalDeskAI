import json
from pathlib import Path

import requests
import streamlit as st

from knowledge_base import build_index, search_index, load_documents

st.set_page_config(page_title="Principal Desk AI — TCEA", page_icon="🏫", layout="wide")

BASE = Path(__file__).parent
DOCS = BASE / "documents"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

st.title("🏫 Principal Desk AI")
st.caption("TCEA — Local AI + institutional knowledge base (Ollama)")

with st.sidebar:
    st.header("System")
    st.success(f"Local model: {MODEL}")
    st.write("The AI model runs on this computer through Ollama. No OpenAI API is required.")
    st.info("Sensitive student lists and personal/contact data are intentionally excluded from this first knowledge base.")

@st.cache_resource
def get_index():
    return build_index(DOCS)

try:
    index = get_index()
except Exception as e:
    st.error(f"Could not build the knowledge base: {e}")
    st.stop()

documents = load_documents(DOCS)
st.metric("Documents in folder", len(documents))

question = st.text_area(
    "Ask the Principal Desk",
    placeholder="Example: What is the minimum attendance required to appear in the SEE?",
    height=110,
)


def ask_ollama(question, hits):
    context = "\n\n".join(
        f"SOURCE: {h['source']}\n{h['text']}" for h in hits
    )

    system = """You are Principal Desk AI for Techno College of Engineering Agartala (TCEA).

Your task is to answer institutional questions using ONLY the supplied TCEA source excerpts.
Do not use your general knowledge to invent or substitute college rules.
If the excerpts do not contain enough information, say: 'The available TCEA documents do not contain enough information to answer this question.'
If the sources contain an apparent conflict, report the conflict and identify the source documents instead of silently resolving it.
Do not invent dates, percentages, fees, names, procedures, approvals, or authorities.
For rules, distinguish what the source explicitly states from any explanation you provide.
Keep answers concise but useful. At the end, list the source document names used."""

    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"USER QUESTION:\n{question}\n\nTCEA SOURCE EXCERPTS:\n{context}"},
        ],
        "options": {"temperature": 0.1},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "").strip()


if st.button("Ask", type="primary") and question.strip():
    hits = search_index(index, question, k=6)

    st.subheader("Answer")

    if not hits:
        st.warning("No sufficiently relevant text was found in the indexed TCEA documents.")
    else:
        try:
            answer = ask_ollama(question, hits)
            st.write(answer)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to Ollama at {OLLAMA_URL}. Make sure Ollama is running.\n\n{e}")
        except Exception as e:
            st.error(f"Local AI request failed: {e}")

    st.subheader("Sources retrieved")
    for h in hits:
        with st.expander(h["source"]):
            st.write(h["text"])

with st.expander("Documents currently included"):
    for d in documents:
        st.write(f"• {d}")
