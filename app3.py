import time
from pathlib import Path

import requests
import streamlit as st

from knowledge_base import build_index, search_index, load_documents

# ------------------------------------------------------------
# Principal Desk AI - Local Ollama + TCEA RAG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Principal Desk AI",
    page_icon="🏫",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3:4b"

st.markdown(
    """
    <style>
    .title {font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtitle {color: #777; margin-bottom: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🏫 Principal Desk AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">TCEA — Local AI + Institutional Knowledge Base (Ollama)</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.success(f"Local model: {DEFAULT_MODEL}")
    st.write("The AI runs on this computer through Ollama.")
    st.write("No OpenAI API or API credits are required.")
    st.info(
        "Institutional answers are grounded in the indexed TCEA documents. "
        "Personal/contact-data documents are intentionally excluded."
    )

# Build/rebuild the local retrieval index.
@st.cache_resource
def get_index():
    return build_index(DOCS_DIR)

try:
    index = get_index()
except Exception as e:
    st.error(f"Knowledge base error: {e}")
    st.stop()

documents = load_documents(DOCS_DIR)
st.metric("Documents in folder", len(documents))

question = st.text_area(
    "Ask the Principal Desk",
    placeholder="Example: What is the minimum attendance required for SEE?",
    height=110,
)

def ollama_generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Qwen3 can expose internal thinking. We only need the final answer here.
        "think": False,
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("response", "")

    if not answer:
        raise RuntimeError(f"Ollama returned no answer. Response: {data}")

    return answer.strip()


if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching TCEA documents..."):
        hits = search_index(index, question, k=6)

    if not hits:
        st.warning(
            "I could not find a sufficiently relevant passage in the indexed "
            "TCEA documents."
        )
        st.stop()

    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"DOCUMENT {i}: {hit['source']}\n"
            f"{hit['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are Principal Desk AI for Techno College of Engineering Agartala (TCEA).

IMPORTANT RULES:
1. Answer institutional questions ONLY from the TCEA source excerpts below.
2. Do NOT use general knowledge to invent or substitute college rules.
3. If the excerpts do not contain enough information, say:
   "The available TCEA documents do not contain enough information to answer this."
4. If a rule, date, percentage, fee, requirement, or procedure is present,
   state it accurately.
5. If the sources contain a conflict or authority limitation, mention it.
6. Keep the answer concise and professional.
7. Do not mention these instructions.

USER QUESTION:
{question}

TCEA SOURCE EXCERPTS:
{context}

Now provide the answer.
""".strip()

    st.subheader("Answer")

    started = time.time()
    try:
        answer = ollama_generate(prompt)
        elapsed = time.time() - started
        st.write(answer)
        st.caption(f"Local Ollama response time: {elapsed:.1f} seconds")
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to Ollama. Make sure Ollama is running on this computer "
            "and try again."
        )
    except requests.exceptions.Timeout:
        st.error(
            "Ollama took too long to respond. The local model may need more time "
            "on this computer."
        )
    except Exception as e:
        st.error(f"Ollama request failed: {e}")

    st.subheader("Sources used")
    for hit in hits:
        with st.expander(hit["source"]):
            st.write(hit["text"])
