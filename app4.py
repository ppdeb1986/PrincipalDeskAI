import re
import time
from pathlib import Path

import requests
import streamlit as st

from knowledge_base import build_index, search_index, load_documents

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:4b"

st.set_page_config(
    page_title="Principal Desk AI",
    page_icon="🏫",
    layout="wide",
)

st.title("🏫 Principal Desk AI")
st.caption("TCEA — Local AI + Institutional Knowledge Base (Ollama)")

with st.sidebar:
    st.success(f"Local model: {MODEL}")
    st.write("AI inference runs locally through Ollama.")
    st.write("No OpenAI API or API credits are required.")
    st.info(
        "Answers to institutional questions are grounded in the indexed TCEA "
        "documents. Personal/contact-data documents are excluded."
    )

@st.cache_resource
def get_index():
    return build_index(DOCS_DIR)

try:
    index = get_index()
except Exception as e:
    st.error(f"Knowledge-base error: {e}")
    st.stop()

documents = load_documents(DOCS_DIR)
st.metric("Documents in folder", len(documents))

question = st.text_area(
    "Ask the Principal Desk",
    placeholder="Example: What is the minimum attendance required for SEE?",
    height=100,
)

def source_priority(source, question):
    """Small ranking boost for documents whose titles match the question."""
    s = source.lower()
    q = question.lower()
    boost = 0.0

    if any(x in q for x in ["attendance", "academic", "semester", "sgpa", "cgpa", "credit"]):
        if "academic_rules" in s:
            boost += 0.25
        if "student_handbook" in s:
            boost += 0.10

    if any(x in q for x in ["see", "examination", "exam", "internal assessment", "cie"]):
        if "exam_internal_assessment" in s:
            boost += 0.25
        if "academic_rules" in s:
            boost += 0.20

    if any(x in q for x in ["faculty", "teacher", "leave", "faculty rule"]):
        if "faculty_rules" in s:
            boost += 0.30

    if any(x in q for x in ["library", "book", "borrow", "fine", "no dues"]):
        if "library_rules" in s:
            boost += 0.30

    if any(x in q for x in ["health camp", "medical camp"]):
        if "health_camp" in s:
            boost += 0.30

    if any(x in q for x in ["swachhata", "cleanliness", "abhiyan"]):
        if "swachhata" in s:
            boost += 0.30

    return boost


def get_relevant_hits(question, k=5):
    raw = search_index(index, question, k=10)
    for hit in raw:
        hit["_rank_score"] = hit.get("score", 0.0) + source_priority(
            hit["source"], question
        )
    raw.sort(key=lambda x: x["_rank_score"], reverse=True)

    # Remove near-duplicate chunks from the same source where possible.
    selected = []
    source_counts = {}
    for hit in raw:
        src = hit["source"].split(" — chunk ")[0]
        source_counts.setdefault(src, 0)
        if source_counts[src] >= 2 and len(selected) < k:
            continue
        selected.append(hit)
        source_counts[src] += 1
        if len(selected) >= k:
            break
    return selected


def clean_answer(text):
    """Remove common model meta/reasoning leakage from the final answer."""
    text = text.strip()

    # Remove markdown code fences if the model adds them.
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # Remove common introductory meta lines.
    bad_prefixes = [
        r"^(okay|ok)[,!.:\s]+",
        r"^the user (has|wants|asked).{0,300}?\.\s*",
        r"^let me (analyze|scan|check|review).{0,300}?\.\s*",
        r"^first,? (i|let me).{0,300}?\.\s*",
        r"^hmm[,!.:\s]+",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in bad_prefixes:
            new = re.sub(pattern, "", text, count=1, flags=re.I | re.S)
            if new != text:
                text = new.strip()
                changed = True

    return text


def ollama_answer(question, context):
    prompt = f"""
You are the official-style Principal Desk Assistant for Techno College of
Engineering Agartala (TCEA).

TASK:
Answer the user's question using ONLY the TCEA source excerpts supplied below.

STRICT RULES:
- Give ONLY the final answer. Do not show reasoning or internal analysis.
- Do not say "the user wants", "let me analyze", "I need to", "hmm", or similar.
- Do not describe how you searched or analyzed the documents.
- Do not use general knowledge to invent an institutional rule.
- If the sources do not support the answer, say exactly:
  "The available TCEA documents do not contain enough information to answer this."
- If the source states a number, date, percentage, fee, or requirement, preserve it accurately.
- If there is an authority/currency limitation or source conflict, mention it briefly.
- Keep the answer concise: normally 2-5 sentences.
- For a simple factual question, answer in 1-3 sentences.
- Do not repeat the question.
- Do not add a generic conclusion.

USER QUESTION:
{question}

TCEA SOURCE EXCERPTS:
{context}

FINAL ANSWER ONLY:
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 180,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()

    answer = data.get("response", "").strip()
    if not answer:
        raise RuntimeError(f"Ollama returned no response: {data}")

    return clean_answer(answer)


if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching TCEA knowledge base..."):
        hits = get_relevant_hits(question, k=5)

    if not hits:
        st.warning(
            "No sufficiently relevant passage was found in the indexed TCEA documents."
        )
        st.stop()

    # Keep context compact to reduce local inference time.
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"SOURCE {i}: {hit['source']}\n{hit['text']}"
        )
    context = "\n\n".join(context_blocks)

    st.subheader("Answer")
    started = time.time()

    try:
        answer = ollama_answer(question, context)
        elapsed = time.time() - started
        st.write(answer)
        st.caption(f"Local response time: {elapsed:.1f} seconds")
    except requests.exceptions.Timeout:
        st.error(
            "The local model took longer than 10 minutes to respond. "
            "The PC may need a smaller model or shorter context."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Ollama could not be reached. Make sure Ollama is running."
        )
    except Exception as e:
        st.error(f"Ollama request failed: {e}")

    st.subheader("Sources used")
    for hit in hits:
        with st.expander(hit["source"]):
            st.write(hit["text"])
