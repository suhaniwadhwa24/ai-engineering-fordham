import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load artifacts (cached so they only load once)
@st.cache_resource
def load_rag_system():
    embeddings = np.load("embeddings.npy")
    with open("chunks.json") as f:
        chunks = json.load(f)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embeddings, chunks, model

embeddings, all_chunks, model = load_rag_system()

# RAG functions 
def retrieve(question, top_k=5):
    q_vec = model.encode([question])
    sims = cosine_similarity(q_vec, embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [{"chunk": all_chunks[i], "similarity": float(sims[i])} for i in top_indices]


def rewrite_query(question, history):
    if not history:
        return question
    recent = history[-3:]
    history_text = "\n".join([f"User: {t['question']}\nAssistant: {t['answer']}" for t in recent])
    prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question to be fully self-contained (no pronouns like "it", "they", "there", "those"):
Follow-up question: {question}

Return ONLY the rewritten question, nothing else."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def generate(question, chunks, history):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}: {r['chunk']['doc_url']}]\n{r['chunk']['text']}"
        for i, r in enumerate(chunks)
    )
    system = """You are a helpful Fordham University assistant.
Answer using only the provided context.
Use conversation history to understand follow-up questions.
If the context doesn't contain the answer, say so honestly."""

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600
    )
    return response.choices[0].message.content


# Streamlit UI 
st.set_page_config(page_title="Fordham RAG", page_icon="🎓", layout="centered")
st.title("🎓 Fordham University Q&A")
st.caption("Ask anything about Fordham. Follow-up questions supported!")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "chat" not in st.session_state:
    st.session_state.chat = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 3, 10, 5)
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.chat = []
        st.rerun()
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Your question is rewritten if it's a follow-up")
    st.markdown("2. Relevant chunks are retrieved from 9,500 Fordham pages")
    st.markdown("3. GPT-4o-mini generates an answer using those chunks")

# Show chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📄 View sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['url']}]({s['url']}) — score: `{s['score']:.3f}`")

# Chat input
if question := st.chat_input("Ask about Fordham University..."):

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat.append({"role": "user", "content": question})

    # Run RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Rewrite query if follow-up
            search_query = rewrite_query(question, st.session_state.history)

            # Show rewritten query if different
            if search_query != question:
                st.caption(f"🔁 Searching for: *{search_query}*")

            # Retrieve and generate
            chunks = retrieve(search_query, top_k=top_k)
            answer = generate(question, chunks, st.session_state.history)
            sources = [{"url": r["chunk"]["doc_url"], "score": r["similarity"]} for r in chunks]

        st.write(answer)
        with st.expander("📄 View sources"):
            for s in sources:
                st.markdown(f"- [{s['url']}]({s['url']}) — score: `{s['score']:.3f}`")

    # Save to history
    st.session_state.history.append({"question": question, "answer": answer})
    if len(st.session_state.history) > 5:
        st.session_state.history.pop(0)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })