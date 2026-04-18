import streamlit as st
from app import graph

st.set_page_config(page_title="Fasal Mitra RAG", page_icon="🌾", layout="centered")

st.title("🌾 Fasal Mitra RAG Dashboard")
st.write("Ask agriculture questions and get context-aware guidance.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_area("Your question", placeholder="e.g., गेहूं की शुरुआती अवस्था में सिंचाई कैसे करें?", height=120)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = graph.invoke({"question": question.strip()})
                answer = response.get("answer", "No answer returned.")
                st.session_state.history.insert(0, {"question": question.strip(), "answer": answer})
            except Exception as exc:
                st.error(f"Failed to generate answer: {exc}")

if st.session_state.history:
    st.subheader("Responses")
    for item in st.session_state.history:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(item["answer"])
        st.divider()
