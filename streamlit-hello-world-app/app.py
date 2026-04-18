from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, TypedDict
import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your Hugging Face API token as environment variable: HF_TOKEN
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Please set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN with your Hugging Face API token")

hf_models = [
    os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-R1:fastest"),
    "openai/gpt-oss-120b:fastest",
]
hf_chat_url = "https://router.huggingface.co/v1/chat/completions"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index at startup from source document
def build_vector_store():
    """Build FAISS vector store from source document at startup"""
    print("Building FAISS index from source document...")
    
    # Read the source document
    doc_path = os.path.join(os.path.dirname(__file__), "rag_short.txt")
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    # Create documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Build FAISS index
    vector_store = FAISS.from_documents(documents, embeddings)
    
    print(f"FAISS index built with {len(documents)} chunks")
    return vector_store

# Build the vector store at module load time
vector_store = build_vector_store()

prompt = ChatPromptTemplate.from_template(
"""
You are Fasal Mitra, a practical farming support assistant.

Your goal is to help farmers with any agriculture-related support, including crop planning,
seed selection, pest and disease issues, irrigation, fertilizer use, soil health,
weather-related decisions, market selling guidance, storage, and government scheme awareness.

Rules:
1) Use ONLY the provided context for factual claims.
2) If the context is missing details, clearly say what is missing instead of guessing.
3) Keep language simple, clear, and actionable for farmers.
4) Reply in the same language as the user's question when possible.
5) Give step-by-step recommendations with approximate timelines (today/this week/next stage) when relevant.
6) If the issue can seriously harm crop/livestock (severe disease, poisoning, heavy losses), add a short urgent note to contact a local agriculture officer/KVK.

Response format:
- Short answer (1-2 lines)
- What to do now (numbered steps)
- Preventive tips (2-4 bullets)
- If unsure / missing data: ask up to 3 specific follow-up questions (e.g., crop stage, location, soil type, symptoms)

Question:
{question}

Context:
{context}
""".strip()
)

# State definition
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def call_hf_inference(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    errors = []
    for model in hf_models:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "max_tokens": 1500,
            "temperature": 0,
            "stream": False,
        }

        try:
            response = requests.post(hf_chat_url, headers=headers, json=payload, timeout=120)
            if response.status_code in (404, 503):
                errors.append(f"{model}: HTTP {response.status_code}")
                continue
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return str(content).strip()
                if "error" in data:
                    errors.append(f"{model}: {data['error']}")
                    continue
            errors.append(f"{model}: unexpected response format")
        except requests.RequestException as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError("Hugging Face inference failed for all models: " + " | ".join(errors))

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_value = prompt.invoke({"question": state["question"], "context": docs_content})
    response = call_hf_inference(prompt_value.to_string())
    return {"answer": response}

# Compile graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# FastAPI application
app = FastAPI()

# Request and response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        response = graph.invoke({"question": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
