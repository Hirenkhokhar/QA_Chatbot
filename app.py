from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
GOOGLE_GEMINI_API = os.getenv("GOOGLE_GEMINI_API")

app = FastAPI()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = Client(Settings(persist_directory="db"))
llm = GoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_GEMINI_API)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class InsertRequest(BaseModel):
    document: str
    metadata: dict
    id: str

class UpdateRequest(BaseModel):
    id: str
    new_document: str = None
    new_metadata: dict = None

class DeleteRequest(BaseModel):
    id: str

@app.post("/query")
async def query_chromadb(request: QueryRequest):
    question = request.question
    top_k = request.top_k

    question_embedding = embedding_model.encode(question).tolist()


    results = chroma_client.get_or_create_collection(name="export_query_data_with_numericals").query(
        query_embeddings=[question_embedding], n_results=top_k
    )


    retrieved_context = "\n".join([
        f"Document: {doc}\nMetadata: {meta}" \
        for doc, meta in zip(results['documents'], results['metadatas'])
    ])

    llm_prompt = f"Answer the question based on the following context:\n{retrieved_context}\n\nQuestion: {question}\nPlease provide a detailed answer:"
    response = llm.invoke(llm_prompt)

    return {"question": question, "answer": response.strip()}

@app.post("/insert")
async def insert_document(request: InsertRequest):
    collection = chroma_client.get_or_create_collection(name="export_query_data_with_numericals")

  
    collection.add(
        documents=[request.document],
        metadatas=[request.metadata],
        ids=[request.id]
    )
    return {"message": "Document inserted successfully", "id": request.id}

@app.put("/update")
async def update_document(request: UpdateRequest):
    collection = chroma_client.get_or_create_collection(name="export_query_data_with_numericals")


    existing_docs = collection.get(ids=[request.id])
    if len(existing_docs['ids']) == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    if request.new_document:
        collection.update(
            documents=[request.new_document],
            ids=[request.id]
        )
    if request.new_metadata:
        collection.update(
            metadatas=[request.new_metadata],
            ids=[request.id]
        )

    return {"message": "Document updated successfully", "id": request.id}

@app.delete("/delete")
async def delete_document(request: DeleteRequest):
    collection = chroma_client.get_or_create_collection(name="export_query_data_with_numericals")


    existing_docs = collection.get(ids=[request.id])
    if len(existing_docs['ids']) == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    collection.delete(ids=[request.id])

    return {"message": "Document deleted successfully", "id": request.id}

