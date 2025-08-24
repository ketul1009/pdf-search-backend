# main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# --- Configuration ---
# In a real app, use environment variables for secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace with your key
client = OpenAI(api_key=OPENAI_API_KEY)

# --- App Initialization ---
app = FastAPI(title="PDF Search API")

# --- Models (Data Schemas) ---
class SearchQuery(BaseModel):
    query: str
    pdf_id: str # To scope search to a specific PDF
    
class Token(BaseModel):
    access_token: str
    token_type: str

# --- Security (Dummy User DB & JWT) ---
# For demo purposes, we have a hardcoded user.
# In a real app, this would be a database.
DUMMY_USERS_DB = {"testuser": "testpassword"}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(username: str, password: str):
    if username in DUMMY_USERS_DB and DUMMY_USERS_DB[username] == password:
        return True
    return False

# --- Load Search Index and Model (at startup) ---
try:
    index = faiss.read_index("pdf_index.faiss")
    with open("metadata.json", 'r') as f:
        metadata = json.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except FileNotFoundError:
    print("ERROR: Run 'ingestion.py' first to create the index files.")
    exit()

# --- API Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # In a real app, you would create a proper JWT token here.
    # For simplicity, we'll use a placeholder token.
    return {"access_token": form_data.username, "token_type": "bearer"}

@app.post("/search")
async def search_pdf(request: SearchQuery, token: str = Depends(oauth2_scheme)):
    # 1. Embed the user's query
    query_embedding = model.encode([request.query])[0].astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0) # Reshape for FAISS

    # 2. Perform Similarity Search (Retrieval)
    k = 3 # Number of relevant chunks to retrieve
    distances, indices = index.search(query_embedding, k)
    
    retrieved_chunks = [metadata[i]['content'] for i in indices[0]]
    
    # 3. Augment and Generate
    try:
        prompt = f"""
        Based on the following context from a document, please answer the user's question.
        If the context does not contain the answer, say 'The document does not provide information on this topic.'

        Context:
        - {"\n- ".join(retrieved_chunks)}

        Question: {request.query}

        Answer:
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting LLM: {str(e)}")

    return {"query": request.query, "answer": answer.strip(), "retrieved_context": retrieved_chunks}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)