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
from sqlalchemy.orm import Session
from database import get_db
from models import User
from auth import authenticate_user, create_access_token, verify_token, get_password_hash
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

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# --- Security ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=Token)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Return token for immediate login
    access_token = create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/search")
async def search_pdf(
    request: SearchQuery, 
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    # Verify the token and get the username
    username = verify_token(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify user exists in database
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
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