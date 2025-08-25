# main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import shutil
from openai import OpenAI
import dotenv
from sqlalchemy.orm import Session
from database import get_db
from models import User, Document
from auth import authenticate_user, create_access_token, verify_token, get_password_hash
from document_service import DocumentProcessor
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
# Note: FAISS index is now loaded dynamically by DocumentProcessor when needed
print("✅ PDF Search API initialized - FAISS index will be loaded on first use")

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
    
    # Verify the user has access to the specified document
    if request.pdf_id:
        try:
            document_id = int(request.pdf_id)
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.user_id == user.id
            ).first()
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this document"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID"
            )
    
    # Use DocumentProcessor for search
    try:
        processor = DocumentProcessor()
        
        # Search within specific document if pdf_id is provided
        if request.pdf_id:
            search_results = processor.search(request.query, document_id=int(request.pdf_id))
        else:
            # Search across all user's documents
            search_results = processor.search(request.query)
            # Filter results to only include user's documents
            user_document_ids = [doc.id for doc in user.documents]
            search_results = [result for result in search_results 
                           if any(doc_id in result.get('pdf_name', '') for doc_id in user_document_ids)]
        
        if not search_results:
            return {
                "query": request.query,
                "answer": "No relevant information found in your documents.",
                "retrieved_context": []
            }
        
        # Extract context from search results
        retrieved_chunks = [result['content'] for result in search_results]
        
        # Generate answer using OpenAI
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

        return {
            "query": request.query,
            "answer": answer.strip(),
            "retrieved_context": retrieved_chunks,
            "search_results": search_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.get("/documents")
async def list_user_documents(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """List all documents uploaded by the authenticated user."""
    
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
    
    # Get user's documents
    documents = db.query(Document).filter(Document.user_id == user.id).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.original_name,
                "upload_date": doc.upload_date,
                "file_size": doc.file_size
            }
            for doc in documents
        ]
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Upload a PDF document and process it for search."""
    
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
    
    # Validate file type (allow PDF and text files for testing)
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are allowed"
        )
    
    # Validate file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be less than 10MB"
        )
    
    try:
        # Create unique filename
        import uuid
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Save file to uploads directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create document record in database
        document = Document(
            filename=unique_filename,
            original_name=file.filename,
            user_id=user.id,
            file_size=file_size
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process the document and update search index
        try:
            processor = DocumentProcessor()
            processing_result = processor.process_document(file_path, document.id, document.original_name)
            
            return {
                "message": "Document uploaded and processed successfully",
                "document_id": document.id,
                "filename": document.original_name,
                "status": "processed",
                "processing_result": processing_result
            }
        except Exception as processing_error:
            # If processing fails, delete the document record and file
            db.delete(document)
            db.commit()
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(processing_error)}"
            )
        
    except Exception as e:
        # Clean up file if database operation fails
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)