from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to documents
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)  # Stored filename
    original_name = Column(String(255), nullable=False)  # Original PDF name
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    file_size = Column(Integer)  # File size in bytes
    
    # Relationship to user
    user = relationship("User", back_populates="documents")