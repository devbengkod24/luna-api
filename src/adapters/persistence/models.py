"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from uuid_extensions import uuid7

Base = declarative_base()


class ChatSessionModel(Base):
    """Chat session model - stores conversation sessions."""
    
    __tablename__ = "chat_sessions"
    
    id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    messages: Mapped[list["MessageModel"]] = relationship("MessageModel", back_populates="session")

class MessageModel(Base):
    """Message model - stores individual messages in a chat session."""
    
    __tablename__ = "messages"
    
    id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)
    session_id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    base64_audio: Mapped[str] = mapped_column(Text, nullable=True)
    character: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(JSONB, nullable=False, default=dict)
    context: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    session: Mapped["ChatSessionModel"] = relationship("ChatSessionModel", back_populates="messages")
    

