"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import String, DateTime, Text, ForeignKey, BigInteger
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from uuid_extensions import uuid7
from sqlalchemy import Integer
Base = declarative_base()


class ConversationModel(Base):
    """Conversation model - stores conversation sessions."""
    
    __tablename__ = "conversations"
    
    id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel", back_populates="conversation",
        cascade="all, delete-orphan", passive_deletes=True,
    )

class MessageModel(Base):
    """Message model - stores individual messages in a chat session."""
    
    __tablename__ = "messages"
    
    id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)
    conversation_id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    character: Mapped[int] = mapped_column(Integer, nullable=False)
    body: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    context: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    conversation: Mapped["ConversationModel"] = relationship("ConversationModel", back_populates="messages")
    

class FileModel(Base):
    """File ORM model."""

    __tablename__ = "files"

    id: Mapped[uuid7] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="processing")
    tags: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

