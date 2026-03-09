from pydantic import BaseModel, Field

from src.drivers.rest.schemas.response_schemas import UserContextSchema


class ConversationResponse(BaseModel):
    """Single conversation item."""
    id: str
    title: str
    created_at: str
    updated_at: str


class ListConversationsPaginatedResponse(BaseModel):
    """Cursor-paginated list of conversations."""
    items: list[ConversationResponse]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page. Omit or null when no more pages.",
    )


class UpdateConversationRequest(BaseModel):
    """Request model for updating a conversation."""
    title: str = Field(..., min_length=1, max_length=255, description="New conversation title")


class MessageResponse(BaseModel):
    """Single message in a list."""
    id: str
    character: int
    body: dict = Field(default_factory=dict, description="Message body (type, content, etc.)")
    context: UserContextSchema | None = None
    created_at: str


class ListMessagesPaginatedResponse(BaseModel):
    """Cursor-paginated list of messages for a conversation."""
    items: list[MessageResponse]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page. Omit or null when no more pages.",
    )
