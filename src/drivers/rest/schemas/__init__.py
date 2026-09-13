"""Schemas for REST routers."""

from .response_schemas import (
    ResponseTextRequest,
    ResponseAudioRequest,
    UserContextSchema,
)
from .conversation_schemas import (
    ConversationResponse,
    ListConversationsPaginatedResponse,
    UpdateConversationRequest,
    MessageResponse,
    ListMessagesPaginatedResponse,
)
from .face_schema import (
    RegisterFaceItem,
    RegisterFaceRequest,
    RegisterFacesResponse,
    SingleRegisterFaceResult,
    FaceResponse,
    ListFacesResponse,
    MatchFaceRequest,
    MatchFaceResponse,
)

__all__ = [
    "ResponseTextRequest",
    "ResponseAudioRequest",
    "UserContextSchema",
    "ConversationResponse",
    "ListConversationsPaginatedResponse",
    "UpdateConversationRequest",
    "MessageResponse",
    "ListMessagesPaginatedResponse",
    "RegisterFaceItem",
    "RegisterFaceRequest",
    "RegisterFacesResponse",
    "SingleRegisterFaceResult",
    "FaceResponse",
    "ListFacesResponse",
    "MatchFaceRequest",
    "MatchFaceResponse",
]
