"""The API credential, checked at the boundary and nowhere else.

One shared token rather than accounts: the callers are one kiosk and the
dashboard it serves on its own loopback, and the thing to stop is anybody else
on the network (or on the far side of a tunnel) listing faces, reading
conversations or driving the turn socket.
"""

import logging
import secrets

from fastapi import HTTPException, Request, WebSocket, status

from src.config import settings

logger = logging.getLogger(__name__)


def _supplied(connection: Request | WebSocket) -> str | None:
    header = connection.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value
    # For what a browser loads by itself — an <img src>, a download link —
    # which cannot carry a header.
    return connection.query_params.get("token")


def token_accepted(connection: Request | WebSocket) -> bool:
    expected = settings.api_access_token
    if not expected:
        return True
    supplied = _supplied(connection)
    return supplied is not None and secrets.compare_digest(supplied, expected)


async def require_api_token(request: Request) -> None:
    """Router dependency for every /api/v1 REST route."""
    if not token_accepted(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorised",
            headers={"WWW-Authenticate": "Bearer"},
        )


def warn_if_open() -> None:
    if not settings.api_access_token:
        logger.warning(
            "API_ACCESS_TOKEN is not set: every /api/v1 route and the turn socket "
            "are open to anyone who can reach this port, including the list of "
            "enrolled faces. Set it before exposing luna-api beyond this machine."
        )
