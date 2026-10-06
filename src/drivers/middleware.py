"""ASGI middleware that has to run before FastAPI touches a request."""

import json

from src.config import settings


class BodySizeLimitMiddleware:
    """Refuse a request body larger than `max_request_body_bytes`.

    Before parsing, not after: FastAPI reads a JSON body whole into memory to
    validate it, so the per-field limits in the schemas (10 MB per face image,
    100 faces) only ran once a body of up to a gigabyte was already in memory.
    A declared Content-Length is checked up front; a chunked body is counted
    as it arrives.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = settings.max_request_body_bytes
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._refuse(send, limit)
            return

        received = 0
        started = False
        refused = False

        async def counted_receive():
            nonlocal received, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # Answered here and now. Raising instead lets the framework
                    # catch it as a malformed body and answer 400; telling the
                    # app the client left makes it stop reading, and whatever it
                    # tries to send afterwards is dropped below.
                    refused = True
                    if not started:
                        await self._refuse(send, limit)
                    return {"type": "http.disconnect"}
            return message

        async def tracked_send(message):
            nonlocal started
            if refused:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, counted_receive, tracked_send)

    @staticmethod
    async def _refuse(send, limit: int) -> None:
        body = json.dumps(
            {"detail": f"Request body too large (max {limit // (1024 * 1024)} MB)"}
        ).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
