---
name: boundary-audit
description: Verify that REST and WebSocket routes enforce the boundary size limits and allowed-extension lists documented in CLAUDE.md (rag_max_file_size, chat_max_audio_base64_size, face_max_image_size, face_max_batch_size, chat_max_prompt_length, rag_allowed_extensions). Use when adding or editing an endpoint that accepts uploads, audio, images, or free-form text, or when the user asks to audit input limits.
---

# Boundary Audit

Enforces the input-size and content-type guards listed in `CLAUDE.md → Repo-specific gotchas`. These limits must be checked at the router or Pydantic schema layer, **not** in use cases — by the time a request reaches a use case, it's too late to reject cheaply.

## The limits (single source of truth: `src/config.py`)

| Setting | What it guards | Where to enforce |
|---|---|---|
| `rag_max_file_size` (50 MB) | RAG file uploads | RAG upload router / schema |
| `rag_allowed_extensions` | RAG file types | RAG upload router |
| `chat_max_audio_base64_size` (10 MB) | Audio in chat payloads (REST + WS) | Chat router + WS handler |
| `chat_max_prompt_length` (10000) | Free-form prompt text | Chat schema |
| `face_max_image_size` (10 MB) | Face recognition image | Face router / schema |
| `face_max_batch_size` (100) | Batch face requests | Face router / schema |

## Audit procedure

When invoked, do this:

1. **Locate the route(s) in question.** If the user named an endpoint, find it in `src/drivers/rest/routers/` or `src/drivers/realtime/routers/`. If they're asking for a sweep, list all routers and process each.

2. **For each input field, identify the matching limit from the table above.** Audio → `chat_max_audio_base64_size`. Image bytes → `face_max_image_size` or `rag_max_file_size` depending on context. Free-form text → `chat_max_prompt_length`. Upload file → `rag_max_file_size` + `rag_allowed_extensions`.

3. **Verify the guard exists.** Acceptable forms:
   - Pydantic schema field with `max_length` / `Field(..., max_length=N)` for strings.
   - `UploadFile` size check against `settings.rag_max_file_size` before reading the body.
   - Explicit `len(base64_str) > settings.chat_max_audio_base64_size` check in the router, raising `HTTPException(413)` or sending a WS error envelope.
   - Extension check against `settings.rag_allowed_extensions`.

4. **Reject these anti-patterns:**
   - Reading the full body into memory before checking size.
   - Using a hardcoded literal (e.g., `10 * 1024 * 1024`) instead of `settings.*` — drift risk.
   - Putting the size check inside a use case (`src/use_cases/`) — boundary is the router.
   - WebSocket handlers that accept arbitrary-length payloads with no size check at all (common gap; WS doesn't get the HTTP body-size middleware).

5. **Report missing or weak guards** with file:line references. For each, propose the minimal patch (schema field, router check, or WS envelope rejection) — don't write the code unless the user asks.

## Output format

```
Endpoint: POST /chat/voice  (src/drivers/rest/routers/chat.py:42)
  ✓ chat_max_prompt_length     (schema: ChatRequest.prompt, max_length=10000)
  ✗ chat_max_audio_base64_size MISSING — audio_b64 has no size check before decoding
  → Suggest: add `@field_validator("audio_b64")` in ChatRequest comparing len() to settings.chat_max_audio_base64_size
```

Be explicit about which limit is missing and where the fix belongs. Don't propose adding a new limit — work only from the documented set.
