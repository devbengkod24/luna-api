# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Text-to-Speech (TTS) Integration**
  - Facebook MMS-TTS Indonesian adapter implementation (facebook/mms-tts-ind)
  - VITS model architecture for high-quality Indonesian speech synthesis
  - Auto-detect CUDA/GPU for optimal performance
  - Integrated with chat audio endpoint (`/v1/chat/audio`)
  - New SSE event type: `audio` for TTS response
  - Configuration: `voice_clone_dir` setting (reserved for future use)
  - Storage directory: `storage/voice/` (reserved for future voice cloning)

### Changed
- Migrated from Chatterbox TTS to Facebook MMS-TTS Indonesian model
- Updated `ChatSpeechUseCase` to support TTS synthesis
- Added `enable_tts` parameter to `ChatAudioRequest` schema
- Enhanced chat audio endpoint with TTS response streaming
- Updated dependencies: added `scipy==1.15.1` for audio processing
- Removed voice cloning functionality (reserved for future implementation)

### Documentation
- Added `docs/TTS_SETUP.md` - Complete TTS setup guide
- Added `docs/TTS_API_EXAMPLES.md` - API usage examples with curl and Python
- Added `storage/voice/README.md` - Voice cloning guide
- Added `scripts/test_tts.py` - TTS testing script

## [2.0.0] - 2025-11-01

### Added
- **Clean Architecture Implementation**
  - Domain layer with entities
  - Ports layer with interfaces
  - Adapters layer with implementations
  - Use cases layer for business logic
  - Drivers layer for external interfaces

- **Database Layer**
  - PostgreSQL with SQLAlchemy 2.0 (async)
  - Alembic for database migrations
  - Chat sessions and messages models
  - Repository pattern implementation

- **Vector Database**
  - Qdrant integration for RAG
  - RAG adapter template

- **API Endpoints**
  - WebSocket: `/v1/vision/stream` for real-time vision
  - POST: `/v1/chat/text` for text chat with SSE streaming
  - POST: `/v1/chat/audio` for audio chat with SSE streaming
  - GET: `/v1/chat/history` for chat history list
  - GET: `/v1/chat/history/{uuid}` for session detail

- **Adapters (Templates)**
  - LLM adapter for language model integration
  - Vision adapter for emotion detection + face recognition
  - STT adapter for speech-to-text
  - TTS adapter for text-to-speech

- **DevOps**
  - Docker Compose setup with all services
  - Dockerfile for containerization
  - Makefile for common tasks
  - Shell scripts for database operations

- **Documentation**
  - Comprehensive README with API specs
  - Database schema documentation
  - Code structure explanation

### Changed
- Migrated from monolithic to Clean Architecture
- Updated dependencies to latest versions

### Deprecated
- Old monolithic structure (app/)

## [1.0.0] - Previous Version

Initial monolithic implementation.

