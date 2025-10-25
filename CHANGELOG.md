# Changelog

All notable changes to this project will be documented in this file.

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

- **Caching**
  - Redis integration
  - Cache adapter implementation

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

