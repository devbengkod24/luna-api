# Database Schema

## Overview

Aplikasi ini menggunakan **PostgreSQL** sebagai relational database dan **Qdrant** sebagai vector database untuk RAG.

## PostgreSQL Schema

### Table: `chat_sessions`

Menyimpan session percakapan.

| Column      | Type          | Constraints                    | Description                    |
|-------------|---------------|--------------------------------|--------------------------------|
| id          | INTEGER       | PRIMARY KEY, AUTO_INCREMENT    | Session ID                     |
| uuid        | VARCHAR(36)   | UNIQUE, NOT NULL, INDEX        | Session UUID (untuk API)       |
| title       | VARCHAR(255)  | NOT NULL                       | Judul session                  |
| created_at  | TIMESTAMP     | NOT NULL, DEFAULT NOW()        | Waktu dibuat                   |
| updated_at  | TIMESTAMP     | NOT NULL, DEFAULT NOW()        | Waktu terakhir diupdate        |

**Relationships:**
- Has many `messages` (one-to-many)

**Indexes:**
- `id` (PRIMARY KEY)
- `uuid` (UNIQUE)

---

### Table: `messages`

Menyimpan pesan dalam session.

| Column      | Type          | Constraints                    | Description                    |
|-------------|---------------|--------------------------------|--------------------------------|
| id          | INTEGER       | PRIMARY KEY, AUTO_INCREMENT    | Message ID                     |
| session_id  | INTEGER       | FOREIGN KEY → chat_sessions.id | Session yang memiliki message  |
| role        | VARCHAR(20)   | NOT NULL                       | 'user' atau 'assistant'        |
| content     | TEXT          | NOT NULL                       | Isi pesan                      |
| context     | JSON          | NULL                           | Konteks (emotion, name, dll)   |
| created_at  | TIMESTAMP     | NOT NULL, DEFAULT NOW()        | Waktu dibuat                   |

**Relationships:**
- Belongs to `chat_sessions` (many-to-one)

**Indexes:**
- `id` (PRIMARY KEY)
- `session_id` (FOREIGN KEY, INDEX)

**Foreign Keys:**
- `session_id` → `chat_sessions.id` (ON DELETE CASCADE)

---

## Qdrant Collections

### Collection: `knowledge_base`

Menyimpan embeddings dokumen untuk RAG (Retrieval Augmented Generation).

**Vector Configuration:**
- **Dimension**: 384 (sesuaikan dengan model embedding Anda)
- **Distance**: Cosine

**Payload Schema:**
```json
{
  "text": "string",           // Chunk text asli
  "metadata": {
    "source": "string",       // Sumber dokumen
    "page": "number",         // Halaman (jika applicable)
    "timestamp": "string"     // Waktu upload
  }
}
```

---

## Redis Keys

Redis digunakan untuk caching dan session storage.

### Cache Keys

| Key Pattern                  | Type   | TTL    | Description                    |
|------------------------------|--------|--------|--------------------------------|
| `latest_vision_result`       | String | 60s    | Latest vision analysis result  |
| `session:{uuid}:context`     | String | 3600s  | User context cache per session |
| `user:{name}:face_encoding`  | String | -      | Face encoding untuk recognition|

---

## Entity Relationship Diagram (ERD)

```
┌─────────────────────┐
│   chat_sessions     │
├─────────────────────┤
│ id (PK)             │
│ uuid (UNIQUE)       │
│ title               │
│ created_at          │
│ updated_at          │
└──────────┬──────────┘
           │
           │ 1:N
           │
┌──────────▼──────────┐
│     messages        │
├─────────────────────┤
│ id (PK)             │
│ session_id (FK)     │
│ role                │
│ content             │
│ context (JSON)      │
│ created_at          │
└─────────────────────┘
```

---

## Database Migrations

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description"

# Create empty migration
alembic revision -m "description"
```

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade <revision_id>

# Downgrade one step
alembic downgrade -1

# Downgrade to base (WARNING: deletes all data)
alembic downgrade base
```

### Migration History

```bash
# Show current revision
alembic current

# Show migration history
alembic history

# Show SQL without running
alembic upgrade head --sql
```

---

## Sample Queries

### Get chat session with messages

```sql
SELECT 
    cs.id,
    cs.uuid,
    cs.title,
    cs.created_at,
    cs.updated_at,
    json_agg(
        json_build_object(
            'role', m.role,
            'content', m.content,
            'context', m.context,
            'created_at', m.created_at
        ) ORDER BY m.created_at ASC
    ) as messages
FROM chat_sessions cs
LEFT JOIN messages m ON m.session_id = cs.id
WHERE cs.uuid = 'your-uuid-here'
GROUP BY cs.id;
```

### Get recent sessions

```sql
SELECT 
    id,
    uuid,
    title,
    created_at,
    updated_at
FROM chat_sessions
ORDER BY updated_at DESC
LIMIT 10;
```

### Get message count per session

```sql
SELECT 
    cs.uuid,
    cs.title,
    COUNT(m.id) as message_count
FROM chat_sessions cs
LEFT JOIN messages m ON m.session_id = cs.id
GROUP BY cs.id, cs.uuid, cs.title
ORDER BY message_count DESC;
```

---

## Performance Considerations

1. **Indexes**: 
   - `chat_sessions.uuid` untuk lookup cepat by UUID
   - `messages.session_id` untuk join performance

2. **Cascade Delete**: 
   - Messages akan otomatis terhapus saat session dihapus

3. **JSON Context**: 
   - Kolom `context` menggunakan JSON untuk flexibility
   - Bisa ditambahkan GIN index jika perlu query JSON

4. **Pagination**: 
   - Gunakan LIMIT/OFFSET untuk list sessions
   - Gunakan cursor-based pagination untuk large datasets

---

## Backup & Restore

### Backup

```bash
# Backup all databases
docker-compose exec postgres pg_dumpall -U vsa_user > backup.sql

# Backup specific database
docker-compose exec postgres pg_dump -U vsa_user vsa_db > backup_vsa.sql
```

### Restore

```bash
# Restore all
docker-compose exec -T postgres psql -U vsa_user < backup.sql

# Restore specific
docker-compose exec -T postgres psql -U vsa_user -d vsa_db < backup_vsa.sql
```

