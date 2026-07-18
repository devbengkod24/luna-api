---
name: alembic-revision
description: Create, review, and apply an Alembic migration for the VSA backend. Wraps the docker-exec workflow with a checklist that guards the sync/async URL split, the "never edit an applied revision" rule, and the FUSE-mode tmpfs gotcha. Use when the user asks to add, generate, autogenerate, or apply a migration.
disable-model-invocation: true
---

# Alembic Revision

A guided workflow for creating an Alembic migration in the VSA backend. Migrations are the highest-risk operation in this repo, so this skill enforces a checklist rather than running things blindly.

## Preconditions to check first

Run these in order, stop if any fail:

1. **Stack is up:** `docker compose ps fastapi` — the `fastapi` container must be running. If not, prompt the user to bring it up (`docker compose up -d`) before continuing.
2. **HEAD is consistent:** `docker compose exec fastapi alembic current` — note the current revision. If it says `(head)` you're aligned; if it lags, ask the user whether to `alembic upgrade head` first.
3. **No uncommitted changes in `alembic/versions/`:** `git -C VSA-BE-FastAPIv2 status alembic/versions/` — if there are unstaged migrations, surface them and ask before generating a new one. **Never edit a migration that has already been applied.**

## Workflow

Ask the user for the migration message if not provided. Then:

```bash
docker compose exec fastapi alembic revision --autogenerate -m "<message>"
```

The new file appears on the host under `alembic/versions/` (volume-mounted from the container).

**Review the generated file before applying.** Read it and verify:

- The diff matches the SQLAlchemy model change the user described. Autogenerate misses: server defaults, enum changes, index name normalization, custom types.
- No unintended drops of columns / tables / indexes. If you see `op.drop_*` calls that the user did not ask for, stop and confirm.
- Both `upgrade()` and `downgrade()` are populated and symmetric.
- If the migration adds a `NOT NULL` column to an existing table, it must have a `server_default` or a two-step plan (add nullable → backfill → set NOT NULL in a follow-up revision). Flag this to the user.

## Apply

After review:

```bash
docker compose exec fastapi alembic upgrade head
```

Then sanity-check:

```bash
docker compose exec fastapi alembic current
```

## Gotchas (from CLAUDE.md)

- Alembic uses `database_url_sync` (psycopg2); the app uses `database_url` (asyncpg). If a migration fails with a driver mismatch, the env config is wrong — don't try to "fix" it inside the migration file.
- In FUSE mode (`docker-compose.fuse.yml`), Postgres is on tmpfs — data is ephemeral. A migration applied in FUSE mode is gone after `docker compose down`. Warn the user if FUSE compose is active.
- `alembic/versions/` is bind-mounted, so the new file appears on the host immediately — but the user still has to `git add` it.

## Downgrade (only if asked)

```bash
docker compose exec fastapi alembic downgrade -1
```

Confirm with the user first. If the migration has been pushed/shared, prefer creating a new corrective revision instead of downgrading.
