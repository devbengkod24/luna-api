"""Database connection management."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.adapters.persistence.models import Base
from src.config import settings


class DatabaseConnection:
    """Manage database connection and session lifecycle."""

    def __init__(self, url: str):
        """Initialize database connection.

        Args:
            url: PostgreSQL connection string (async format: postgresql+asyncpg://...)
        """
        self.engine = create_async_engine(
            url,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            future=True,
            echo=False,
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self):
        """Initialize the database."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def reset_db(self):
        """Drop all tables from the database."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session.

        Yields:
            AsyncSession: SQLAlchemy async session
        """
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self):
        """Close database connection."""
        await self.engine.dispose()
