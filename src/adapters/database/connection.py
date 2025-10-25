"""Database connection management."""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.adapters.database.models import Base


class DatabaseConnection:
    """Manage database connection and session lifecycle."""
    
    def __init__(self, database_url: str):
        """Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection string (async format: postgresql+asyncpg://...)
        """
        self.engine = create_async_engine(
            database_url,
            echo=False,
            poolclass=NullPool,  # Use NullPool for simpler connection management
            future=True,
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def create_tables(self):
        """Create all tables in the database."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
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
            finally:
                await session.close()
    
    async def close(self):
        """Close database connection."""
        await self.engine.dispose()

