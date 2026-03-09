"""update message body

Revision ID: caab420f76c0
Revises: 821b83ca540f
Create Date: 2026-03-09 01:28:37.220703

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'caab420f76c0'
down_revision: Union[str, Sequence[str], None] = '821b83ca540f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename messages.content to messages.body (no new column, no nulls)."""
    op.alter_column(
        'messages',
        'content',
        new_column_name='body',
    )


def downgrade() -> None:
    """Rename messages.body back to messages.content."""
    op.alter_column(
        'messages',
        'body',
        new_column_name='content',
    )
