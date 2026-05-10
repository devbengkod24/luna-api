"""Add indexes for messages conversation history.

Revision ID: b2d3e4f5a6b7
Revises: caab420f76c0
Create Date: 2026-05-09

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b2d3e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "caab420f76c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "idx_messages_conversation_id",
        "messages",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        "idx_messages_conversation_id_created_at",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_messages_conversation_id_created_at", table_name="messages")
    op.drop_index("idx_messages_conversation_id", table_name="messages")

