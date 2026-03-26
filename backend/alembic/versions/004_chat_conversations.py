"""chat_conversations + chat_messages for persistent UI sessions."""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_chat_conv"
down_revision: Union[str, None] = "v3_003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index("idx_chat_conv_updated", "chat_conversations", ["updated_at"])
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("conversation_id", sa.String(32), sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index("idx_chat_msg_conv_id", "chat_messages", ["conversation_id", "id"])


def downgrade() -> None:
    op.drop_index("idx_chat_msg_conv_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("idx_chat_conv_updated", table_name="chat_conversations")
    op.drop_table("chat_conversations")
