"""documents: law_intents JSON (multi-label topic tags per law)

Revision ID: 005_law_intents
Revises: 004_chat_conv
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005_law_intents"
down_revision: Union[str, None] = "004_chat_conv"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "law_intents",
            sa.JSON(),
            nullable=True,
            comment="Multi-label legal topic tags (aligned with domain classifier)",
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "law_intents")
