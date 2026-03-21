"""documents.issuer: VARCHAR(255) -> TEXT (trích yếu có thể dài)

Revision ID: v3_002
Revises: v3_001
Create Date: 2026-03-20
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "v3_002"
down_revision: Union[str, None] = "v3_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ALTER COLUMN issuer TYPE TEXT")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents ALTER COLUMN issuer TYPE VARCHAR(255) "
        "USING LEFT(COALESCE(issuer, ''), 255)"
    )
