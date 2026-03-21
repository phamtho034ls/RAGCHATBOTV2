"""documents: title + file_path VARCHAR -> TEXT (title dài, đường dẫn đầy đủ)

Revision ID: v3_003
Revises: v3_002
Create Date: 2026-03-20
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "v3_003"
down_revision: Union[str, None] = "v3_002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ALTER COLUMN title TYPE TEXT")
    op.execute("ALTER TABLE documents ALTER COLUMN file_path TYPE TEXT")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents ALTER COLUMN title TYPE VARCHAR(255) "
        "USING LEFT(COALESCE(title, ''), 255)"
    )
    op.execute(
        "ALTER TABLE documents ALTER COLUMN file_path TYPE VARCHAR(255) "
        "USING LEFT(COALESCE(file_path, ''), 255)"
    )
