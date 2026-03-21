"""V3: unaccent + pg_trgm + articles.search_vector GIN

Revision ID: v3_001
Revises:
Create Date: 2026-03-20
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "v3_001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        ALTER TABLE articles ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
          setweight(to_tsvector('simple', unaccent(coalesce(title,''))), 'A') ||
          setweight(to_tsvector('simple', unaccent(coalesce(content,''))), 'B')
        ) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_articles_search_vector
        ON articles USING GIN(search_vector)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_search_vector")
    op.execute("ALTER TABLE articles DROP COLUMN IF EXISTS search_vector")
