"""Fix doc_numbers in PostgreSQL and Qdrant that were stored with trailing IDs.

Example: '45_2024_QH15_583769' → '45/2024/QH15'

Usage:
    cd backend
    ./venv/Scripts/python -m scripts.fix_doc_numbers
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update
from app.database.session import async_engine, _session_factory
from app.database.models import Document, VectorChunk

_CLEAN_RE = re.compile(r"^(\d+)[/_](\d{4})[/_]([A-Za-z][A-Za-z0-9\-]*)(?:[/_]\d+)*$")


def _normalize_doc_number(raw: str) -> str | None:
    """Extract clean doc_number: '45_2024_QH15_583769' → '45/2024/QH15'."""
    m = _CLEAN_RE.match((raw or "").strip())
    if not m:
        return None
    clean = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    if clean.replace("/", "_") == raw.replace("/", "_"):
        return None
    return clean


async def fix_all():
    async with _session_factory() as db:
        stmt = select(Document.id, Document.doc_number)
        rows = (await db.execute(stmt)).all()

        updates = []
        for doc_id, old_num in rows:
            new_num = _normalize_doc_number(old_num)
            if new_num:
                updates.append((doc_id, old_num, new_num))

        if not updates:
            print("All doc_numbers are already clean. Nothing to fix.")
            return

        print(f"Found {len(updates)} doc_numbers to fix:\n")
        for doc_id, old, new in updates:
            print(f"  [{doc_id}] {old!r} → {new!r}")

        if "--yes" not in sys.argv:
            confirm = input("\nApply fixes? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                return

        for doc_id, old_num, new_num in updates:
            await db.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(doc_number=new_num)
            )
            print(f"  DB updated: {old_num} → {new_num}")

        await db.commit()
        print(f"\nPostgreSQL: {len(updates)} documents updated.")

    # Update Qdrant payloads
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import SetPayload, Filter, FieldCondition, MatchValue
        from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        qdrant_count = 0
        for _, old_num, new_num in updates:
            try:
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload={"doc_number": new_num},
                    points=Filter(
                        must=[FieldCondition(key="doc_number", match=MatchValue(value=old_num))]
                    ),
                )
                qdrant_count += 1
                print(f"  Qdrant updated: {old_num} → {new_num}")
            except Exception as e:
                print(f"  Qdrant skip {old_num}: {e}")

        print(f"\nQdrant: {qdrant_count} payload groups updated.")
    except ImportError:
        print("\nQdrant client not available, skipping vector payload update.")
    except Exception as e:
        print(f"\nQdrant update error: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(fix_all())
