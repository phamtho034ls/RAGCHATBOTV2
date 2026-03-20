"""
Conversation Memory Store – lưu trữ lịch sử hội thoại.

Sử dụng in-memory storage cho đơn giản.
Có thể mở rộng sang Redis/PostgreSQL cho production.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional


class ConversationStore:
    """Thread-safe in-memory conversation store."""

    def __init__(self, max_conversations: int = 1000, max_messages: int = 100):
        self._conversations: Dict[str, dict] = {}
        self._lock = Lock()
        self._max_conversations = max_conversations
        self._max_messages = max_messages

    def create(self, title: Optional[str] = None) -> dict:
        """Tạo conversation mới."""
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        conv = {
            "id": conv_id,
            "title": title or f"Cuộc hội thoại {len(self._conversations) + 1}",
            "messages": [],
            "context": {},
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            # Evict oldest if at capacity
            if len(self._conversations) >= self._max_conversations:
                oldest = min(
                    self._conversations,
                    key=lambda k: self._conversations[k]["created_at"],
                )
                del self._conversations[oldest]
            self._conversations[conv_id] = conv
        return conv

    def get(self, conv_id: str) -> Optional[dict]:
        """Lấy conversation theo ID."""
        return self._conversations.get(conv_id)

    def list_all(self) -> List[dict]:
        """Liệt kê tất cả conversations (mới nhất trước)."""
        convs = sorted(
            self._conversations.values(),
            key=lambda c: c["updated_at"],
            reverse=True,
        )
        return [
            {
                "id": c["id"],
                "title": c["title"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
                "message_count": len(c["messages"]),
            }
            for c in convs
        ]

    def add_message(self, conv_id: str, role: str, content: str) -> bool:
        """Thêm message vào conversation."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return False
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            conv["messages"].append(msg)
            if len(conv["messages"]) > self._max_messages:
                conv["messages"] = conv["messages"][-self._max_messages:]
            conv["updated_at"] = msg["timestamp"]
        return True

    def get_history(self, conv_id: str, limit: int = 20) -> List[dict]:
        """Lấy lịch sử messages gần nhất."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return []
        return conv["messages"][-limit:]

    def delete(self, conv_id: str) -> bool:
        """Xóa conversation."""
        with self._lock:
            return self._conversations.pop(conv_id, None) is not None

    def update_context(self, conv_id: str, **kwargs) -> bool:
        """Lưu context retrieval cho follow-up questions."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return False
        with self._lock:
            conv.setdefault("context", {}).update(kwargs)
            conv["updated_at"] = datetime.now().isoformat()
        return True

    def update_document_context(self, conv_id: str, doc_meta: dict) -> bool:
        """Lưu metadata văn bản đã tạo/tham chiếu để hỗ trợ câu hỏi follow-up.

        Args:
            conv_id: ID conversation.
            doc_meta: Dict chứa metadata văn bản, ví dụ:
                {
                    "loai_van_ban": "ke_hoach",
                    "linh_vuc": "le_hoi",
                    "co_quan": "UBND_xa",
                    "chu_de": "quản lý lễ hội",
                    "noi_dung_tom_tat": "...",
                }
        """
        conv = self._conversations.get(conv_id)
        if not conv:
            return False
        with self._lock:
            conv.setdefault("context", {})
            # Lưu danh sách document context (giữ lại 5 văn bản gần nhất)
            doc_history = conv["context"].setdefault("document_history", [])
            doc_history.append({
                **doc_meta,
                "timestamp": datetime.now().isoformat(),
            })
            if len(doc_history) > 5:
                conv["context"]["document_history"] = doc_history[-5:]
            # Cập nhật last_document cho quick access
            conv["context"]["last_document"] = doc_meta
            conv["updated_at"] = datetime.now().isoformat()
        return True

    def get_last_document_context(self, conv_id: str) -> Optional[dict]:
        """Lấy metadata văn bản cuối cùng đã tạo/tham chiếu."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return None
        return conv.get("context", {}).get("last_document")

    def get_context(self, conv_id: str) -> dict:
        """Lấy context đã lưu của conversation."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return {}
        return conv.get("context", {})


# ── Singleton ──────────────────────────────────────────────
conversation_store = ConversationStore()
