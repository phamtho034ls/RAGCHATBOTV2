import React, { useState, useEffect } from "react";
import {
  Plus,
  ChevronLeft,
  ChevronRight,
  Settings,
  MessageSquare,
  Trash2,
} from "lucide-react";
import {
  getIntentIndexStats,
  listConversations,
  createConversation,
  deleteConversationApi,
} from "../api/client";
import UploadModal from "./UploadModal";
import TemperatureSlider from "./TemperatureSlider";
import GpuBadge from "./GpuBadge";

export default function Sidebar({
  temperature,
  onTemperatureChange,
  activeConversationId,
  onSelectConversation,
  onConversationCreated,
  sidebarVersion,
}) {
  const [conversations, setConversations] = useState([]);
  const [showUpload, setShowUpload] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [intentStats, setIntentStats] = useState(null);

  const refreshConversations = async () => {
    try {
      const res = await listConversations();
      setConversations(res.conversations || []);
    } catch {
      setConversations([]);
    }
  };

  useEffect(() => {
    getIntentIndexStats()
      .then(setIntentStats)
      .catch(() => setIntentStats(null));
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [sidebarVersion]);

  const handleUploadDone = () => {
    setShowUpload(false);
  };

  const handleNewConversation = async () => {
    try {
      const c = await createConversation();
      await refreshConversations();
      const id = c.id != null ? String(c.id) : "";
      onConversationCreated?.(id);
      onSelectConversation?.(id);
    } catch {
      /* ignore */
    }
  };

  const handleDeleteConversation = async (e, id) => {
    e.stopPropagation();
    if (!confirm("Xóa cuộc hội thoại này?")) return;
    try {
      await deleteConversationApi(id);
      await refreshConversations();
      if (activeConversationId === id) onSelectConversation?.(null);
    } catch {
      /* ignore */
    }
  };

  if (collapsed) {
    return (
      <div className="flex flex-col items-center py-4 gap-3 bg-surface-card border-r border-surface-border w-12">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          className="p-2 rounded-lg hover:bg-surface-hover text-gray-400"
        >
          <ChevronRight size={18} />
        </button>
      </div>
    );
  }

  return (
    <>
      <aside className="w-72 bg-surface-card border-r border-surface-border flex flex-col h-full">
        <div className="flex items-center justify-between px-4 py-4 border-b border-surface-border">
          <h1 className="text-base font-semibold text-gray-100 flex items-center gap-2">
            <span className="text-xl">🤖</span> RAG Chat
          </h1>
          <button
            type="button"
            onClick={() => setCollapsed(true)}
            className="p-1.5 rounded-lg hover:bg-surface-hover text-gray-400"
          >
            <ChevronLeft size={16} />
          </button>
        </div>

        <div className="px-3 pt-3 space-y-2">
          <button
            type="button"
            onClick={() => setShowUpload(true)}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-surface-border hover:border-primary-400 hover:bg-surface-hover text-sm text-gray-300 transition-colors"
          >
            <Plus size={16} />
            Tải lên tài liệu (.doc, .docx)
          </button>
          <button
            type="button"
            onClick={handleNewConversation}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-surface-border hover:bg-surface-hover text-sm text-gray-300 transition-colors"
          >
            <MessageSquare size={16} />
            Cuộc hội thoại mới
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider px-1 py-2">
              Hội thoại
            </p>
            {conversations.length === 0 && (
              <p className="text-xs text-gray-600 px-1">Chưa có hội thoại lưu.</p>
            )}
            <div className="space-y-1">
              {conversations.map((c, idx) => {
                const cid = c.id != null ? String(c.id) : `row-${idx}`;
                return (
                  <div
                    key={cid}
                    role="button"
                    tabIndex={0}
                    onClick={() => onSelectConversation?.(cid)}
                    onKeyDown={(ev) => {
                      if (ev.key === "Enter" || ev.key === " ") onSelectConversation?.(cid);
                    }}
                    className={`group flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer text-sm transition-colors ${
                      activeConversationId != null && String(activeConversationId) === cid
                        ? "bg-primary-600/15 text-primary-400 border border-primary-600/30"
                        : "text-gray-400 hover:bg-surface-hover hover:text-gray-200 border border-transparent"
                    }`}
                  >
                    <MessageSquare size={15} className="flex-shrink-0 opacity-70" />
                    <span className="flex-1 truncate">{c.title || cid}</span>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteConversation(e, cid)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 text-gray-500 hover:text-red-400 transition"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="border-t border-surface-border px-4 py-3 space-y-3">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wider">
            <Settings size={13} />
            Cài đặt
          </div>
          <TemperatureSlider value={temperature} onChange={onTemperatureChange} />
          {intentStats && (
            <p className="text-[10px] text-gray-600 leading-relaxed">
              Intent index: {intentStats.total_prototypes ?? 0} prototypes,{" "}
              {intentStats.intents_covered ?? 0}/23 intents
            </p>
          )}
          <GpuBadge />
        </div>
      </aside>

      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} onDone={handleUploadDone} />
      )}
    </>
  );
}
