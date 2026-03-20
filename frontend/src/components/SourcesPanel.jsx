import React, { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp, FileText } from "lucide-react";

export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="animate-slide-up ml-11 max-w-[75%]">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition mb-1"
      >
        <BookOpen size={13} />
        {sources.length} nguồn tham khảo
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {open && (
        <div className="space-y-2 mt-1">
          {sources.map((s, i) => {
            const citation = s.citation || "";
            const snippet = s.snippet || s.content || "";
            const meta = s.metadata || {};
            const fallbackCitation =
              meta.law_name && meta.article_number
                ? `${meta.law_name} – Điều ${meta.article_number}`
                : meta.law_name || meta.doc_number || "";
            const displayCitation = citation || fallbackCitation;

            return (
              <div
                key={i}
                className="bg-surface-card border border-surface-border rounded-xl px-3 py-2 text-xs text-gray-400"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-500 font-medium flex items-center gap-1">
                    <FileText size={11} />
                    Nguồn #{i + 1}
                  </span>
                </div>
                {displayCitation && <div className="mb-1 text-gray-300 font-medium">{displayCitation}</div>}
                <p className="text-gray-400 leading-relaxed line-clamp-3">
                  {snippet}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
