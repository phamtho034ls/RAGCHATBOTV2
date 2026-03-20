import React, { useState, useRef } from "react";
import { SendHorizontal } from "lucide-react";

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setText(e.target.value);
    // Auto-resize textarea
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  return (
    <div className="flex items-end gap-2 bg-surface-card border border-surface-border rounded-2xl px-4 py-3">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="Nhập câu hỏi tại đây..."
        disabled={disabled}
        rows={1}
        className="flex-1 bg-transparent text-gray-200 placeholder-gray-500 resize-none outline-none text-sm max-h-40"
      />
      <button
        onClick={handleSend}
        disabled={!text.trim() || disabled}
        className="p-2 rounded-xl bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
      >
        <SendHorizontal size={18} />
      </button>
    </div>
  );
}
