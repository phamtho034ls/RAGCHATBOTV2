import React from "react";
import ReactMarkdown from "react-markdown";
import { Bot, User } from "lucide-react";

/**
 * Split bot answer into main content and citation footer (Nguồn: ...).
 */
function splitCitations(text) {
  const marker = /\n\nNguồn:\n/;
  const match = text.match(marker);
  if (!match) return { body: text, citations: null };
  const idx = match.index;
  return {
    body: text.slice(0, idx),
    citations: text.slice(idx + match[0].length).trim(),
  };
}

export default function ChatMessage({ role, content, confidence, retried }) {
  const isUser = role === "user";
  const { body, citations } = isUser ? { body: content, citations: null } : splitCitations(content);

  return (
    <div className={`flex gap-3 animate-slide-up ${isUser ? "justify-end" : "justify-start"}`}>
      {/* Avatar bot */}
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-600/20 flex items-center justify-center mt-1">
          <Bot size={18} className="text-primary-400" />
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-primary-600 text-white rounded-br-md"
            : "bg-surface-card text-gray-200 rounded-bl-md bot-message"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <>
            <ReactMarkdown>{body}</ReactMarkdown>
            {citations && (
              <div className="mt-3 pt-2 border-t border-surface-border text-xs text-gray-500">
                <ReactMarkdown>{citations}</ReactMarkdown>
              </div>
            )}
            {typeof confidence === "number" && (
              <p className="mt-2 text-[10px] text-gray-500">
                Độ tin cậy ước lượng: {(confidence * 100).toFixed(0)}%
                {retried ? " • đã thử lại với nhiều nguồn" : ""}
              </p>
            )}
          </>
        )}
      </div>

      {/* Avatar user */}
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center mt-1">
          <User size={18} className="text-white" />
        </div>
      )}
    </div>
  );
}
