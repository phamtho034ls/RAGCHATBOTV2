import React, { useState, useRef, useEffect } from "react";
import { MessageSquareText, RefreshCw, Loader2 } from "lucide-react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import TypingIndicator from "../components/TypingIndicator";
import SourcesPanel from "../components/SourcesPanel";
import { chatStream, createConversation, getConversationDetail } from "../api/client";

export default function ChatPage({
  conversationId,
  temperature,
  onConversationIdChange,
  onNotifySidebar,
}) {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [currentSources, setCurrentSources] = useState(null);
  const [lastMeta, setLastMeta] = useState(null);
  const [chatError, setChatError] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const bottomRef = useRef(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streaming]);

  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      setCurrentSources(null);
      setLastMeta(null);
      setChatError(null);
      setHistoryLoading(false);
      return;
    }
    let cancelled = false;
    const cid = String(conversationId);
    setHistoryLoading(true);
    setChatError(null);
    getConversationDetail(cid)
      .then((detail) => {
        if (cancelled) return;
        const msgs = (detail.messages || []).map((m) => ({
          role: m.role === "assistant" ? "bot" : m.role,
          content: m.content || "",
        }));
        setMessages(msgs);
        setCurrentSources(null);
        setLastMeta(null);
      })
      .catch((err) => {
        if (!cancelled) {
          setChatError(err?.message || "Không tải được lịch sử hội thoại.");
          setMessages([]);
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  const handleNewChat = async () => {
    try {
      const c = await createConversation();
      const id = c.id != null ? String(c.id) : "";
      onConversationIdChange?.(id);
      onNotifySidebar?.();
      setMessages([]);
      setCurrentSources(null);
      setLastMeta(null);
      setChatError(null);
    } catch (e) {
      setChatError(e.message || "Không tạo được hội thoại mới.");
    }
  };

  const handleSend = async (question) => {
    setChatError(null);

    let cid = conversationId != null ? String(conversationId) : "";
    if (!cid) {
      try {
        const c = await createConversation();
        cid = c.id != null ? String(c.id) : "";
        onConversationIdChange?.(cid);
        onNotifySidebar?.();
      } catch (e) {
        setChatError(e.message || "Không tạo được hội thoại.");
        return;
      }
    }

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setStreaming(true);
    setCurrentSources(null);

    try {
      await chatStream(
        question,
        temperature,
        (token) => {
          setMessages((prev) => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            const prevBot = copy[lastIdx]?.role === "bot" ? copy[lastIdx].content : "";
            const nextContent = prevBot + token;
            if (copy[lastIdx]?.role === "bot") {
              copy[lastIdx] = { ...copy[lastIdx], content: nextContent };
            } else {
              copy.push({ role: "bot", content: nextContent });
            }
            return copy;
          });
        },
        (sources) => {
          setCurrentSources(sources);
        },
        () => {
          setStreaming(false);
        },
        cid,
        (meta) => {
          if (meta.conversation_id) onConversationIdChange?.(String(meta.conversation_id));
          setLastMeta(meta);
          setMessages((prev) => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            if (copy[lastIdx]?.role === "bot") {
              copy[lastIdx] = {
                ...copy[lastIdx],
                confidence: meta.confidence_score,
                retried: meta.retried,
              };
            } else if (copy[lastIdx]?.role === "user") {
              copy.push({
                role: "bot",
                content: "",
                confidence: meta.confidence_score,
                retried: meta.retried,
              });
            }
            return copy;
          });
        },
        (fin) => {
          const text = fin.text ?? "";
          setMessages((prev) => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            if (copy[lastIdx]?.role === "bot") {
              copy[lastIdx] = {
                ...copy[lastIdx],
                content: text,
                confidence: fin.confidence_score,
                retried: fin.retried,
              };
            }
            return copy;
          });
          setLastMeta({
            conversation_id: cid,
            confidence_score: fin.confidence_score,
            retried: fin.retried,
          });
        }
      );
    } catch (err) {
      setChatError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: `❌ Lỗi: ${err.message}`,
        },
      ]);
      setStreaming(false);
    }
  };

  const showGlobalEmpty = !conversationId && messages.length === 0 && !historyLoading;
  const showConvEmpty =
    conversationId && !historyLoading && messages.length === 0;

  return (
    <div className="flex-1 flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex-1 overflow-y-auto min-h-0 px-4 md:px-8 lg:px-16 xl:px-32 py-6 space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
          <button
            type="button"
            onClick={handleNewChat}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-surface-border hover:bg-surface-hover text-gray-300"
          >
            <RefreshCw size={12} /> Cuộc hội thoại mới
          </button>
          {conversationId && (
            <span className="font-mono text-[10px] text-gray-600">
              ID: {String(conversationId)}
            </span>
          )}
          {lastMeta && typeof lastMeta.confidence_score === "number" && (
            <span>
              Độ tin cậy: {(lastMeta.confidence_score * 100).toFixed(0)}%
              {lastMeta.retried ? " • đã retry" : ""}
            </span>
          )}
        </div>
        {chatError && (
          <div className="text-xs text-red-400 bg-red-950/30 border border-red-900/50 rounded-lg px-3 py-2">
            {chatError}
          </div>
        )}
        {historyLoading && (
          <div className="flex items-center justify-center gap-2 py-12 text-sm text-gray-500">
            <Loader2 className="animate-spin" size={18} />
            Đang tải lịch sử…
          </div>
        )}

        {showGlobalEmpty && (
          <div className="flex flex-col items-center justify-center min-h-[40vh] text-center">
            <div className="w-12 h-12 rounded-xl bg-primary-600/10 flex items-center justify-center mb-3">
              <MessageSquareText size={24} className="text-primary-400" />
            </div>
            <p className="text-sm text-gray-500 max-w-sm">
              Chọn hoặc tạo cuộc hội thoại từ thanh bên, rồi đặt câu hỏi pháp luật. Bạn cũng có thể gửi tin nhắn
              ngay — hệ thống sẽ tạo hội thoại mới nếu cần.
            </p>
          </div>
        )}

        {showConvEmpty && (
          <div className="flex flex-col items-center justify-center min-h-[30vh] text-center">
            <div className="w-12 h-12 rounded-xl bg-primary-600/10 flex items-center justify-center mb-3">
              <MessageSquareText size={24} className="text-primary-400" />
            </div>
            <p className="text-sm text-gray-500 max-w-sm">
              Cuộc hội thoại này chưa có tin nhắn (hoặc chưa lưu xong). Hãy gửi câu hỏi bên dưới; sau khi stream
              kết thúc, tin nhắn sẽ được lưu vào PostgreSQL.
            </p>
          </div>
        )}

        {!historyLoading &&
          messages.map((msg, i) => (
            <React.Fragment key={i}>
              <ChatMessage
                role={msg.role}
                content={msg.content}
                confidence={msg.confidence}
                retried={msg.retried}
              />
              {msg.role === "bot" && i === messages.length - 1 && currentSources && (
                <SourcesPanel sources={currentSources} />
              )}
            </React.Fragment>
          ))}

        {streaming && messages[messages.length - 1]?.role !== "bot" && <TypingIndicator />}

        <div ref={bottomRef} />
      </div>

      <div className="px-4 md:px-8 lg:px-16 xl:px-32 pb-4 pt-2">
        <ChatInput onSend={handleSend} disabled={streaming || historyLoading} />
        <p className="text-[10px] text-gray-600 text-center mt-2">
          RAG Chatbot – Tra cứu pháp luật • Hội thoại lưu trên PostgreSQL • Streaming từ LLM
        </p>
      </div>
    </div>
  );
}
