import React, { useState, useRef, useEffect } from "react";
import { MessageSquareText, RefreshCw } from "lucide-react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import TypingIndicator from "../components/TypingIndicator";
import SourcesPanel from "../components/SourcesPanel";
import { chatStream } from "../api/client";

export default function ChatPage({ datasetId, temperature, hasDatasets }) {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [currentSources, setCurrentSources] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [lastMeta, setLastMeta] = useState(null);
  const [chatError, setChatError] = useState(null);
  const bottomRef = useRef(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streaming]);

  // Reset chat khi đổi dataset
  useEffect(() => {
    setMessages([]);
    setCurrentSources(null);
    setConversationId(null);
    setLastMeta(null);
    setChatError(null);
  }, [datasetId]);

  const handleNewChat = () => {
    setMessages([]);
    setCurrentSources(null);
    setConversationId(null);
    setLastMeta(null);
    setChatError(null);
  };

  const handleSend = async (question) => {
    if (!hasDatasets) return;

    // Thêm message user
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setStreaming(true);
    setCurrentSources(null);
    setChatError(null);

    let botText = "";

    try {
      await chatStream(
        question,
        temperature,
        // onToken
        (token) => {
          botText += token;
          setMessages((prev) => {
            const copy = [...prev];
            const lastIdx = copy.length - 1;
            // Update hoặc thêm bot message
            if (copy[lastIdx]?.role === "bot") {
              copy[lastIdx] = { ...copy[lastIdx], content: botText };
            } else {
              copy.push({ role: "bot", content: botText });
            }
            return copy;
          });
        },
        // onSources
        (sources) => {
          setCurrentSources(sources);
        },
        // onDone
        () => {
          setStreaming(false);
        },
        conversationId,
        // onMeta
        (meta) => {
          if (meta.conversation_id) setConversationId(meta.conversation_id);
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

  // ── Empty state ──────────────────────────────────────────
  if (!hasDatasets) {
    return (
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="flex-1 flex flex-col items-center justify-center text-center px-6">
          <div className="w-16 h-16 rounded-2xl bg-primary-600/10 flex items-center justify-center mb-4">
            <MessageSquareText size={32} className="text-primary-400" />
          </div>
          <h2 className="text-xl font-semibold text-gray-200 mb-2">
            Chào mừng bạn đến với RAG Chatbot
          </h2>
          <p className="text-sm text-gray-500 max-w-sm">
            Tải lên file tài liệu Word (.doc, .docx) từ thanh bên trái, sau đó đặt câu
            hỏi để bắt đầu hỏi đáp thông minh.
          </p>
        </div>
        <div className="px-4 md:px-8 lg:px-16 xl:px-32 pb-4 pt-2">
          <ChatInput onSend={() => {}} disabled />
          <p className="text-[10px] text-gray-600 text-center mt-2">
            RAG Chatbot – Hỏi đáp dựa trên tài liệu • Chạy hoàn toàn local
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full min-h-0 overflow-hidden">
      {/* Messages area */}
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
            <span className="font-mono text-[10px] text-gray-600">ID: {conversationId}</span>
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
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-12 h-12 rounded-xl bg-primary-600/10 flex items-center justify-center mb-3">
              <MessageSquareText size={24} className="text-primary-400" />
            </div>
            <p className="text-sm text-gray-500">
              Đặt câu hỏi về nội dung tài liệu để bắt đầu…
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <React.Fragment key={i}>
            <ChatMessage
              role={msg.role}
              content={msg.content}
              confidence={msg.confidence}
              retried={msg.retried}
            />
            {/* Show sources after bot message */}
            {msg.role === "bot" && i === messages.length - 1 && currentSources && (
              <SourcesPanel sources={currentSources} />
            )}
          </React.Fragment>
        ))}

        {streaming && messages[messages.length - 1]?.role !== "bot" && (
          <TypingIndicator />
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-4 md:px-8 lg:px-16 xl:px-32 pb-4 pt-2">
        <ChatInput onSend={handleSend} disabled={streaming || !hasDatasets} />
        <p className="text-[10px] text-gray-600 text-center mt-2">
          RAG Chatbot – Hỏi đáp dựa trên tài liệu • Chạy hoàn toàn local
        </p>
      </div>
    </div>
  );
}
