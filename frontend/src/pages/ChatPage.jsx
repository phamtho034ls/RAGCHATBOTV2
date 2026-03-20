import React, { useState, useRef, useEffect } from "react";
import { MessageSquareText } from "lucide-react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import TypingIndicator from "../components/TypingIndicator";
import SourcesPanel from "../components/SourcesPanel";
import { chatStream } from "../api/client";

export default function ChatPage({ datasetId, temperature, hasDatasets }) {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [currentSources, setCurrentSources] = useState(null);
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
  }, [datasetId]);

  const handleSend = async (question) => {
    if (!hasDatasets) return;

    // Thêm message user
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setStreaming(true);
    setCurrentSources(null);

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
        }
      );
    } catch (err) {
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
            <ChatMessage role={msg.role} content={msg.content} />
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
