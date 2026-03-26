import React, { useState, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatPage from "./pages/ChatPage";

const STORAGE_KEY = "rag_chat_conversation_id";

function readStoredConversationId() {
  try {
    return localStorage.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export default function App() {
  const [temperature, setTemperature] = useState(0.5);
  const [conversationId, setConversationIdState] = useState(readStoredConversationId);
  const [sidebarVersion, setSidebarVersion] = useState(0);

  const setConversationId = useCallback((id) => {
    setConversationIdState(id);
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar
        temperature={temperature}
        onTemperatureChange={setTemperature}
        activeConversationId={conversationId}
        onSelectConversation={setConversationId}
        onConversationCreated={(id) => {
          setConversationId(id);
          setSidebarVersion((v) => v + 1);
        }}
        sidebarVersion={sidebarVersion}
      />
      <ChatPage
        conversationId={conversationId}
        temperature={temperature}
        onConversationIdChange={setConversationId}
        onNotifySidebar={() => setSidebarVersion((v) => v + 1)}
      />
    </div>
  );
}
