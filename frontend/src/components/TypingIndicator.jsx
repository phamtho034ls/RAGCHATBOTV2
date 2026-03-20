import React from "react";
import { Bot } from "lucide-react";

export default function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-slide-up">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-600/20 flex items-center justify-center">
        <Bot size={18} className="text-primary-400" />
      </div>
      <div className="bg-surface-card rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-1.5">
        <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full inline-block" />
        <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full inline-block" />
        <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full inline-block" />
      </div>
    </div>
  );
}
