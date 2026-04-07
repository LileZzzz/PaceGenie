import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Zap, Target, Flame, Heart, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '@/types';
import { API_ENDPOINTS } from '@/lib/config';

const suggestedQuestions = [
  { id: '1', text: 'Analyze my weekly performance', icon: <Target className="w-3.5 h-3.5" /> },
  { id: '2', text: 'Create a training plan', icon: <Zap className="w-3.5 h-3.5" /> },
  { id: '3', text: 'How to improve pace?', icon: <Flame className="w-3.5 h-3.5" /> },
  { id: '4', text: 'Is my heart rate normal?', icon: <Heart className="w-3.5 h-3.5" /> },
];

const initialMessages: ChatMessage[] = [
  {
    role: 'assistant',
    content: "Hi! I'm PaceGenie, your AI running coach. I can help you analyze running data, create training plans, and answer running-related questions. What would you like to talk about today?",
    timestamp: new Date(),
  },
];

// Markdown component map — applies Tailwind styles to each rendered element
const markdownComponents = {
  p: ({ children }: { children?: React.ReactNode }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  h1: ({ children }: { children?: React.ReactNode }) => <h1 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
  h2: ({ children }: { children?: React.ReactNode }) => <h2 className="text-sm font-bold mb-2 mt-3 first:mt-0 text-red-400">{children}</h2>,
  h3: ({ children }: { children?: React.ReactNode }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h3>,
  strong: ({ children }: { children?: React.ReactNode }) => <strong className="font-semibold text-white">{children}</strong>,
  em: ({ children }: { children?: React.ReactNode }) => <em className="italic text-white/80">{children}</em>,
  ul: ({ children }: { children?: React.ReactNode }) => <ul className="list-disc list-inside space-y-0.5 mb-2 ml-1">{children}</ul>,
  ol: ({ children }: { children?: React.ReactNode }) => <ol className="list-decimal list-inside space-y-0.5 mb-2 ml-1">{children}</ol>,
  li: ({ children }: { children?: React.ReactNode }) => <li className="text-sm leading-relaxed">{children}</li>,
  hr: () => <hr className="border-white/10 my-3" />,
  code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
    const isBlock = className?.includes('language-');
    return isBlock
      ? <code className="block bg-black/30 rounded-lg p-2 text-xs font-mono my-2 overflow-x-auto">{children}</code>
      : <code className="bg-black/30 rounded px-1.5 py-0.5 text-xs font-mono text-red-300">{children}</code>;
  },
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="overflow-x-auto my-2 rounded-lg border border-white/10">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => <thead className="bg-white/5 border-b border-white/10">{children}</thead>,
  tbody: ({ children }: { children?: React.ReactNode }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
  tr: ({ children }: { children?: React.ReactNode }) => <tr className="hover:bg-white/5 transition-colors">{children}</tr>,
  th: ({ children }: { children?: React.ReactNode }) => <th className="px-3 py-2 text-left font-semibold text-white/70 whitespace-nowrap">{children}</th>,
  td: ({ children }: { children?: React.ReactNode }) => <td className="px-3 py-2 text-white/90">{children}</td>,
};

interface ChatInterfaceProps {
  userId: string;
  sessionId: string;
  backendOnline: boolean | null;
}

export function ChatInterface({ userId, sessionId, backendOnline }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    const messageText = inputValue.trim();
    if (!messageText) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Streaming message ID — we update this message in-place as tokens arrive
    const streamingId = `stream-${Date.now()}`;

    try {
      const response = await fetch(API_ENDPOINTS.chatStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          user_id: userId,
          session_id: sessionId,
        }),
      });

      if (!response.ok || !response.body) {
        // Fallback to non-streaming if stream endpoint fails
        const data = await response.json().catch(() => ({ reply: 'Backend returned an error.' }));
        setMessages(prev => [...prev, { role: 'assistant', content: data.reply ?? 'Backend returned an error.', timestamp: new Date() }]);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let firstToken = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value, { stream: true }).split('\n');
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6)) as { token?: string; status?: string; done?: boolean; error?: string };
            if (data.status) {
              // Tool call in progress — show status under the typing indicator
              setToolStatus(data.status);
            }
            if (data.token) {
              if (firstToken) {
                // First content token — swap typing indicator for streaming message
                firstToken = false;
                setIsTyping(false);
                setToolStatus(null);
                setMessages(prev => [
                  ...prev,
                  { id: streamingId, role: 'assistant', content: data.token ?? '', timestamp: new Date() },
                ]);
              } else {
                setMessages(prev => prev.map(m =>
                  m.id === streamingId
                    ? { ...m, content: m.content + data.token }
                    : m
                ));
              }
            }
            if (data.error) {
              setIsTyping(false);
              setToolStatus(null);
              setMessages(prev => [
                ...prev.filter(m => m.id !== streamingId),
                { role: 'assistant', content: `Error: ${data.error}`, timestamp: new Date() },
              ]);
            }
          } catch {
            // Malformed SSE line — skip
          }
        }
      }
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Backend is offline. Start it with: `uv run uvicorn api.main:app --reload --port 8000`',
        timestamp: new Date(),
      }]);
    } finally {
      setIsTyping(false);
      setToolStatus(null);
    }
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputValue(question);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const { dot: statusDot, label: statusLabel } =
    backendOnline === null
      ? { dot: 'bg-yellow-400 animate-pulse', label: 'Connecting...' }
      : backendOnline
      ? { dot: 'bg-emerald-500', label: 'Online' }
      : { dot: 'bg-red-500', label: 'Backend offline' };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-card to-background rounded-xl border border-border/50 overflow-hidden shadow-card">
      {/* Chat header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50 bg-card/50">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center shadow-glow">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 ${statusDot} rounded-full border-2 border-card`} />
          </div>
          <div>
            <h3 className="font-semibold text-white">PaceGenie AI</h3>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <span className={`w-1.5 h-1.5 ${statusDot} rounded-full`} />
              {statusLabel}
            </p>
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 p-4 overflow-y-auto" ref={scrollRef}>
        <div className="space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className="flex gap-3 animate-slide-in"
              style={{ animationDelay: `${Math.min(index * 0.05, 0.3)}s` }}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center flex-shrink-0">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
              )}

              <div className={`flex-1 ${message.role === 'user' ? 'flex justify-end' : ''}`}>
                <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-red-500 to-red-700 text-white rounded-br-md'
                    : 'bg-white/5 text-white border border-white/6 rounded-bl-md'
                }`}>
                  <div className="text-sm">
                    {message.role === 'user' ? (
                      <p className="leading-relaxed">{message.content}</p>
                    ) : (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {message.content}
                      </ReactMarkdown>
                    )}
                  </div>
                  <span className={`text-[10px] mt-1 block ${
                    message.role === 'user' ? 'text-white/70' : 'text-muted-foreground'
                  }`}>
                    {formatTime(message.timestamp)}
                  </span>
                </div>
              </div>

              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                  <span className="text-sm text-muted-foreground">Me</span>
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div className="bg-white/5 border border-white/6 rounded-2xl rounded-bl-md py-3 px-4">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                  </div>
                  {toolStatus && (
                    <span className="text-xs text-muted-foreground italic">{toolStatus}</span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Suggested questions */}
      {messages.length < 3 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-muted-foreground mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.map((q) => (
              <button
                key={q.id}
                onClick={() => handleSuggestedQuestion(q.text)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-background/80 border border-border/50 text-xs text-muted-foreground hover:text-white hover:border-red-500/30 transition-all"
              >
                {q.icon}
                {q.text}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="p-4 border-t border-border/50 bg-card/50">
        <div className="flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Type your question or request..."
              className="w-full px-4 py-3 bg-background/80 border border-border/50 rounded-xl text-white placeholder:text-muted-foreground focus:outline-none focus:border-red-500/50 transition-colors"
            />
            {inputValue && (
              <button
                onClick={() => setInputValue('')}
                className="absolute right-3 top-1/2 -translate-y-1/2"
              >
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              </button>
            )}
          </div>
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isTyping}
            className="p-3 rounded-xl bg-gradient-to-br from-red-500 to-red-700 text-white hover:from-red-400 hover:to-red-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-sm"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
