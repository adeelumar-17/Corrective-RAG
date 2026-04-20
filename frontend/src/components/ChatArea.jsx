import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import LoadingDots from './LoadingDots'

/**
 * ChatArea — Messages display + input form.
 *
 * Props:
 *   messages      – array of message objects
 *   isQuerying    – boolean: is a query in progress?
 *   onSendMessage – (question: string) => void
 */
export default function ChatArea({ messages, isQuerying, onSendMessage }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isQuerying])

  function handleSubmit(e) {
    e.preventDefault()
    const q = input.trim()
    if (!q || isQuerying) return
    onSendMessage(q)
    setInput('')
  }

  const isEmpty = messages.length === 0

  return (
    <main className="chat-area">
      {/* Header */}
      <div className="chat-header">
        <h2>Corrective RAG Assistant</h2>
        <p>Ask questions about your documents — or anything else</p>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {isEmpty && (
          <div className="welcome-message">
            <div className="welcome-icon"><span>🤖</span></div>
            <h3>Welcome!</h3>
            <p>
              Upload PDF documents using the sidebar, then ask me questions.
              I'll search your documents first and fall back to web search if needed.
            </p>
            <div className="welcome-badges">
              <span className="badge badge-docs">📄 From Documents</span>
              <span className="badge badge-web">🌐 From Web Search</span>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isQuerying && (
          <div className="message message-assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="message-bubble">
                <span style={{ color: '#94a3b8', marginRight: 8 }}>Thinking</span>
                <LoadingDots />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} autoComplete="off">
          <div className="input-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
              maxLength={1000}
              disabled={isQuerying}
            />
            <button type="submit" disabled={isQuerying || !input.trim()}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </main>
  )
}
