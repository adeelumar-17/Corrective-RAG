import { useState } from 'react'

/**
 * MessageBubble — A single chat message with optional badge + sources.
 *
 * Props:
 *   message – { role, content, sources?, usedWebSearch? }
 */
export default function MessageBubble({ message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const { role, content, sources, usedWebSearch } = message

  // System messages (info toasts)
  if (role === 'system') {
    return (
      <div className="message message-assistant">
        <div className="message-avatar">ℹ️</div>
        <div className="message-content">
          <div className="message-bubble system-bubble">{content}</div>
        </div>
      </div>
    )
  }

  const isUser = role === 'user'
  const avatar = isUser ? '👤' : '🤖'

  return (
    <div className={`message message-${role}`}>
      <div className="message-avatar">{avatar}</div>
      <div className="message-content">
        {/* Message text */}
        <div
          className="message-bubble"
          dangerouslySetInnerHTML={{ __html: formatContent(content) }}
        />

        {/* Source badge */}
        {!isUser && usedWebSearch !== undefined && (
          usedWebSearch
            ? <span className="badge badge-web">🌐 From Web Search</span>
            : <span className="badge badge-docs">📄 From Documents</span>
        )}

        {/* Expandable sources */}
        {sources && sources.length > 0 && (
          <div className="sources-expander">
            <button
              className={`sources-toggle ${sourcesOpen ? 'active' : ''}`}
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              <span>View Sources ({sources.length})</span>
              <span className="sources-toggle-arrow">▼</span>
            </button>
            {sourcesOpen && (
              <div className="sources-list open">
                {sources.map((src, i) => (
                  <div className="source-item" key={i}>
                    {src.startsWith('http') ? (
                      <a href={src} target="_blank" rel="noopener noreferrer">{src}</a>
                    ) : (
                      <>📎 {src}</>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Basic markdown-like formatting for message content */
function formatContent(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(
      /`(.*?)`/g,
      '<code style="background:rgba(124,58,237,0.15);padding:2px 6px;border-radius:4px;font-size:0.85em;">$1</code>'
    )
}
