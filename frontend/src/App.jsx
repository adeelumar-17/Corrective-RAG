import { useState, useEffect, useRef } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import * as api from './api'

/**
 * App — Root component.
 *
 * Manages global state: messages, doc status, query loading, session identity.
 *
 * On EVERY page load (including refresh), generates a fresh session ID and
 * clears any previous data. Each session uses a unique Pinecone namespace
 * so concurrent users are fully isolated.
 */
export default function App() {
  const [messages, setMessages] = useState([])
  const [docsLoaded, setDocsLoaded] = useState(false)
  const [chunkCount, setChunkCount] = useState(0)
  const [isQuerying, setIsQuerying] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Generate a unique session ID on mount — persists for this page lifetime only
  const sessionIdRef = useRef(crypto.randomUUID())

  // -------------------------------------------------------------------
  // Session management: fresh session on every page load
  // Each load gets a new UUID namespace — previous data is abandoned.
  // We also call clear to clean up the new namespace (no-op, but safe).
  // -------------------------------------------------------------------
  useEffect(() => {
    // Fresh session on every mount — reset everything
    api.clearDatabase(sessionIdRef.current)
      .then(() => {
        setDocsLoaded(false)
        setChunkCount(0)
      })
      .catch(() => {
        // If clear fails (e.g. backend not ready), continue anyway
        setDocsLoaded(false)
        setChunkCount(0)
      })
  }, [])

  // -------------------------------------------------------------------
  // Handlers passed to child components
  // -------------------------------------------------------------------
  function handleUploadComplete(count) {
    setDocsLoaded(true)
    setChunkCount((prev) => prev + count)
    addSystemMessage(`✅ Processed ${count} chunks from your documents. Ask away!`)
  }

  function addSystemMessage(text) {
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: 'system', content: text },
    ])
  }

  async function handleSendMessage(question) {
    // Add user message immediately
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: 'user', content: question },
    ])

    setIsQuerying(true)

    try {
      const data = await api.sendQuery(question, sessionIdRef.current)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          usedWebSearch: data.used_web_search,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          content: `Sorry, something went wrong: ${err.message}`,
        },
      ])
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <div className="app-container">
      <Sidebar
        docsLoaded={docsLoaded}
        chunkCount={chunkCount}
        onUploadComplete={handleUploadComplete}
        sessionId={sessionIdRef.current}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open sidebar"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <ChatArea
        messages={messages}
        isQuerying={isQuerying}
        onSendMessage={handleSendMessage}
      />
    </div>
  )
}
