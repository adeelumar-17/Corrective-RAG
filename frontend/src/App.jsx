import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import * as api from './api'

/**
 * App — Root component.
 *
 * Manages global state: messages, doc status, query loading.
 * On NEW session (tab open / browser restart), auto-clears Pinecone
 * so the free-tier doesn't fill up across sessions.
 */
export default function App() {
  const [messages, setMessages] = useState([])
  const [docsLoaded, setDocsLoaded] = useState(false)
  const [chunkCount, setChunkCount] = useState(0)
  const [isQuerying, setIsQuerying] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // -------------------------------------------------------------------
  // Session management: clear Pinecone on every NEW browser session
  // sessionStorage is cleared when the tab/browser is closed,
  // but persists across page refreshes within the same session.
  // -------------------------------------------------------------------
  useEffect(() => {
    const sessionActive = sessionStorage.getItem('crag_session')

    if (!sessionActive) {
      // New session → clear the database, then check status
      api.clearDatabase()
        .then(() => {
          sessionStorage.setItem('crag_session', 'true')
          setDocsLoaded(false)
          setChunkCount(0)
        })
        .catch(() => {
          // If clear fails (e.g. backend not ready), still mark session
          sessionStorage.setItem('crag_session', 'true')
        })
    } else {
      // Existing session (page refresh) → just check current status
      api.getStatus().then((data) => {
        setDocsLoaded(data.docs_loaded)
        setChunkCount(data.chunk_count)
      })
    }
  }, [])

  // -------------------------------------------------------------------
  // Handlers passed to child components
  // -------------------------------------------------------------------
  function handleUploadComplete(count) {
    setDocsLoaded(true)
    setChunkCount((prev) => prev + count)
    addSystemMessage(`✅ Processed ${count} chunks from your documents. Ask away!`)
  }

  function handleClear() {
    setDocsLoaded(false)
    setChunkCount(0)
    addSystemMessage('🗑️ Database cleared. Upload new documents to get started.')
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
      const data = await api.sendQuery(question)
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
        onClear={handleClear}
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
