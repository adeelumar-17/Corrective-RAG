import { useState, useRef } from 'react'
import * as api from '../api'

/**
 * Sidebar — PDF upload, document status, and clear button.
 *
 * Props:
 *   docsLoaded      – boolean: are documents in Pinecone?
 *   chunkCount      – number: total vectors in Pinecone
 *   onUploadComplete – (count) => void: called after successful upload
 *   onClear          – () => void: called after clearing the database
 *   isOpen           – boolean: mobile sidebar visibility
 *   onClose          – () => void: close sidebar on mobile
 */
export default function Sidebar({
  docsLoaded,
  chunkCount,
  onUploadComplete,
  onClear,
  isOpen,
  onClose,
}) {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  // ---- File selection + auto-process ----
  async function addFiles(fileList) {
    const pdfs = Array.from(fileList).filter(
      (f) => f.type === 'application/pdf'
    )
    if (pdfs.length === 0) return

    // Show files in the list while processing
    setSelectedFiles(pdfs)
    setIsProcessing(true)

    try {
      const data = await api.uploadPDFs(pdfs)
      setSelectedFiles([])
      onUploadComplete(data.chunk_count)
    } catch (err) {
      alert(`Upload failed: ${err.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  // ---- Clear ----
  async function handleClear() {
    if (!confirm('Clear all uploaded documents?')) return
    try {
      await api.clearDatabase()
      onClear()
    } catch (err) {
      alert(`Clear failed: ${err.message}`)
    }
  }

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="logo">
          <span className="logo-icon">📚</span>
          <div>
            <h1>CRAG</h1>
            <p className="logo-subtitle">Corrective RAG</p>
          </div>
        </div>
        <button className="sidebar-close" onClick={onClose} aria-label="Close sidebar">
          ✕
        </button>
      </div>

      {/* How it works */}
      <div className="sidebar-section">
        <h3>How it works</h3>
        <div className="steps">
          {['Upload PDF documents', 'Ask questions naturally', 'AI grades retrieved chunks', 'Falls back to web if needed'].map((text, i) => (
            <div className="step" key={i}>
              <span className="step-num">{i + 1}</span>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Upload */}
      <div className="sidebar-section">
        <h3>Upload Documents</h3>
        <div
          className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            addFiles(e.dataTransfer.files)
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf"
            hidden
            onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
          />
          <div className="drop-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="32" height="32">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="drop-text">Drag & drop PDFs here</p>
          <p className="drop-subtext">or click to browse</p>
        </div>

        {selectedFiles.length > 0 && (
          <div className="file-list">
            {selectedFiles.map((file) => (
              <div className="file-item" key={file.name}>
                <span className="file-item-name" title={file.name}>📄 {file.name}</span>
              </div>
            ))}
          </div>
        )}

        {isProcessing && (
          <div className="processing-indicator">
            <span className="btn-loader"><span className="dot" /><span className="dot" /><span className="dot" /></span>
            <span>Processing documents...</span>
          </div>
        )}
      </div>

      {/* Status */}
      <div className="sidebar-section">
        <div className="status-badge">
          <span className={`status-dot ${docsLoaded ? 'status-loaded-dot' : 'status-empty-dot'}`} />
          <span>{docsLoaded ? `${chunkCount} chunks loaded` : 'No documents loaded'}</span>
        </div>
      </div>

      {/* Clear */}
      <button className="btn btn-danger" onClick={handleClear}>
        🗑️ Clear Database
      </button>

      <div className="sidebar-footer">
        <p>Built with LangGraph + Pinecone + Groq</p>
      </div>
    </aside>
  )
}
