/**
 * API Helper — All backend calls in one place.
 *
 * In development: calls go to /api/... (Vite proxy → localhost:8000)
 * In production:  calls go to VITE_API_URL (your Render backend)
 *
 * All calls include a session_id for per-user data isolation.
 */

const API_BASE = import.meta.env.VITE_API_URL || ''

export async function uploadPDFs(files, sessionId) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  formData.append('session_id', sessionId)

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Upload failed')
  }

  return res.json()
}

export async function sendQuery(question, sessionId) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Query failed')
  }

  return res.json()
}

export async function getStatus(sessionId) {
  try {
    const params = sessionId ? `?session_id=${sessionId}` : ''
    const res = await fetch(`${API_BASE}/api/status${params}`)
    return res.json()
  } catch {
    return { docs_loaded: false, chunk_count: 0 }
  }
}

export async function clearDatabase(sessionId) {
  const params = sessionId ? `?session_id=${sessionId}` : ''
  const res = await fetch(`${API_BASE}/api/clear${params}`, { method: 'DELETE' })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Clear failed')
  }

  return res.json()
}
