/**
 * API Helper — All backend calls in one place.
 *
 * In development: calls go to /api/... (Vite proxy → localhost:8000)
 * In production:  calls go to VITE_API_URL (your Render backend)
 */

const API_BASE = import.meta.env.VITE_API_URL || ''

export async function uploadPDFs(files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

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

export async function sendQuery(question) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Query failed')
  }

  return res.json()
}

export async function getStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`)
    return res.json()
  } catch {
    return { docs_loaded: false, chunk_count: 0 }
  }
}

export async function clearDatabase() {
  const res = await fetch(`${API_BASE}/api/clear`, { method: 'DELETE' })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Clear failed')
  }

  return res.json()
}
