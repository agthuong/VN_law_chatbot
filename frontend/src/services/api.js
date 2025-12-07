/**
 * API service for VN Law RAG system
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

/**
 * Send chat request (non-streaming)
 * @param {string} question - User question
 * @param {Object} options - Request options (top_k, temperature, max_tokens)
 * @returns {Promise<Object>} Chat response
 */
export async function chat(question, options = {}) {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      top_k: options.top_k || 5,
      temperature: options.temperature || 0.7,
      max_tokens: options.max_tokens || null,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Send chat request with streaming
 * @param {string} question - User question
 * @param {Object} options - Request options
 * @param {Function} onChunk - Callback for each chunk
 * @returns {Promise<void>}
 */
export async function chatStream(question, options = {}, onChunk) {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      top_k: options.top_k || 5,
      temperature: options.temperature || 0.7,
      max_tokens: options.max_tokens || null,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (onChunk) {
              onChunk(data);
            }
          } catch (e) {
            console.error('Error parsing SSE data:', e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    
    if (!response.ok) {
      // if 404, server might not be running or route not found
      if (response.status === 404) {
        throw new Error(`API endpoint not found. Make sure the server is running at ${API_BASE_URL}`);
      }
      // try to get error message from response
      try {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Health check failed: ${response.status}`);
      } catch {
        throw new Error(`Health check failed: ${response.status}`);
      }
    }

    return await response.json();
  } catch (error) {
    // if fetch fails (network error, CORS, etc)
    if (error.message.includes('Failed to fetch') || 
        error.message.includes('NetworkError') ||
        error.message.includes('Network request failed')) {
      throw new Error(`Cannot connect to server at ${API_BASE_URL}. Make sure the API is running.`);
    }
    throw error;
  }
}

/**
 * Check vector store health
 * @returns {Promise<Object>} Vector store status
 */
export async function checkVectorStoreHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health/vector-store`);
  
  if (!response.ok) {
    throw new Error(`Vector store health check failed: ${response.status}`);
  }

  return await response.json();
}

