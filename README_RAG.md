# VN Law RAG System với LangChain và Ollama

Hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh cho pháp luật Việt Nam sử dụng:
- **LangChain**: Framework quản lý RAG pipeline
- **Ollama + Qwen30B**: LLM để generate câu trả lời
- **VN-Law-Embedding**: Embedding model cho semantic search
- **FAISS**: Vector database
- **FastAPI**: REST API interface

## Yêu cầu

1. **Ollama đã cài đặt và model Qwen30B đã pull**:
   ```bash
   ollama pull qwen2.5:30b
   # hoặc
   ollama pull qwen:30b
   ```

2. **Vector store đã được build** (từ `scripts/build_index.py`)

3. **Python 3.8+**

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements_rag.txt
```

### 2. Cấu hình environment variables

Copy `.env.example` thành `.env` và chỉnh sửa nếu cần:

```bash
cp .env.example .env
```

Các biến môi trường chính:
- `OLLAMA_BASE_URL`: URL của Ollama server (mặc định: `http://localhost:11434`)
- `OLLAMA_MODEL`: Tên model (mặc định: `qwen2.5:30b`)
- `VECTOR_STORE_DIR`: Thư mục chứa vector store (mặc định: `./vector_db`)
- `TOP_K`: Số documents để retrieve (mặc định: `5`)
- `TEMPERATURE`: LLM temperature (mặc định: `0.7`)

### 3. Khởi động Ollama

Đảm bảo Ollama đang chạy:

```bash
ollama serve
```

Kiểm tra model có sẵn:

```bash
ollama list
```

## Sử dụng

### Khởi động API

```bash
python -m api.main
```

Hoặc sử dụng uvicorn trực tiếp:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: `http://localhost:8000`

### API Documentation

Truy cập Swagger UI tại: `http://localhost:8000/docs`

Hoặc ReDoc tại: `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Kiểm tra health của toàn bộ system (Ollama, vector store).

**Response:**
```json
{
  "status": "healthy",
  "ollama": {
    "status": "healthy",
    "ollama_running": true,
    "model_available": true,
    "model_name": "qwen2.5:30b"
  },
  "vector_store": {
    "status": "healthy",
    "num_documents": 15000,
    "embedding_dim": 768
  }
}
```

**GET** `/api/health/vector-store`

Kiểm tra health của vector store.

### 2. Chat

**POST** `/api/chat`

Chat với RAG system.

**Request:**
```json
{
  "question": "Người sử dụng lao động có phải thiết lập cơ chế đối thoại với người lao động không?",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": null
}
```

**Response:**
```json
{
  "answer": "Có, theo quy định của Bộ luật Lao động 2019...",
  "sources": [
    {
      "id": "doc_123",
      "text": "Nội dung văn bản pháp luật...",
      "source": "vn-law-corpus",
      "score": 0.85,
      "rank": 1,
      "chunk_index": 0
    }
  ],
  "metadata": {
    "retrieved_count": 5,
    "top_k": 5,
    "temperature": 0.7
  }
}
```

**POST** `/api/chat/stream`

Chat với streaming response (Server-Sent Events).

**Request:** Giống như `/api/chat`

**Response:** Stream với format:
```
data: {"type": "sources", "content": [...], "done": false}

data: {"type": "answer", "content": "Có, theo quy định...", "done": false}

data: {"type": "answer", "content": "", "done": true}
```

## Ví dụ sử dụng

### Python

```python
import requests

# Chat request
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "question": "Quy định về thời gian làm việc của người lao động",
        "top_k": 5,
        "temperature": 0.7
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quy định về thời gian làm việc của người lao động",
    "top_k": 5,
    "temperature": 0.7
  }'
```

### JavaScript (Fetch)

```javascript
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'Quy định về thời gian làm việc của người lao động',
    top_k: 5,
    temperature: 0.7
  })
});

const result = await response.json();
console.log(result.answer);
```

### Streaming (JavaScript)

```javascript
const response = await fetch('http://localhost:8000/api/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'Quy định về thời gian làm việc của người lao động'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'answer') {
        process.stdout.write(data.content);
      }
    }
  }
}
```

## Cấu trúc dự án

```
vn_law/
├── build/                    # RAG system core
│   ├── config.py            # Configuration
│   ├── ollama_client.py      # Ollama client
│   ├── prompts.py            # Prompt templates
│   └── rag_chain.py          # RAG chain
├── api/                      # FastAPI application
│   ├── main.py              # FastAPI app
│   ├── routes/              # API routes
│   │   ├── chat.py          # Chat endpoints
│   │   └── health.py        # Health check
│   ├── models/              # Pydantic schemas
│   │   └── schemas.py
│   └── dependencies.py      # Dependency injection
├── src/                      # Existing embedding system
├── vector_db/               # FAISS vector store
└── requirements_rag.txt     # Dependencies
```

## Troubleshooting

### Ollama không kết nối được

1. Kiểm tra Ollama đang chạy:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Kiểm tra model có sẵn:
   ```bash
   ollama list
   ```

3. Nếu chưa có model, pull model:
   ```bash
   ollama pull qwen2.5:30b
   ```

### Vector store không tìm thấy

1. Đảm bảo đã chạy `scripts/build_index.py` trước
2. Kiểm tra `VECTOR_STORE_DIR` trong `.env` đúng đường dẫn
3. Kiểm tra file `vector_db/index.faiss` tồn tại

### Lỗi import modules

Đảm bảo đã cài đặt tất cả dependencies:

```bash
pip install -r requirements_rag.txt
```

## Performance Tips

1. **Sử dụng GPU cho Ollama**: Qwen30B chạy tốt hơn với GPU
2. **Tối ưu top_k**: Giảm `top_k` nếu cần tốc độ nhanh hơn
3. **Streaming**: Sử dụng `/api/chat/stream` cho UX tốt hơn
4. **Caching**: Có thể thêm caching cho các câu hỏi thường gặp

## License

Apache 2.0

