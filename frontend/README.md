# VN Law RAG Frontend

Frontend React đơn giản để tương tác với VN Law RAG API.

## Yêu cầu

- Node.js 16+ và npm/yarn
- API server đang chạy tại `http://localhost:8000`

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd frontend
npm install
```

### 2. Cấu hình (tùy chọn)

Tạo file `.env` trong thư mục `frontend/` nếu API chạy ở URL khác:

```env
VITE_API_URL=http://localhost:8000
```

## Sử dụng

### Development mode

```bash
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

### Build cho production

```bash
npm run build
```

Files build sẽ được tạo trong thư mục `dist/`

### Preview production build

```bash
npm run preview
```

## Tính năng

- ✅ Chat interface đơn giản
- ✅ Streaming response (real-time)
- ✅ Hiển thị sources (nguồn tham khảo)
- ✅ Health check status
- ✅ Responsive design

## Cấu trúc

```
frontend/
├── src/
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # React entry point
│   ├── components/
│   │   ├── Chat.jsx         # Chat interface
│   │   ├── Message.jsx      # Message display
│   │   └── Loading.jsx       # Loading indicator
│   ├── services/
│   │   └── api.js           # API service functions
│   └── styles/
│       └── App.css          # Main styles
├── index.html
├── package.json
└── vite.config.js
```

## API Integration

Frontend gọi các endpoints sau:

- `POST /api/chat` - Chat không streaming
- `POST /api/chat/stream` - Chat với streaming (SSE)
- `GET /api/health` - Health check

## Troubleshooting

### Không kết nối được với API

1. Đảm bảo API server đang chạy tại `http://localhost:8000`
2. Kiểm tra CORS settings trong FastAPI
3. Kiểm tra file `.env` nếu đã thay đổi API URL

### Lỗi khi build

```bash
# Xóa node_modules và cài lại
rm -rf node_modules package-lock.json
npm install
```

## License

Apache 2.0

