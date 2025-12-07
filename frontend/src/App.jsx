import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';
import Loading from './components/Loading';
import { checkHealth } from './services/api';
import './styles/App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    // Check API health on mount
    checkHealth()
      .then((status) => {
        setHealthStatus(status);
        setIsLoading(false);
        
        // Add welcome message
        if (status.status === 'healthy') {
          setMessages([
            {
              type: 'assistant',
              content: {
                answer: 'Xin chào! Tôi là trợ lý pháp luật Việt Nam. Tôi có thể giúp bạn trả lời các câu hỏi về pháp luật. Hãy đặt câu hỏi của bạn!',
                sources: [],
              },
            },
          ]);
        } else {
          setMessages([
            {
              type: 'assistant',
              content: {
                answer: `Cảnh báo: Hệ thống đang gặp sự cố. Trạng thái: ${status.status}. Vui lòng kiểm tra lại server.`,
                sources: [],
              },
            },
          ]);
        }
      })
      .catch((error) => {
        console.error('Health check failed:', error);
        setIsLoading(false);
        setMessages([
          {
            type: 'assistant',
            content: {
              answer: `Không thể kết nối với server: ${error.message}. Vui lòng đảm bảo API đang chạy tại http://localhost:8000`,
              sources: [],
            },
          },
        ]);
      });
  }, []);

  const handleNewMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  if (isLoading) {
    return (
      <div className="app-container">
        <div className="app-loading">
          <Loading />
          <p>Đang kiểm tra kết nối...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>VN Law RAG - Trợ lý Pháp luật</h1>
        {healthStatus && (
          <div className={`health-status ${healthStatus.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
            {healthStatus.status === 'healthy' ? '✓ Hệ thống hoạt động bình thường' : '⚠ Hệ thống có vấn đề'}
          </div>
        )}
      </header>
      
      <main className="app-main">
        <Chat
          messages={messages}
          onNewMessage={handleNewMessage}
          isLoading={isLoading}
        />
      </main>
      
      <footer className="app-footer">
        <p>Hệ thống RAG cho Pháp luật Việt Nam | Powered by Ollama & LangChain</p>
      </footer>
    </div>
  );
}

export default App;

