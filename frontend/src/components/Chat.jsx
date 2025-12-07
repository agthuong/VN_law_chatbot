import React, { useState, useRef, useEffect } from 'react';
import { chatStream } from '../services/api';
import Message from './Message';
import Loading from './Loading';
import './Chat.css';

/**
 * Main Chat component
 */
export default function Chat({ messages, onNewMessage, isLoading }) {
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [streamingSources, setStreamingSources] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingAnswer]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading || isStreaming) {
      return;
    }

    const question = input.trim();
    setInput('');
    
    // Add user message
    onNewMessage({
      type: 'user',
      content: question,
    });

    setIsStreaming(true);
    setStreamingAnswer('');
    setStreamingSources(null);

    try {
      let fullAnswer = '';
      let sources = null;

      await chatStream(
        question,
        { top_k: 5, temperature: 0.7 },
        (chunk) => {
          if (chunk.type === 'sources') {
            setStreamingSources(chunk.content);
            sources = chunk.content;
          } else if (chunk.type === 'answer') {
            if (chunk.content) {
              fullAnswer += chunk.content;
              setStreamingAnswer(fullAnswer);
            }
            if (chunk.done) {
              setIsStreaming(false);
              onNewMessage({
                type: 'assistant',
                content: {
                  answer: fullAnswer,
                  sources: sources,
                },
              });
              setStreamingAnswer('');
              setStreamingSources(null);
            }
          } else if (chunk.type === 'error') {
            setIsStreaming(false);
            onNewMessage({
              type: 'assistant',
              content: {
                answer: `Lỗi: ${chunk.content}`,
                sources: [],
              },
            });
            setStreamingAnswer('');
            setStreamingSources(null);
          }
        }
      );
    } catch (error) {
      setIsStreaming(false);
      onNewMessage({
        type: 'assistant',
        content: {
          answer: `Lỗi khi kết nối với server: ${error.message}`,
          sources: [],
        },
      });
      setStreamingAnswer('');
      setStreamingSources(null);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <Message
            key={index}
            message={msg.type === 'user' ? msg.content : msg.content}
            isUser={msg.type === 'user'}
          />
        ))}
        
        {isStreaming && (
          <div className="streaming-message">
            <Message
              message={{
                answer: streamingAnswer || 'Đang suy nghĩ...',
                sources: streamingSources,
              }}
              isUser={false}
            />
            {streamingAnswer && (
              <div className="streaming-indicator">●</div>
            )}
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Nhập câu hỏi về pháp luật Việt Nam..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading || isStreaming}
        />
        <button
          type="submit"
          className="chat-submit"
          disabled={!input.trim() || isLoading || isStreaming}
        >
          Gửi
        </button>
      </form>
    </div>
  );
}

