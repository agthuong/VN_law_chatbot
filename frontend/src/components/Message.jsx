import React from 'react';
import './Message.css';

/**
 * Message component to display chat messages
 */
export default function Message({ message, isUser = false }) {
  if (!message) return null;

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-content">
        {isUser ? (
          <div className="message-text">{message}</div>
        ) : (
          <>
            <div className="message-text">{message.answer || message}</div>
            {message.sources && message.sources.length > 0 && (
              <div className="message-sources">
                <div className="sources-header">Nguồn tham khảo:</div>
                {message.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <span className="source-rank">#{source.rank || index + 1}</span>
                    {source.score && (
                      <span className="source-score">({(source.score * 100).toFixed(1)}%)</span>
                    )}
                    <div className="source-text">{source.text.substring(0, 200)}...</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

