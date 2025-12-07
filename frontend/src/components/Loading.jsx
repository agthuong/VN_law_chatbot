import React from 'react';
import './Loading.css';

/**
 * Loading component
 */
export default function Loading() {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
      </div>
      <div className="loading-text">Đang xử lý...</div>
    </div>
  );
}

