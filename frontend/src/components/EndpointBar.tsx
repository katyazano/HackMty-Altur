import React, { useState } from 'react';

interface EndpointBarProps {
  endpoint: string;
  method?: string;
}

export const EndpointBar: React.FC<EndpointBarProps> = ({
  endpoint,
  method = 'POST',
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    const urlToCopy =
      window.location.origin &&
      window.location.origin !== 'null' &&
      !window.location.origin.startsWith('file:')
        ? window.location.origin + endpoint
        : endpoint;

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(urlToCopy).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }).catch(() => {
        fallbackCopy(urlToCopy);
      });
    } else {
      fallbackCopy(urlToCopy);
    }
  };

  const fallbackCopy = (text: string) => {
    const input = document.createElement('input');
    input.value = text;
    document.body.appendChild(input);
    input.select();
    try {
      document.execCommand('copy');
    } catch {
      // ignore fallback error
    }
    document.body.removeChild(input);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="engine-endpoint-bar">
      <div className="endpoint-info-group">
        <span className="endpoint-method-badge">{method}</span>
        <span className="endpoint-path-text">{endpoint}</span>
      </div>
      <button
        type="button"
        className={`btn-copy-endpoint ${copied ? 'copied' : ''}`}
        onClick={handleCopy}
        title="Copy endpoint"
      >
        {copied ? (
          <>
            <i className="fa-solid fa-check"></i>
            <span>Copied!</span>
          </>
        ) : (
          <>
            <i className="fa-regular fa-copy"></i>
            <span>Copy</span>
          </>
        )}
      </button>
    </div>
  );
};
