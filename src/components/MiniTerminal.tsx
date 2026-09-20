import React from 'react';

interface MiniTerminalProps {
  isExample: boolean;
  endpointTag: string;
  responseContent: object | string | null;
  isLoading?: boolean;
}

export const MiniTerminal: React.FC<MiniTerminalProps> = ({
  isExample,
  endpointTag,
  responseContent,
  isLoading = false,
}) => {
  const renderContent = () => {
    if (isLoading) {
      return <span style={{ color: '#7B8290' }}>// Running inference...</span>;
    }

    if (typeof responseContent === 'string') {
      return <span>{responseContent}</span>;
    }

    if (!responseContent) {
      return <span>{'{}'}</span>;
    }

    const jsonStr = JSON.stringify(responseContent, null, 2);
    // Custom syntax highlighting tokens
    const lines = jsonStr.split('\n');

    return (
      <>
        {lines.map((line, idx) => {
          // Replace keys, strings, numbers, booleans with colored spans
          const highlighted = line
            .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*")(\s*:)?/g, (_match, p1, _p2, p3) => {
              if (p3) {
                return `<span class="json-key">${p1}</span>:`;
              }
              return `<span class="json-str">${p1}</span>`;
            })
            .replace(/\b(true|false)\b/g, '<span class="json-bool">$1</span>')
            .replace(/\b(-?\d+(?:\.\d+)?)\b/g, '<span class="json-num">$1</span>');

          return (
            <div
              key={idx}
              dangerouslySetInnerHTML={{ __html: highlighted }}
            />
          );
        })}
      </>
    );
  };

  return (
    <div className={`mini-terminal-box ${isExample ? 'is-example' : ''}`}>
      <div className="mini-terminal-header">
        <div className="terminal-dots-row">
          <div className="terminal-dot dot-red" />
          <div className="terminal-dot dot-yellow" />
          <div className="terminal-dot dot-green" />
        </div>
        <span className="terminal-title-tag">{endpointTag}</span>
      </div>
      <pre className="mini-terminal-body">{renderContent()}</pre>
    </div>
  );
};
