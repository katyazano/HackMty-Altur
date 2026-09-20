import React, { useState } from 'react';
import { EngineConfig, DetectResponse } from '../types/detection';
import { detectAudio } from '../services/api';
import { UploadDropzone } from './UploadDropzone';
import { MiniTerminal } from './MiniTerminal';
import { EndpointBar } from './EndpointBar';

interface EngineCardProps {
  config: EngineConfig;
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
  className?: string;
}

export const EngineCard: React.FC<EngineCardProps> = ({
  config,
  onNavigate,
  className = '',
}) => {
  const [isExample, setIsExample] = useState(true);
  const [terminalTag, setTerminalTag] = useState('example');
  const [terminalData, setTerminalData] = useState<DetectResponse | object>(
    config.exampleResponse
  );
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('Upload Test Audio');
  const [fileSub, setFileSub] = useState('.WAV • 8kHz / 16kHz');
  const [hasError, setHasError] = useState(false);
  const [hasSuccess, setHasSuccess] = useState(false);

  const handleAudioUpload = async (file: File) => {
    setIsExample(false);
    setIsLoading(true);
    setHasError(false);
    setHasSuccess(false);
    setFileName(file.name);
    setFileSub(`${(file.size / 1024).toFixed(1)} KB • Processing...`);
    setTerminalTag(`POST ${config.endpoint}`);

    try {
      const { data, latency } = await detectAudio(file, config.endpoint);
      setTerminalData(data);
      setIsLoading(false);
      setHasSuccess(true);
      setFileSub(
        file.name.endsWith('.wav')
          ? `.WAV • ${latency}ms`
          : `${latency}ms`
      );
    } catch (err: unknown) {
      setIsLoading(false);
      setHasError(true);
      setTerminalTag(`POST ${config.endpoint} (ERR)`);
      const errorObj = typeof err === 'object' && err !== null ? err : { error: String(err) };
      setTerminalData(errorObj);
      const latencyVal = (err as { latency?: number })?.latency;
      setFileSub(latencyVal ? `Error • ${latencyVal}ms` : 'Error');
    }
  };

  const handleDocsClick = (e: React.MouseEvent) => {
    if (onNavigate) {
      e.preventDefault();
      onNavigate(config.id === 'baseline' ? 'engine1' : 'engine2');
    }
  };

  return (
    <div className={className}>
      {/* Header & Model Details */}
      <div className="engine-card-header">
        <div className="card-top-bar">
          <span className="engine-tag-label">{config.titleTag}</span>
          <a
            href={`#/${config.id === 'baseline' ? 'engine1' : 'engine2'}`}
            onClick={handleDocsClick}
            className="btn-view-docs"
            title="View full documentation"
          >
            <span>View docs</span>
            <i className="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
          </a>
        </div>
        <h2 className="engine-card-title">{config.title}</h2>
        <p className="engine-card-desc">{config.description}</p>

        {/* Component Models Row */}
        <div className="component-chips-row">
          {config.chips.map((chip, idx) => (
            <span key={idx} className="component-chip">
              <i
                className={`${chip.icon} ${
                  chip.highlight ? 'text-[#FF5500]' : 'text-[#555B67]'
                }`}
              ></i>{' '}
              {chip.label}
            </span>
          ))}
        </div>
      </div>

      {/* Workbench: Upload + Mini Terminal */}
      <div className="engine-workbench-grid">
        <UploadDropzone
          onFileSelect={handleAudioUpload}
          isLoading={isLoading}
          fileName={fileName}
          fileSub={fileSub}
          hasError={hasError}
          hasSuccess={hasSuccess}
        />

        <MiniTerminal
          isExample={isExample}
          endpointTag={terminalTag}
          responseContent={terminalData}
          isLoading={isLoading}
        />
      </div>

      {/* Endpoint Bar */}
      <EndpointBar endpoint={config.endpoint} method="POST" />
    </div>
  );
};
