import React, { useRef, useState } from 'react';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
  fileName?: string;
  fileSub?: string;
  hasError?: boolean;
  hasSuccess?: boolean;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  onFileSelect,
  isLoading,
  fileName = 'Upload Test Audio',
  fileSub = '.WAV • 8kHz / 16kHz',
  hasError = false,
  hasSuccess = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const renderIcon = () => {
    if (isLoading) {
      return <i className="fa-solid fa-spinner fa-spin"></i>;
    }
    if (hasError) {
      return <i className="fa-solid fa-triangle-exclamation text-[#FF5F56]"></i>;
    }
    if (hasSuccess) {
      return <i className="fa-solid fa-circle-check text-[#00E599]"></i>;
    }
    return <i className="fa-solid fa-cloud-arrow-up"></i>;
  };

  return (
    <div
      className={`upload-sub-box ${isDragOver ? 'drag-over' : ''}`}
      onClick={() => fileInputRef.current?.click()}
      onDragEnter={handleDragOver}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,audio/wav"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <div className="upload-icon-circle">{renderIcon()}</div>
      <div>
        <div className="upload-label-main">{fileName}</div>
        <div className="upload-label-sub">{fileSub}</div>
      </div>
      <button
        type="button"
        className="btn-choose-file"
        onClick={(e) => {
          e.stopPropagation();
          fileInputRef.current?.click();
        }}
      >
        Choose File
      </button>
    </div>
  );
};
