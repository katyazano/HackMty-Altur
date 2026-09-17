/* ==============================================================================
   ALTUR VAULT — SHARED CLIENT UTILITIES & AUDIO HANDLERS
   ============================================================================== */

// Generates high-fidelity test audio presets (8kHz 16-bit PCM Stereo)
function generateAudioPreset(type) {
  const sampleRate = 8000;
  const durationSec = 3.5;
  const numSamples = sampleRate * durationSec;
  const buffer = new ArrayBuffer(44 + numSamples * 4);
  const view = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 4, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 2, true); // 2 channels (Stereo)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 4, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    let caller = 0;
    let agent = 0.15 * Math.sin(2 * Math.PI * 160 * t);

    if (type === 'synthetic_scam') {
      // Flat robotic buzz without pitch micro-jitter (Synthetic Vocoder)
      caller = 0.35 * Math.sin(2 * Math.PI * 220 * t) + 0.18 * Math.sin(2 * Math.PI * 440 * t) + 0.08 * Math.sin(2 * Math.PI * 880 * t);
    } else {
      // Natural human speech: modulated fundamental frequency + micro-jitter
      const f0 = 135 + 18 * Math.sin(2 * Math.PI * 1.4 * t) + (Math.random() - 0.5) * 4;
      caller = 0.35 * Math.sin(2 * Math.PI * f0 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 2.2 * t));
    }

    view.setInt16(offset, Math.max(-32768, Math.min(32767, caller * 32767)), true);
    view.setInt16(offset + 2, Math.max(-32768, Math.min(32767, agent * 32767)), true);
    offset += 4;
  }

  const blob = new Blob([buffer], { type: 'audio/wav' });
  return new File([blob], `${type}_call.wav`, { type: 'audio/wav' });
}

// Configures drag & drop for audio inputs
function initDropzone(dropzoneId, inputId, onFileSelected) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  if (!dropzone || !input) return;

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener('change', (e) => {
    if (e.target.files.length) {
      onFileSelected(e.target.files[0]);
    }
  });
}
