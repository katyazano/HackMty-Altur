/* ==============================================================================
   ALTUR VAULT — SHARED CLIENT UTILITIES, WAVEFORM CANVAS & AUDIO HANDLERS
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
      caller = 0.38 * Math.sin(2 * Math.PI * 220 * t) + 0.20 * Math.sin(2 * Math.PI * 440 * t) + 0.09 * Math.sin(2 * Math.PI * 880 * t);
    } else {
      // Natural human speech: modulated fundamental frequency + micro-jitter
      const f0 = 135 + 18 * Math.sin(2 * Math.PI * 1.4 * t) + (Math.random() - 0.5) * 5;
      caller = 0.38 * Math.sin(2 * Math.PI * f0 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 2.2 * t));
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

// Interactive Web Audio Waveform Renderer
class TelephonyWaveformVisualizer {
  constructor(canvasId, audioPlayerId) {
    this.canvas = document.getElementById(canvasId);
    this.audio = document.getElementById(audioPlayerId);
    if (!this.canvas || !this.audio) return;
    this.ctx = this.canvas.getContext('2d');
    this.audioCtx = null;
    this.audioBuffer = null;
    this.isPlaying = false;
    this.animId = null;

    this.audio.addEventListener('timeupdate', () => this.draw());
    this.audio.addEventListener('play', () => this.startAnimation());
    this.audio.addEventListener('pause', () => this.stopAnimation());
    this.audio.addEventListener('ended', () => this.stopAnimation());

    // Click on canvas to scrub
    this.canvas.addEventListener('click', (e) => {
      if (!this.audio.duration) return;
      const rect = this.canvas.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      this.audio.currentTime = pos * this.audio.duration;
      this.draw();
    });
  }

  async loadFromBase64(base64Data) {
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const res = await fetch(base64Data);
      const arrayBuffer = await res.arrayBuffer();
      this.audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
      this.draw();
    } catch (err) {
      console.warn('Waveform decode error:', err);
    }
  }

  startAnimation() {
    this.isPlaying = true;
    const loop = () => {
      if (!this.isPlaying) return;
      this.draw();
      this.animId = requestAnimationFrame(loop);
    };
    loop();
  }

  stopAnimation() {
    this.isPlaying = false;
    if (this.animId) cancelAnimationFrame(this.animId);
    this.draw();
  }

  draw() {
    if (!this.canvas || !this.ctx) return;
    const width = this.canvas.width = this.canvas.offsetWidth * (window.devicePixelRatio || 1);
    const height = this.canvas.height = this.canvas.offsetHeight * (window.devicePixelRatio || 1);
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    // Background grid lines
    ctx.strokeStyle = '#F1F5F9';
    ctx.lineWidth = 1;
    for (let y = height / 4; y < height; y += height / 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!this.audioBuffer) {
      // Idle line
      ctx.strokeStyle = '#CBD5E1';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const numChannels = this.audioBuffer.numberOfChannels;
    const callerData = this.audioBuffer.getChannelData(0);
    const agentData = numChannels > 1 ? this.audioBuffer.getChannelData(1) : null;

    const step = Math.ceil(callerData.length / width);
    const progress = this.audio.duration ? (this.audio.currentTime / this.audio.duration) : 0;
    const progressX = width * progress;

    // Draw Channel 1 (Agent - Subdued Slate)
    if (agentData) {
      ctx.strokeStyle = '#94A3B8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < width; i++) {
        let min = 1.0, max = -1.0;
        for (let j = 0; j < step; j++) {
          const datum = agentData[(i * step) + j];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }
        const yTop = ((1 + min) * 0.5) * height;
        const yBot = ((1 + max) * 0.5) * height;
        ctx.moveTo(i, yTop);
        ctx.lineTo(i, yBot);
      }
      ctx.stroke();
    }

    // Draw Channel 0 (Caller - Primary Blue / Highlighted)
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = callerData[(i * step) + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      const yTop = ((1 + min) * 0.5) * height;
      const yBot = ((1 + max) * 0.5) * height;

      ctx.strokeStyle = i < progressX ? '#2563EB' : '#93C5FD';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(i, yTop);
      ctx.lineTo(i, yBot);
      ctx.stroke();
    }

    // Draw Scrubber Line
    ctx.strokeStyle = '#0F172A';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(progressX, 0);
    ctx.lineTo(progressX, height);
    ctx.stroke();
  }
}

// Copy to clipboard helper
function copyToClipboard(text, btnElement) {
  navigator.clipboard.writeText(text).then(() => {
    if (btnElement) {
      const originalHTML = btnElement.innerHTML;
      btnElement.innerHTML = '<i class="fa-solid fa-check text-emerald-600"></i> Copied';
      setTimeout(() => { btnElement.innerHTML = originalHTML; }, 2000);
    }
  });
}

