import { DetectResponse } from '../types/detection';

// Defaults to Backend on Port 8000, or custom VITE_API_BASE_URL if configured
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

export async function detectAudio(
  file: File,
  endpoint: string
): Promise<{ data: DetectResponse; latency: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read audio file'));
    reader.onload = async () => {
      try {
        const result = reader.result as string;
        const base64Audio = result.split(',')[1];
        const payload = {
          audio: base64Audio,
          audio_base64: base64Audio,
          call_id: 'test_' + Date.now().toString(36)
        };

        const targetUrl = endpoint.startsWith('http')
          ? endpoint
          : `${API_BASE_URL}${endpoint}`;

        const t0 = performance.now();
        const response = await fetch(targetUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const latency = Math.round(performance.now() - t0);

        if (!response.ok) {
          const errData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
          return reject({ ...errData, latency });
        }

        const data: DetectResponse = await response.json();
        resolve({ data, latency });
      } catch (err: unknown) {
        reject(err);
      }
    };
    reader.readAsDataURL(file);
  });
}
