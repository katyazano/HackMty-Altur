import { DetectResponse } from '../types/detection';

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

        const t0 = performance.now();
        const response = await fetch(endpoint, {
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
