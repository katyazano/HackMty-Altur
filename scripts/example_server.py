"""Minimal example server implementing the Altur Voice Anti-Spoofing challenge contract.

Usage:
    python scripts/example_server.py --port 8000

Standard library only. Returns a placeholder decision (is_synthetic: False, confidence: 0.50).
"""

import argparse
import base64
import http.server
import json
import socketserver


class DetectHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "status": "healthy",
                    "service": "Altur Voice Anti-Spoofing Example Server",
                    "version": "1.0.0"
                }).encode("utf-8")
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/detect":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body.decode("utf-8"))
                call_id = data.get("call_id", "")
                audio_b64 = data.get("audio_base64") or data.get("audio", "")
                sample_rate = data.get("sample_rate", 8000)
                channels = data.get("channels", 2)

                # Decode raw WAV bytes
                raw_bytes = base64.b64decode(audio_b64)

                # Placeholder response: Classify as human (False) with 50% confidence
                response = {
                    "is_synthetic": False,
                    "confidence": 0.50
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Run example anti-spoofing server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"Starting example server on http://{args.host}:{args.port} (POST /detect)...")
    with socketserver.TCPServer((args.host, args.port), DetectHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
