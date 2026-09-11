import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

class HelloWorldHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Hello, World!\n")

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}", flush=True)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), HelloWorldHandler) as httpd:
        print(f"Serving HTTP on 0.0.0.0 port {PORT} (http://localhost:{PORT}/)...", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...", flush=True)
