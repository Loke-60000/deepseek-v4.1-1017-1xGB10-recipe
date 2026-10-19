import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recipe.bench.speed import SpeedBench


class FakeStream(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(5):
            event = {"choices": [{"delta": {"content": f"t{i}"}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 5}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class SpeedTest(unittest.TestCase):
    def test_counts_tokens_and_concurrency(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeStream)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            result = SpeedBench(server.server_port, "m").run("prose", 3)
        finally:
            server.shutdown()
        self.assertEqual(len(result.streams), 3)
        self.assertTrue(all(s.tokens == 5 for s in result.streams))
        self.assertGreater(result.aggregate, 0)


if __name__ == "__main__":
    unittest.main()
