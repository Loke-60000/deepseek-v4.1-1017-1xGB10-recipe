import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from recipe.probe import Probe


class FakeApi(BaseHTTPRequestHandler):
    answer = "391"

    def do_GET(self):
        self._send({"data": [{"id": "m"}, {"id": "alias"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"choices": [{"message": {"content": self.answer}}], "usage": {"completion_tokens": 3}})

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), FakeApi)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.probe = Probe(self.server.server_port, "m")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_ready_lists_models_and_smoke_passes(self):
        self.assertEqual(self.probe.wait_ready(timeout=5), ["m", "alias"])
        self.assertEqual(self.probe.smoke().completion_tokens, 3)

    def test_wrong_answer_fails(self):
        FakeApi.answer = "392"
        try:
            with self.assertRaises(AssertionError):
                self.probe.smoke()
        finally:
            FakeApi.answer = "391"


if __name__ == "__main__":
    unittest.main()
