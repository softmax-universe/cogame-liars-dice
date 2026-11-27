#!/usr/bin/env python3
"""Deterministic local OpenAI-compatible HTTP fixture for credential-free CI only.

NEVER reference this endpoint in coworld_manifest_template.json or any published
manifest. It exists solely to let credential-free CI (`docker_smoke.sh` and
local `coworld certify`) exercise the real `/bin/liars-dice-player` HTTP/JSON
model transport end to end without provider secrets.
"""

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(reply_mode: str, log_path: Path | None):
    class FixtureHandler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_req = self.rfile.read(length)
            req = json.loads(raw_req.decode("utf-8"))
            messages = req.get("messages") or []
            user_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_text = str(msg.get("content") or "")
            is_opening = "STANDING BID: (none" in user_text
            if reply_mode == "invalid":
                content = "I refuse to return JSON."
            elif reply_mode == "illegal":
                content = json.dumps({"action": "challenge", "say": "illegal open"})
            elif reply_mode == "refusal":
                content = ""
            else:
                if is_opening:
                    action = {
                        "action": "bid",
                        "quantity": 1,
                        "face": 1,
                        "say": "opening bid",
                        "notes": "ci fixture open",
                    }
                else:
                    action = {
                        "action": "challenge",
                        "say": "calling it",
                        "notes": "ci fixture challenge",
                    }
                encoded = json.dumps(action, separators=(",", ":"))
                if reply_mode == "fenced":
                    content = f"Here is my move:\n```json\n{encoded}\n```\nDone."
                else:
                    content = encoded

            call_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            message_obj: dict[str, object] = {"role": "assistant", "content": content}
            finish_reason = "stop"
            if reply_mode == "refusal":
                message_obj["refusal"] = "policy refusal fixture"
                finish_reason = "content_filter"
            response = {
                "id": call_id,
                "object": "chat.completion",
                "model": req.get("model") or "fixture/mock",
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 64, "completion_tokens": 16, "total_tokens": 80},
            }
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "id": call_id,
                                "slot": self.headers.get("X-Coworld-Player-Slot"),
                                "model": req.get("model"),
                                "is_opening": is_opening,
                                "max_tokens": req["max_tokens"],
                                "reply_mode": reply_mode,
                            }
                        )
                        + "\n"
                    )
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return FixtureHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--mode",
        choices=("valid", "fenced", "invalid", "illegal", "refusal"),
        default="valid",
    )
    args = parser.parse_args()
    log_path = Path(args.log_file) if args.log_file else None
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.mode, log_path))
    actual_port = server.server_port
    if args.port_file:
        Path(args.port_file).write_text(str(actual_port), encoding="utf-8")
    print(f"mock_llm_server listening on {args.host}:{actual_port} (mode={args.mode})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
