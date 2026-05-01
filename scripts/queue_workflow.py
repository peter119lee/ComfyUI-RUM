from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue a ComfyUI API workflow JSON.")
    parser.add_argument("workflow", type=Path, help="Path to API-format workflow JSON.")
    parser.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL.")
    parser.add_argument("--client-id", default=None, help="Optional ComfyUI client id.")
    args = parser.parse_args()

    workflow_path = args.workflow.expanduser()
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    payload = {
        "prompt": prompt,
        "client_id": args.client_id or str(uuid.uuid4()),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.server.rstrip("/") + "/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach ComfyUI API at {args.server}: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
