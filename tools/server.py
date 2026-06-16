#!/usr/bin/env python3
"""
Dataset Tools Server  —  landing page, 3D Viewer, and Dataset Wizard.

Usage:
    python tools/server.py [--port 8080] [--cache /path/to/datasets]
    dataset-server          [--port 8080] [--cache /path/to/datasets]

    http://localhost:8080         → Landing page
    http://localhost:8080/viewer  → 3D Robot Dataset Viewer
    http://localhost:8080/wizard  → Dataset Wizard
"""
import argparse
import json
import mimetypes
import os
import queue
import subprocess
import sys
import threading
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

TOOLS_DIR     = Path(__file__).parent
VIEWER_DIR    = TOOLS_DIR / "viewer"
CONFIG_PATH   = TOOLS_DIR / "config.yaml"
WIZARD_SCRIPT = TOOLS_DIR / "dataset-wizard.py"
TROSSEN_DIR   = TOOLS_DIR / "assets" / "trossen_arm_description"
URDF_PATH     = TROSSEN_DIR / "urdf" / "generated" / "mobile_ai.urdf"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css",               ".css")
mimetypes.add_type("model/stl",              ".stl")
mimetypes.add_type("model/vnd.collada+xml",  ".dae")
mimetypes.add_type("image/png",              ".png")

# ── Job management ─────────────────────────────────────────────────────────────

_jobs: dict = {}
_jobs_lock  = threading.Lock()


def _run_job(job_id: str, cmd: list, cwd: str):
    q = _jobs[job_id]["queue"]
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, "FORCE_COLOR": "1", "COLORTERM": "truecolor",
                 "PYTHONUNBUFFERED": "1"},
        )
        _jobs[job_id]["process"] = proc

        # Auto-answer "y" to any Rich Confirm.ask() prompts
        def _feed():
            try:
                for _ in range(20):
                    proc.stdin.write("y\n")
                    proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass
        threading.Thread(target=_feed, daemon=True).start()

        for line in proc.stdout:
            line = line.rstrip()
            if '\r' in line:
                line = line.split('\r')[-1]
            q.put(line)
        proc.wait()
        _jobs[job_id]["returncode"] = proc.returncode
    except Exception as exc:
        q.put(f"[ERROR] {exc}")
        _jobs[job_id]["returncode"] = 1
    finally:
        _jobs[job_id]["done"] = True
        q.put(None)  # sentinel


# ── Dataset helpers ────────────────────────────────────────────────────────────

def find_datasets(cache_dir: Path):
    """Return ALL LeRobot datasets under cache_dir (rglob)."""
    results = []
    for info_path in sorted(cache_dir.rglob("meta/info.json")):
        try:
            info  = json.loads(info_path.read_text())
            feats = info.get("features", {})
            root  = info_path.parent.parent
            try:
                name = str(root.relative_to(cache_dir))
            except ValueError:
                name = root.name
            results.append({
                "name":               name,
                "path":               str(root),
                "total_episodes":     info.get("total_episodes", 0),
                "total_frames":       info.get("total_frames",   0),
                "has_ee_left":        "observation.ee_left"     in feats,
                "has_ee_right":       "observation.ee_right"    in feats,
                "has_action_ee":      "action.ee_left"          in feats,
                "has_ee_delta":       "action.ee_left.delta"    in feats,
                "has_ee_relative":    "action.ee_left.relative" in feats,
                "has_joint_delta":    "action.delta"            in feats,
                "has_joint_relative": "action.relative"         in feats,
                "state_names":        feats.get("observation.state", {}).get("names", []),
                "action_names":       feats.get("action",            {}).get("names", []),
            })
        except Exception as e:
            print(f"  [skip] {info_path}: {e}")
    return results


def find_direct_datasets(base_path: Path):
    """Find dataset dirs that are direct children of base_path."""
    results = []
    if not base_path.is_dir():
        return results
    for child in sorted(base_path.iterdir()):
        info_path = child / "meta" / "info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                results.append({
                    "name":           child.name,
                    "path":           str(child),
                    "total_episodes": info.get("total_episodes", 0),
                    "total_frames":   info.get("total_frames",   0),
                })
            except Exception:
                pass
    return results


def get_episodes(dataset_path: str):
    root     = Path(dataset_path)
    ep_count = {}
    for pq_file in sorted((root / "data").rglob("*.parquet")):
        schema = pq.read_schema(pq_file)
        if "episode_index" not in schema.names:
            continue
        for ep in pq.read_table(pq_file, columns=["episode_index"])["episode_index"].to_pylist():
            ep_count[ep] = ep_count.get(ep, 0) + 1
    return [{"episode": ep, "frames": cnt} for ep, cnt in sorted(ep_count.items())]


def get_frames(dataset_path: str, episode_idx: int):
    root = Path(dataset_path)
    WANT = [
        "observation.state", "action", "timestamp", "frame_index",
        "observation.ee_left",       "observation.ee_right",
        "action.ee_left",            "action.ee_right",
        "action.ee_left.delta",      "action.ee_right.delta",
        "action.ee_left.relative",   "action.ee_right.relative",
        "action.delta",              "action.relative",
    ]
    chunks = []
    for pq_file in sorted((root / "data").rglob("*.parquet")):
        schema = pq.read_schema(pq_file)
        if "episode_index" not in schema.names:
            continue
        cols     = ["episode_index"] + [c for c in WANT if c in schema.names]
        tbl      = pq.read_table(pq_file, columns=cols)
        filtered = tbl.filter(pc.equal(tbl["episode_index"], episode_idx))
        if len(filtered):
            keep = [c for c in filtered.schema.names if c != "episode_index"]
            chunks.append(filtered.select(keep))

    if not chunks:
        return {"frames": [], "error": f"episode {episode_idx} not found"}

    combined = pa.concat_tables(chunks, promote_options="default")
    if "frame_index" in combined.schema.names:
        combined = combined.sort_by("frame_index")

    return {
        "frames": [
            {col: combined[col][i].as_py() for col in combined.schema.names}
            for i in range(len(combined))
        ],
        "count": len(combined),
    }


def list_directory(path_str: str) -> dict:
    """Return directory contents for the file browser."""
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return {"error": f"not a directory: {path_str}"}
    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.is_dir():
                entries.append({
                    "name":       child.name,
                    "type":       "dir",
                    "is_dataset": (child / "meta" / "info.json").exists(),
                    "path":       str(child),
                })
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries}


def read_wizard_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return {"error": str(e)}


# ── HTTP Handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    cache_dir: Path = Path("/home/edgeai/lerobot/cache")

    def log_message(self, fmt, *args):
        if not args or not isinstance(args[0], str):
            return
        parts = args[0].split()
        if len(parts) < 2:
            return
        if any(parts[1].startswith(p) for p in ("/api/", "/robot.urdf", "/pkg/")):
            print(f"  {args[0]}")

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p  = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            # Landing page
            if p == "/":
                self._file(TOOLS_DIR / "landing.html")
            # Viewer
            elif p in ("/viewer", "/viewer/"):
                self._file(VIEWER_DIR / "index.html")
            elif p in ("/main.js", "/style.css"):
                self._file(VIEWER_DIR / p.lstrip("/"))
            elif p == "/robot.urdf":
                self._file(URDF_PATH)
            elif p.startswith("/lib/"):
                self._file(VIEWER_DIR / "lib" / p[5:])
            elif p.startswith("/pkg/trossen_arm_description/"):
                self._file(TROSSEN_DIR / p[len("/pkg/trossen_arm_description/"):])
            # Wizard
            elif p in ("/wizard", "/wizard/"):
                self._file(VIEWER_DIR / "wizard.html")
            elif p in ("/wizard.js", "/wizard.css"):
                self._file(VIEWER_DIR / p.lstrip("/"))
            # Viewer API
            elif p == "/api/datasets":
                path_arg = qs.get("path", [None])[0]
                search_dir = Path(path_arg).expanduser().resolve() if path_arg else self.cache_dir
                self._json(find_datasets(search_dir))
            elif p == "/api/episodes":
                self._json(get_episodes(qs.get("dataset", [""])[0]))
            elif p == "/api/frames":
                ds = qs.get("dataset", [""])[0]
                ep = int(qs.get("episode", ["0"])[0])
                self._json(get_frames(ds, ep))
            # Wizard API
            elif p == "/api/files":
                self._json(list_directory(qs.get("path", [str(Path.home())])[0]))
            elif p == "/api/wizard/datasets":
                path_arg = qs.get("path", [str(self.cache_dir)])[0]
                self._json(find_direct_datasets(Path(path_arg).expanduser().resolve()))
            elif p == "/api/wizard/config":
                self._json(read_wizard_config())
            elif p == "/api/wizard/log":
                self._stream_log(qs.get("job", [""])[0])
            else:
                self.send_error(404)
        except Exception:
            traceback.print_exc()
            self.send_error(500)

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self):
        p      = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        try:
            if p == "/api/wizard/config":
                CONFIG_PATH.write_text(
                    yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )
                self._json({"ok": True})

            elif p == "/api/wizard/run":
                cmd = [sys.executable, str(WIZARD_SCRIPT), "--config", str(CONFIG_PATH)]
                if data.get("start_from"):        cmd += ["--start-from", data["start_from"]]
                if data.get("stop_at"):           cmd += ["--stop-at",    data["stop_at"]]
                if data.get("ee_frame"):          cmd += ["--ee-frame",   data["ee_frame"]]
                if data.get("ee_rot_repr"):    cmd += ["--ee-rot-repr", data["ee_rot_repr"]]
                if data.get("ee_joint_repr") is False:
                    cmd += ["--no-ee-joint-repr"]
                if data.get("skip_ee"):
                    cmd += ["--skip-ee"]

                job_id = uuid.uuid4().hex[:8]
                with _jobs_lock:
                    _jobs[job_id] = {"queue": queue.Queue(), "done": False,
                                     "returncode": None, "process": None}
                threading.Thread(
                    target=_run_job, args=(job_id, cmd, str(TOOLS_DIR)), daemon=True
                ).start()
                self._json({"job_id": job_id})
            else:
                self.send_error(404)
        except Exception:
            traceback.print_exc()
            self.send_error(500)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _stream_log(self, job_id: str):
        if job_id not in _jobs:
            self.send_error(404, "job not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = _jobs[job_id]["queue"]
        try:
            while True:
                line = q.get()
                if line is None:
                    rc = _jobs[job_id].get("returncode", 0)
                    self.wfile.write(
                        f"data: {json.dumps({'done': True, 'returncode': rc})}\n\n".encode()
                    )
                    self.wfile.flush()
                    break
                self.wfile.write(f"data: {json.dumps({'line': line})}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _file(self, path: Path):
        path = Path(path)
        if not path.exists():
            self.send_error(404, f"Not found: {path}")
            return
        data  = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type",   ctype)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type",                "application/json")
        self.send_header("Content-Length",              len(data))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dataset Tools Server")
    ap.add_argument("--port",  type=int, default=8080)
    ap.add_argument("--cache", type=str,
                    default=str(Path("/home/edgeai/lerobot/cache")),
                    help="Path to LeRobot dataset cache")
    args = ap.parse_args()

    cache = Path(args.cache).expanduser().resolve()
    Handler.cache_dir = cache

    print(f"Landing →  http://localhost:{args.port}")
    print(f"Viewer  →  http://localhost:{args.port}/viewer")
    print(f"Wizard  →  http://localhost:{args.port}/wizard")
    print(f"Cache   →  {cache}")
    if not cache.exists():
        print(f"  WARNING: cache directory does not exist: {cache}")
    else:
        datasets = find_datasets(cache)
        print(f"  Found {len(datasets)} dataset(s):")
        for d in datasets:
            ee  = [k for k in ("ee_left", "ee_right") if d[f"has_{k}"]]
            tag = f"  [{', '.join(ee)}]" if ee else "  [no EE]"
            print(f"    {d['name']}  ({d['total_episodes']} ep){tag}")

    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
