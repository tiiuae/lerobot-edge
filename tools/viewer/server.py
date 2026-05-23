#!/usr/bin/env python3
"""
Dataset Viewer Server — serves the 3D robot visualization app.

Usage (from the lerobot/ root or viewer/ directory):
    python viewer/server.py [--port 8080]
    Open http://localhost:8080
"""
import argparse
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TROSSEN_DIR = Path("/home/edgeai/trossen_arm_ros/trossen_arm_description")
URDF_PATH   = TROSSEN_DIR / "urdf/generated/mobile_ai.urdf"
VIEWER_DIR  = Path(__file__).parent

mimetypes.add_type("application/javascript",     ".js")
mimetypes.add_type("text/css",                   ".css")
mimetypes.add_type("model/stl",                  ".stl")
mimetypes.add_type("model/vnd.collada+xml",      ".dae")
mimetypes.add_type("image/png",                  ".png")


def find_datasets(cache_dir: Path):
    """Return ALL LeRobot datasets under cache_dir, EE or not."""
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
                "name":           name,
                "path":           str(root),
                "total_episodes": info.get("total_episodes", 0),
                "total_frames":   info.get("total_frames",   0),
                "has_ee_left":    "observation.ee_left"  in feats,
                "has_ee_right":   "observation.ee_right" in feats,
                "state_names":    feats.get("observation.state", {}).get("names", []),
                "action_names":   feats.get("action",            {}).get("names", []),
            })
        except Exception as e:
            print(f"  [skip] {info_path}: {e}")
            continue
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
        "observation.ee_left", "observation.ee_right",
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

    frames = [
        {col: combined[col][i].as_py() for col in combined.schema.names}
        for i in range(len(combined))
    ]
    return {"frames": frames, "count": len(frames)}


class Handler(BaseHTTPRequestHandler):
    cache_dir: Path = Path.home() / ".cache/huggingface/lerobot"

    def log_message(self, fmt, *args):
        # args[0] is the HTTP request line for normal requests but can be an int
        # (error code) when called from send_error → log_error, so guard carefully.
        if not args or not isinstance(args[0], str):
            return
        parts = args[0].split()
        if len(parts) < 2:
            return
        path = parts[1]
        if any(path.startswith(p) for p in ('/api/', '/robot.urdf', '/pkg/')):
            print(f"  {args[0]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p  = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if p == "/":
                self._file(VIEWER_DIR / "index.html")
            elif p in ("/main.js", "/style.css"):
                self._file(VIEWER_DIR / p.lstrip("/"))
            elif p == "/robot.urdf":
                self._file(URDF_PATH)
            elif p.startswith("/lib/"):
                self._file(VIEWER_DIR / "lib" / p[5:])
            elif p.startswith("/pkg/trossen_arm_description/"):
                rel = p[len("/pkg/trossen_arm_description/"):]
                self._file(TROSSEN_DIR / rel)
            elif p == "/api/datasets":
                self._json(find_datasets(self.cache_dir))
            elif p == "/api/episodes":
                self._json(get_episodes(qs.get("dataset", [""])[0]))
            elif p == "/api/frames":
                ds = qs.get("dataset", [""])[0]
                ep = int(qs.get("episode", ["0"])[0])
                self._json(get_frames(ds, ep))
            else:
                self.send_error(404)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.send_error(500, str(exc))

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
        self.send_header("Content-Type",              "application/json")
        self.send_header("Content-Length",            len(data))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(description="Robot Dataset Viewer Server")
    ap.add_argument("--port",  type=int,  default=8080)
    ap.add_argument("--cache", type=str,  default=str(Path.home() / ".cache/huggingface/lerobot"),
                    help="Path to LeRobot dataset cache (absolute or relative to CWD)")
    args = ap.parse_args()

    # expanduser + resolve turns any relative or ~ path into an absolute one
    cache = Path(args.cache).expanduser().resolve()
    Handler.cache_dir = cache

    print(f"Viewer  →  http://localhost:{args.port}")
    print(f"Cache   →  {cache}")
    if not cache.exists():
        print(f"  WARNING: cache directory does not exist: {cache}")
    else:
        datasets = find_datasets(cache)
        print(f"  Found {len(datasets)} dataset(s):")
        for d in datasets:
            ee = []
            if d["has_ee_left"]:
                ee.append("ee_left")
            if d["has_ee_right"]:
                ee.append("ee_right")
            ee_tag = f"  [{', '.join(ee)}]" if ee else "  [no EE — joint panel only]"
            print(f"    {d['name']}  ({d['total_episodes']} ep){ee_tag}")
        if not datasets:
            print("  No datasets found. Check that the path contains meta/info.json files.")

    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
