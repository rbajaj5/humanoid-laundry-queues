"""Fetch a pinned BSD-3-Clause humanoid model, without redistributing private RAG data."""
from pathlib import Path
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REVISION = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/unitree_g1/"
MODEL = ROOT / "models" / "unitree_g1"


def fetch(name):
    path = MODEL / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "humanoid-delay-research"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def main():
    manifest_path = ROOT / "output/model_manifest.json"
    expected = json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else []
    for record in expected:
        path = MODEL / record["path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Cached model differs from retained manifest: {path}")
    names = ["LICENSE", "README.md", "g1.xml", "scene.xml", "g1_mjx.xml", "scene_mjx.xml"]
    for name in names:
        fetch(name)
    meshes = set()
    for name in ("g1.xml", "g1_mjx.xml"):
        for element in ET.parse(MODEL / name).findall(".//mesh[@file]"):
            meshes.add("assets/" + element.attrib["file"])
    for name in sorted(meshes):
        print("model asset:", name, flush=True)
        fetch(name)
    records = [{"path": name, "sha256": hashlib.sha256((MODEL/name).read_bytes()).hexdigest(),
                "bytes": (MODEL/name).stat().st_size} for name in sorted(set(names) | meshes)]
    if expected and records != expected:
        raise ValueError("Downloaded files differ from the retained model manifest")
    manifest = {"upstream": "https://github.com/google-deepmind/mujoco_menagerie", "revision": REVISION,
                "license": "BSD-3-Clause; models/unitree_g1/LICENSE", "files": records}
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Model ready:", MODEL, sum(r["bytes"] for r in records), "bytes")


if __name__ == "__main__":
    main()
