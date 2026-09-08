#!/usr/bin/env python3
"""
Upload the AN-15 data-release package to Zenodo.

USAGE
-----
  # 1) test on the Zenodo sandbox first (recommended):
  ZENODO_TOKEN=<your_sandbox_token> python3 upload_to_zenodo.py --sandbox

  # 2) real deposit on production:
  ZENODO_TOKEN=<your_production_token> python3 upload_to_zenodo.py

The script creates the deposition, uploads every file in this directory
(preserving the data/ lexicon/ prompts/ code/ layout), attaches the
metadata from zenodo_metadata.json, and prints an EDIT link. It does
NOT auto-publish -- open the link on zenodo.org and click "Publish"
yourself (publishing is what mints the DOI and is hard to undo).

Add --publish to publish automatically once everything is uploaded.

Requires: pip install requests
"""
import os, sys, json, argparse, pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parent
API = {
    "production": "https://zenodo.org/api",
    "sandbox":    "https://sandbox.zenodo.org/api",
}

# top-level entries to skip (the upload script itself + the metadata file)
SKIP = {"upload_to_zenodo.py", "zenodo_metadata.json", ".gitignore"}


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel.name in SKIP or rel.parts[0] in (".git", "__pycache__"):
            continue
        yield rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", action="store_true", help="publish after upload")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN environment variable first.")
    base = API["sandbox"] if args.sandbox else API["production"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1) create deposition (no files yet)
    meta = json.loads((ROOT / "zenodo_metadata.json").read_text(encoding="utf-8"))
    r = requests.post(f"{base}/deposit/depositions",
                     json={"metadata": meta["metadata"]}, headers=headers)
    if r.status_code >= 400:
        sys.exit(f"create deposition failed {r.status_code}: {r.text}")
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"created deposition {dep_id}  ({'sandbox' if args.sandbox else 'production'})")

    # 2) upload files into the bucket, preserving relative paths
    for rel in iter_files():
        data = (ROOT / rel).read_bytes()
        # Zenodo bucket upload uses the filename; we keep the relative path as name
        r = requests.put(f"{bucket}/{rel.as_posix()}",
                         data=data, headers=headers)
        if r.status_code >= 400:
            sys.exit(f"upload failed for {rel}: {r.status_code} {r.text}")
        print(f"  uploaded {rel}  ({len(data)//1024} KB)")

    # 3) re-read deposition to confirm, then optionally publish
    r = requests.get(f"{base}/deposit/depositions/{dep_id}", headers=headers)
    dep = r.json()
    edit_link = dep["links"].get("html")
    print(f"\nDeposit ready. Review & publish here: {edit_link}")
    print(f"API id: {dep_id}")

    if args.publish:
        r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                          headers=headers)
        if r.status_code >= 400:
            sys.exit(f"publish failed {r.status_code}: {r.text}")
        print("PUBLISHED. DOI:", r.json().get("doi"))


if __name__ == "__main__":
    main()
