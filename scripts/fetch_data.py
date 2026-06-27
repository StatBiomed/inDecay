#!/usr/bin/env python3
"""Reproducible, checksum-verified download of inDecay data and model weights.

Two archival sources (resolved by their permanent DOIs):

  * Training / somatic data  -> figshare article 25133564 (version 3)
    https://doi.org/10.6084/m9.figshare.25133564  -> extracted into  data/
  * Fig 4 & 5 model weights  -> Zenodo record 20977675
    https://doi.org/10.5281/zenodo.20977675       -> extracted into  pl_trainer_log/

Every file is verified against the publisher-supplied MD5 before use, and the
download is idempotent (re-running skips files already present with a good hash).

Usage:
    python scripts/fetch_data.py              # fetch everything
    python scripts/fetch_data.py --data       # data only
    python scripts/fetch_data.py --weights    # weights only
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGSHARE_ARTICLE = 25133564
FIGSHARE_VERSION = 3            # PINNED — never "latest"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/versions/{FIGSHARE_VERSION}"

ZENODO_RECORD = 20977675       # https://doi.org/10.5281/zenodo.20977675
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def md5(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, out, expected_md5=None):
    if expected_md5 and os.path.exists(out) and md5(out) == expected_md5:
        print(f"  [skip] {os.path.basename(out)} (md5 ok)")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"  [get ] {os.path.basename(out)}")
    urllib.request.urlretrieve(url, out)
    if expected_md5:
        got = md5(out)
        if got != expected_md5:
            sys.exit(f"  CHECKSUM FAIL {os.path.basename(out)}: {got} != {expected_md5}")
        print(f"  [ok  ] md5={got}")


def _safe_extract(tar, dest):
    dest_abs = os.path.abspath(dest)
    for m in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, m.name))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            sys.exit(f"  UNSAFE PATH in archive: {m.name}")
    tar.extractall(dest)


def fetch_data():
    print(f"== figshare data (article {FIGSHARE_ARTICLE} v{FIGSHARE_VERSION}) -> data/ ==")
    meta = json.load(urllib.request.urlopen(FIGSHARE_API))
    dest = os.path.join(REPO, "data")
    for f in meta["files"]:
        out = os.path.join(dest, f["name"])
        download(f["download_url"], out, f.get("supplied_md5"))
        if out.endswith((".tar.gz", ".tgz")):
            print(f"  [tar ] extracting {f['name']}")
            with tarfile.open(out) as t:
                _safe_extract(t, dest)


def fetch_weights():
    print(f"== Zenodo weights (record {ZENODO_RECORD}) -> pl_trainer_log/ ==")
    meta = json.load(urllib.request.urlopen(ZENODO_API))
    dest = os.path.join(REPO, ".cache_weights")
    os.makedirs(dest, exist_ok=True)
    for f in meta["files"]:
        name = f.get("key") or f.get("filename")
        link = f["links"].get("self") or f["links"].get("download")
        checksum = (f.get("checksum") or "").replace("md5:", "") or None
        out = os.path.join(dest, name)
        download(link, out, checksum)
        print(f"  [tar ] extracting {name} -> pl_trainer_log/")
        with tarfile.open(out) as t:           # archives carry the pl_trainer_log/ prefix
            _safe_extract(t, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true", help="fetch figshare data only")
    ap.add_argument("--weights", action="store_true", help="fetch Zenodo weights only")
    args = ap.parse_args()
    do_all = not (args.data or args.weights)
    if args.data or do_all:
        fetch_data()
    if args.weights or do_all:
        fetch_weights()
    print("done.")


if __name__ == "__main__":
    main()
