#!/usr/bin/env python3
"""Cross-checks the C++ ITCH parser against the independent pure-Python
decoder on real NASDAQ data.

Both decode the first N frames of the file. The C++ replay prints field
checksums (wrapping sums of timestamps, locates, references, sizes, prices
and match numbers per message type) and message counts; the Python decoder,
whose layouts are written separately from the spec as struct formats,
computes the same quantities. Any field read from a wrong offset changes a
checksum.

    python3 scripts/itch_crosscheck.py [--frames N] [--file data/12302019.NASDAQ_ITCH50.gz]

Writes docs/results/itch_crosscheck.txt.
"""
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from qsl.microstructure import itch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(ROOT / "data" / "12302019.NASDAQ_ITCH50.gz"))
    ap.add_argument("--frames", type=int, default=20_000_000)
    args = ap.parse_args()
    path = Path(args.file)
    replay = ROOT / "build" / "bin" / "itch_replay"
    if not path.exists() or not replay.exists():
        print("need the ITCH file and build/bin/itch_replay (run make)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    gz = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
    cpp = subprocess.run([str(replay), "-", "--checksum", "--limit", str(args.frames)], stdin=gz.stdout,
                         capture_output=True, text=True, check=False)
    gz.stdout.close()
    gz.wait()
    t_cpp = time.perf_counter() - t0
    if cpp.returncode != 0:
        print(cpp.stdout, cpp.stderr, file=sys.stderr)
        return 1
    cpp_sums = {k: int(v) for k, v in re.findall(r"^checksum (\w+) (\d+)$", cpp.stdout, re.M)}
    block = cpp.stdout.split("messages by type")[1].split("Session")[0]
    cpp_counts = {t: int(c) for t, c in re.findall(r"(?:^|\s)(\S) +(\d+)(?=\s|$)", block)}

    t0 = time.perf_counter()
    with gzip.open(path, "rb") as f:
        py_sums, py_counts = itch.checksums(f, limit=args.frames)
    t_py = time.perf_counter() - t0

    lines = [f"C++ parser vs independent Python decoder, first {args.frames:,} frames of {path.name}", "",
             f"{'field checksum':22s} {'C++':>22s} {'Python':>22s}"]
    ok = True
    for k in itch.CHECKSUM_KEYS:
        same = cpp_sums.get(k) == py_sums[k]
        ok &= same
        lines.append(f"{k:22s} {cpp_sums.get(k, -1):22d} {py_sums[k]:22d}  {'ok' if same else 'MISMATCH'}")
    lines += ["", f"{'type':6s} {'C++':>12s} {'Python':>12s}"]
    for t in sorted(set(cpp_counts) | set(py_counts)):
        same = cpp_counts.get(t, 0) == py_counts.get(t, 0)
        ok &= same
        lines.append(f"{t:6s} {cpp_counts.get(t, 0):12d} {py_counts.get(t, 0):12d}  {'ok' if same else 'MISMATCH'}")
    lines += ["", f"result: {'all checksums and counts agree' if ok else 'DISAGREEMENT'}",
              f"time: C++ {t_cpp:.1f} s (including gunzip), Python {t_py:.1f} s"]
    (ROOT / "docs" / "results" / "itch_crosscheck.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
