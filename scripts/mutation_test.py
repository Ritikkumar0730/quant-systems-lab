#!/usr/bin/env python3
"""Mutation test for the order book: inject realistic bugs into a copy of
order_book.hpp and confirm the C++ test suite catches every one.

A test suite that still passes with a bug injected is not testing that
behaviour. Run from the repository root:  python3 scripts/mutation_test.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "cpp/include/qsl/lob/order_book.hpp"

MUTANTS = {
    "LIFO queue instead of FIFO (breaks time priority)": (
        "pool_[slot] = Order{id, qty, static_cast<std::uint32_t>(tick), level.tail, kNil, side};\n"
        "        if (level.tail != kNil) pool_[level.tail].next = slot;\n"
        "        else level.head = slot;\n"
        "        level.tail = slot;",
        "pool_[slot] = Order{id, qty, static_cast<std::uint32_t>(tick), kNil, level.head, side};\n"
        "        if (level.head != kNil) pool_[level.head].prev = slot;\n"
        "        else level.tail = slot;\n"
        "        level.head = slot;"),
    "trade printed one tick away from the maker price": (
        "const Price price = to_price(tick);", "const Price price = to_price(tick) + 1;"),
    "post-only cross check off by one": (
        "return best_ask_ != kNone && tick >= best_ask_;", "return best_ask_ != kNone && tick > best_ask_;"),
    "reduce forgets to update level quantity": (
        "        levels_[order.tick].qty -= order.qty - new_qty;\n        order.qty = new_qty;\n        return Reject::None;",
        "        order.qty = new_qty;\n        return Reject::None;"),
    "fill-or-kill liquidity check off by one": (
        "if (side == Side::Buy ? tick > limit : tick < limit) break;",
        "if (side == Side::Buy ? tick >= limit : tick < limit) break;"),
    "best bid not refreshed when its level empties": (
        "if (tick == best_bid_) best_bid_ = tick == 0 ? kNone : bids_.find_prev(tick - 1);", "(void)tick;"),
}


def main() -> int:
    source = HEADER.read_text()
    survivors = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, (old, new)) in enumerate(MUTANTS.items()):
            if old not in source:
                print(f"ERROR   mutant '{name}' no longer applies; update scripts/mutation_test.py")
                return 2
            work = Path(tmp) / f"m{i}"
            shutil.copytree(ROOT / "cpp/include", work / "include")
            (work / "include/qsl/lob/order_book.hpp").write_text(source.replace(old, new))
            exe = work / "test"
            subprocess.run(["c++", "-std=c++20", "-O2", "-I", str(work / "include"), "-I", str(ROOT / "cpp/include"),
                            str(ROOT / "cpp/tests/test_main.cpp"), str(ROOT / "cpp/tests/test_order_book.cpp"),
                            "-o", str(exe)], check=True)
            result = subprocess.run([str(exe), "lob_"], capture_output=True, text=True)
            failed = [line.split()[0] for line in result.stdout.splitlines() if "FAILED" in line]
            killed = result.returncode != 0
            survivors += not killed
            print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}  ({len(failed)} failing tests)")
    print(f"\n{len(MUTANTS) - survivors}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
