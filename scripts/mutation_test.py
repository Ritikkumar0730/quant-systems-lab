#!/usr/bin/env python3
"""Mutation test: inject realistic bugs into copies of the matching engine
and the ITCH feed handler, and confirm the C++ test suite catches every one.

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

# Which test files exercise each header, and the test-name filter to run.
SUITES = {
    "qsl/lob/order_book.hpp": (["test_order_book.cpp"], "lob_"),
    "qsl/itch/parser.hpp": (["test_itch.cpp"], "itch_"),
    "qsl/itch/messages.hpp": (["test_itch.cpp"], "itch_"),
    "qsl/itch/book.hpp": (["test_itch.cpp"], "itch_"),
    "qsl/itch/order_map.hpp": (["test_itch.cpp"], "itch_"),
}

# (description, header, original text, mutated text)
MUTANTS = [
    # ------------------------------------------------------- matching engine
    ("LIFO queue instead of FIFO (breaks time priority)", "qsl/lob/order_book.hpp",
     "pool_[slot] = Order{id, qty, static_cast<std::uint32_t>(tick), level.tail, kNil, side};\n"
     "        if (level.tail != kNil) pool_[level.tail].next = slot;\n"
     "        else level.head = slot;\n"
     "        level.tail = slot;",
     "pool_[slot] = Order{id, qty, static_cast<std::uint32_t>(tick), kNil, level.head, side};\n"
     "        if (level.head != kNil) pool_[level.head].prev = slot;\n"
     "        else level.tail = slot;\n"
     "        level.head = slot;"),
    ("trade printed one tick away from the maker price", "qsl/lob/order_book.hpp",
     "const Price price = to_price(tick);", "const Price price = to_price(tick) + 1;"),
    ("post-only cross check off by one", "qsl/lob/order_book.hpp",
     "return best_ask_ != kNone && tick >= best_ask_;", "return best_ask_ != kNone && tick > best_ask_;"),
    ("reduce forgets to update level quantity", "qsl/lob/order_book.hpp",
     "        levels_[order.tick].qty -= order.qty - new_qty;\n        order.qty = new_qty;\n        return Reject::None;",
     "        order.qty = new_qty;\n        return Reject::None;"),
    ("fill-or-kill liquidity check off by one", "qsl/lob/order_book.hpp",
     "if (side == Side::Buy ? tick > limit : tick < limit) break;",
     "if (side == Side::Buy ? tick >= limit : tick < limit) break;"),
    ("best bid not refreshed when its level empties", "qsl/lob/order_book.hpp",
     "if (tick == best_bid_) best_bid_ = tick == 0 ? kNone : bids_.find_prev(tick - 1);", "(void)tick;"),
    # ---------------------------------------------------- ITCH feed handler
    ("ITCH: Add Order price read one byte early", "qsl/itch/parser.hpp",
     "symbol_at(p + 24), be32(p + 32),\n                                  {' ', ' ', ' ', ' '}});",
     "symbol_at(p + 24), be32(p + 31),\n                                  {' ', ' ', ' ', ' '}});"),
    ("ITCH: 48-bit timestamp truncated to 32 bits", "qsl/itch/messages.hpp",
     "return (std::uint64_t{be16(p)} << 32) | be32(p + 2);", "return be32(p + 2);"),
    ("ITCH: oversized frames accepted", "qsl/itch/parser.hpp",
     "if (length != expected) {", "if (length < expected) {"),
    ("ITCH: non-printable executions counted as volume", "qsl/itch/book.hpp",
     "m.printable == 'Y' ? Fill::WithPrice : Fill::NonPrintable", "Fill::WithPrice"),
    ("ITCH: replace puts the new order on the wrong side", "qsl/itch/book.hpp",
     "add_order(m.new_ref, m.header.locate, is_bid, m.price, m.shares);",
     "add_order(m.new_ref, m.header.locate, !is_bid, m.price, m.shares);"),
    ("ITCH: price-level search stops one level short", "qsl/itch/book.hpp",
     "levels_[j - 1].key >= k", "levels_[j - 1].key > k"),
    ("ITCH: order map backward shift skips displaced entries", "qsl/itch/order_map.hpp",
     "if (probe_len >= dist_to_hole) {", "if (probe_len > dist_to_hole) {"),
]


def main() -> int:
    survivors = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, header, old, new) in enumerate(MUTANTS):
            source = (ROOT / "cpp/include" / header).read_text()
            if source.count(old) != 1:
                print(f"ERROR   mutant '{name}' no longer applies exactly once; update scripts/mutation_test.py")
                return 2
            work = Path(tmp) / f"m{i}"
            shutil.copytree(ROOT / "cpp/include", work / "include")
            (work / "include" / header).write_text(source.replace(old, new))
            tests, prefix = SUITES[header]
            exe = work / "test"
            subprocess.run(["c++", "-std=c++20", "-O2", "-I", str(work / "include"),
                            str(ROOT / "cpp/tests/test_main.cpp"), *[str(ROOT / "cpp/tests" / t) for t in tests],
                            "-o", str(exe)], check=True)
            result = subprocess.run([str(exe), prefix], capture_output=True, text=True)
            failed = [line.split()[0] for line in result.stdout.splitlines() if "FAILED" in line]
            killed = result.returncode != 0
            survivors += not killed
            print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}  ({len(failed)} failing tests)")
    print(f"\n{len(MUTANTS) - survivors}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
