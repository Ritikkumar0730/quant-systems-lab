"""ITCH 5.0: the pure-Python codec, and the C++ replay tool cross-checked
against it on a generated feed (field checksums, message counts, integrity
and printed volume)."""
from __future__ import annotations

import io
import random
import re
import subprocess
from pathlib import Path

import pytest

from qsl.microstructure import itch

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "build" / "bin" / "itch_replay"

# Payload lengths from the ITCH 5.0 specification (type byte included).
SPEC_LENGTHS = {"S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28, "J": 35, "h": 21,
                "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19, "U": 35, "P": 44, "Q": 40, "B": 19,
                "I": 50, "N": 20, "O": 48}


def test_layout_lengths_match_the_spec():
    assert itch.MESSAGE_LENGTH == SPEC_LENGTHS


def _random_value(code: str, rng: random.Random):
    if code == "c":
        return rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    if code.endswith("s"):
        return "".join(rng.choice("ABCXYZ ") for _ in range(int(code[:-1])))
    bits = {"H": 16, "I": 32, "Q": 64}[code]
    return rng.getrandbits(bits)


@pytest.mark.parametrize("mtype", sorted(SPEC_LENGTHS))
def test_encode_decode_round_trip(mtype):
    rng = random.Random(ord(mtype))
    fmt, names = itch.LAYOUTS[mtype]
    for _ in range(200):
        msg = {"type": mtype, "locate": rng.getrandbits(16), "tracking": rng.getrandbits(16),
               "timestamp": rng.getrandbits(48)}
        for name, code in zip(names, itch._field_codes(fmt)):
            msg[name] = _random_value(code, rng)
        framed = itch.encode(msg)
        assert int.from_bytes(framed[:2], "big") == SPEC_LENGTHS[mtype] == len(framed) - 2
        assert itch.decode(framed[2:]) == msg


def test_decodes_hand_assembled_add_order():
    # Add Order laid out byte by byte from the spec's offset table.
    payload = (b"A" + (7).to_bytes(2, "big") + (3).to_bytes(2, "big") + (34_200_000_000_123).to_bytes(6, "big")
               + (0x0102030405060708).to_bytes(8, "big") + b"S" + (250).to_bytes(4, "big") + b"MSFT    "
               + (1_575_900).to_bytes(4, "big"))
    assert len(payload) == 36
    m = itch.decode(payload)
    assert (m["locate"], m["tracking"], m["timestamp"]) == (7, 3, 34_200_000_000_123)
    assert (m["ref"], m["side"], m["shares"], m["stock"], m["price"]) == (0x0102030405060708, "S", 250, "MSFT    ",
                                                                          1_575_900)


def test_rejects_wrong_lengths_and_unknown_types():
    with pytest.raises(ValueError):
        itch.decode(b"z" + bytes(10))
    with pytest.raises(ValueError):
        itch.decode(b"D" + bytes(17))


def _generated_feed(seed: int, n: int):
    """Random but valid order flow: every reference names a live order and no
    order is overfilled. Returns (bytes, expected printed shares)."""
    rng = random.Random(seed)
    out = io.BytesIO()
    ts = 30_000 * 10**9
    locates = [5, 77, 4000]

    def w(**m):
        out.write(itch.encode(m))

    def hdr(t, loc):
        return {"type": t, "locate": loc, "tracking": 0, "timestamp": ts}

    w(**hdr("S", 0), event_code="O")
    for loc in locates:
        w(**hdr("R", loc), stock=f"T{loc}".ljust(8), market_category="Q", financial_status="N",
          round_lot_size=100, round_lots_only="N", issue_classification="C", issue_subtype="Z ",
          authenticity="P", short_sale_threshold="N", ipo_flag=" ", luld_reference_price_tier="1",
          etp_flag="N", etp_leverage_factor=0, inverse_indicator="N")
        w(**hdr("H", loc), stock=f"T{loc}".ljust(8), trading_state="T", reserved=" ", reason="    ")
    live: dict[int, list] = {}  # ref -> [locate, shares, price]
    next_ref, match, printed = 1, 1, 0
    for _ in range(n):
        ts += rng.randrange(1, 10**6)
        u = rng.random()
        if not live or u < 0.45:
            loc = rng.choice(locates)
            shares, price = rng.randrange(1, 5000), rng.randrange(900_000, 1_100_000, 100)
            t = "F" if rng.random() < 0.2 else "A"
            extra = {"attribution": "MPID"} if t == "F" else {}
            w(**hdr(t, loc), ref=next_ref, side=rng.choice("BS"), shares=shares, stock=f"T{loc}".ljust(8),
              price=price, **extra)
            live[next_ref] = [loc, shares, price]
            next_ref += rng.choice((1, 4, 4, 8))
            continue
        ref = rng.choice(list(live))
        loc, shares, price = live[ref]
        if u < 0.60:
            k = rng.randrange(1, shares + 1)
            w(**hdr("E", loc), ref=ref, executed_shares=k, match_number=match)
            printed += k
            live[ref][1] -= k
        elif u < 0.65:
            k = rng.randrange(1, shares + 1)
            printable = rng.choice("YN")
            w(**hdr("C", loc), ref=ref, executed_shares=k, match_number=match, printable=printable,
              execution_price=price + 100)
            printed += k if printable == "Y" else 0
            live[ref][1] -= k
        elif u < 0.75:
            k = rng.randrange(1, shares + 1)
            w(**hdr("X", loc), ref=ref, cancelled_shares=k)
            live[ref][1] -= k
        elif u < 0.88:
            w(**hdr("D", loc), ref=ref)
            live[ref][1] = 0
        else:
            new_shares, new_price = rng.randrange(1, 5000), price + rng.choice((-100, 0, 100))
            w(**hdr("U", loc), original_ref=ref, new_ref=next_ref, shares=new_shares, price=new_price)
            live[next_ref] = [loc, new_shares, new_price]
            next_ref += 4
            live[ref][1] = 0
        match += 1
        if live[ref][1] == 0:
            del live[ref]
        if rng.random() < 0.03:
            k = rng.randrange(1, 1000)
            w(**hdr("P", loc), ref=0, side="B", shares=k, stock=f"T{loc}".ljust(8), price=1_000_000,
              match_number=match)
            printed += k
        if rng.random() < 0.002:
            k = rng.randrange(1, 10**6)
            w(**hdr("Q", loc), shares=k, stock=f"T{loc}".ljust(8), cross_price=1_000_000, match_number=match,
              cross_type="C")
            printed += k
    w(**hdr("S", 0), event_code="C")
    return out.getvalue(), printed


def _run_replay(path: Path) -> str:
    if not REPLAY.exists():
        pytest.skip("build/bin/itch_replay not built (run make)")
    res = subprocess.run([str(REPLAY), str(path), "--checksum"], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


@pytest.mark.parametrize("seed", [1, 2])
def test_cpp_replay_agrees_with_python_decoder(tmp_path, seed):
    data, printed = _generated_feed(seed, 30_000)
    path = tmp_path / "feed.itch"
    path.write_bytes(data)
    out = _run_replay(path)

    py_sums, py_counts = itch.checksums(io.BytesIO(data))
    cpp_sums = {k: int(v) for k, v in re.findall(r"^checksum (\w+) (\d+)$", out, re.M)}
    assert cpp_sums == py_sums

    cpp_counts = {t: int(c) for t, c in re.findall(r"(?:^|\s)(\S) +(\d+)(?=\s|$)",
                                                  out.split("messages by type")[1].split("Session")[0])}
    assert cpp_counts == dict(py_counts)
    assert int(re.search(r"messages\s+(\d+)", out).group(1)) == sum(py_counts.values())

    integrity = re.search(r"unknown order (\d+)  duplicate (\d+)  overfill (\d+)  locate mismatch (\d+)  "
                          r"level mismatch (\d+)", out).groups()
    assert all(v == "0" for v in integrity)
    assert int(re.search(r"printed executions \d+ \((\d+) shares\)", out).group(1)) == printed
