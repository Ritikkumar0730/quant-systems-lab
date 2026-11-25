"""NASDAQ TotalView-ITCH 5.0 decoder and encoder in pure Python.

This is a second, independent implementation of the message layouts used by
the C++ parser (``cpp/include/qsl/itch``). The layouts here are written from
the specification's field lists as ``struct`` formats, not from the C++
offsets, so agreement between the two on real NASDAQ data is evidence that
both read every field correctly. It is far slower than the C++ parser and is
meant for cross-checks, tests and small files.

Every message starts with: type (1 byte), stock locate (2), tracking number
(2), timestamp (6 bytes, nanoseconds since midnight). Integers are
big-endian; prices are unsigned 32-bit with 4 implied decimals. In the binary
file format each message is preceded by a 2-byte big-endian length.
"""
from __future__ import annotations

import struct
from collections import Counter
from typing import BinaryIO, Iterator

_HEADER = "HH6s"  # locate, tracking, timestamp (after the type byte)

# type -> (struct format of the body after the common header, field names)
LAYOUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "S": ("c", ("event_code",)),
    "R": ("8sccIcc2scccccIc", ("stock", "market_category", "financial_status", "round_lot_size",
                              "round_lots_only", "issue_classification", "issue_subtype", "authenticity",
                              "short_sale_threshold", "ipo_flag", "luld_reference_price_tier", "etp_flag",
                              "etp_leverage_factor", "inverse_indicator")),
    "H": ("8scc4s", ("stock", "trading_state", "reserved", "reason")),
    "Y": ("8sc", ("stock", "reg_sho_action")),
    "L": ("4s8sccc", ("mpid", "stock", "primary_market_maker", "market_maker_mode", "participant_state")),
    "V": ("QQQ", ("level1", "level2", "level3")),
    "W": ("c", ("breached_level",)),
    "K": ("8sIcI", ("stock", "release_time", "release_qualifier", "ipo_price")),
    "J": ("8sIIII", ("stock", "reference_price", "upper_price", "lower_price", "extension")),
    "h": ("8scc", ("stock", "market_code", "halt_action")),
    "A": ("QcI8sI", ("ref", "side", "shares", "stock", "price")),
    "F": ("QcI8sI4s", ("ref", "side", "shares", "stock", "price", "attribution")),
    "E": ("QIQ", ("ref", "executed_shares", "match_number")),
    "C": ("QIQcI", ("ref", "executed_shares", "match_number", "printable", "execution_price")),
    "X": ("QI", ("ref", "cancelled_shares")),
    "D": ("Q", ("ref",)),
    "U": ("QQII", ("original_ref", "new_ref", "shares", "price")),
    "P": ("QcI8sIQ", ("ref", "side", "shares", "stock", "price", "match_number")),
    "Q": ("Q8sIQc", ("shares", "stock", "cross_price", "match_number", "cross_type")),
    "B": ("Q", ("match_number",)),
    "I": ("QQc8sIIIcc", ("paired_shares", "imbalance_shares", "imbalance_direction", "stock", "far_price",
                         "near_price", "current_reference_price", "cross_type", "price_variation_indicator")),
    "N": ("8sc", ("stock", "interest_flag")),
    "O": ("8scIIIQII", ("stock", "open_eligibility", "min_price", "max_price", "near_execution_price",
                        "near_execution_time", "lower_price_range_collar", "upper_price_range_collar")),
}

_STRUCTS = {t: struct.Struct(">" + _HEADER + fmt) for t, (fmt, _) in LAYOUTS.items()}

#: Payload length of each message type, including the type byte.
MESSAGE_LENGTH = {t: 1 + s.size for t, s in _STRUCTS.items()}


def _value(v):
    if isinstance(v, bytes):
        return v.decode("ascii")
    return v


def decode(payload: bytes) -> dict:
    """Decodes one message payload (without its 2-byte frame length)."""
    t = chr(payload[0])
    if t not in _STRUCTS:
        raise ValueError(f"unknown ITCH message type {t!r}")
    if len(payload) != MESSAGE_LENGTH[t]:
        raise ValueError(f"type {t!r}: length {len(payload)}, expected {MESSAGE_LENGTH[t]}")
    locate, tracking, ts, *body = _STRUCTS[t].unpack_from(payload, 1)
    msg = {"type": t, "locate": locate, "tracking": tracking, "timestamp": int.from_bytes(ts, "big")}
    for name, v in zip(LAYOUTS[t][1], body):
        msg[name] = _value(v)
    return msg


def encode(msg: dict) -> bytes:
    """Encodes a message dict (as returned by ``decode``) into a framed message."""
    t = msg["type"]
    fmt, names = LAYOUTS[t]
    fields = []
    for name, code in zip(names, _field_codes(fmt)):
        v = msg[name]
        if code == "c" or code.endswith("s"):  # character fields: exact width, ASCII
            width = 1 if code == "c" else int(code[:-1])
            v = v.encode("ascii") if isinstance(v, str) else v
            if len(v) != width:
                raise ValueError(f"field {name!r} must be {width} bytes")
        fields.append(v)
    payload = t.encode("ascii") + _STRUCTS[t].pack(msg["locate"], msg["tracking"],
                                                   int(msg["timestamp"]).to_bytes(6, "big"), *fields)
    return len(payload).to_bytes(2, "big") + payload


def _field_codes(fmt: str) -> list[str]:
    """Splits a struct format such as '8scI' into per-field codes ['8s', 'c', 'I']."""
    out, digits = [], ""
    for ch in fmt:
        if ch.isdigit():
            digits += ch
        else:
            out.append(digits + ch)
            digits = ""
    return out


def iter_messages(stream: BinaryIO, limit: int | None = None) -> Iterator[dict]:
    """Yields decoded messages from a length-prefixed ITCH file."""
    n = 0
    while limit is None or n < limit:
        head = stream.read(2)
        if len(head) < 2:
            return
        length = int.from_bytes(head, "big")
        payload = stream.read(length)
        if len(payload) < length:
            return
        n += 1
        yield decode(payload)


CHECKSUM_KEYS = ("ts", "locate", "add_ref", "add_shares", "add_price", "add_buys", "exec_shares", "exec_match",
                 "execpx_shares", "execpx_price", "execpx_printable", "cancel_shares", "delete_ref",
                 "replace_new_ref", "replace_shares", "replace_price", "trade_shares", "trade_price",
                 "cross_shares", "cross_price", "directory_round_lot")


def checksums(stream: BinaryIO, limit: int | None = None) -> tuple[dict[str, int], Counter]:
    """Field checksums (sums modulo 2**64) over the first `limit` frames.

    Same definitions as ``itch_replay --checksum``: timestamps and locates of
    every message type the C++ replay forwards, plus selected fields of
    order, trade and directory messages. Returns (checksums, counts by type).
    """
    mask = (1 << 64) - 1
    s = dict.fromkeys(CHECKSUM_KEYS, 0)
    counts: Counter = Counter()
    hdr = struct.Struct(">cHH6s")
    tracked = set("SRAFECXDUPQB")
    read = stream.read
    frames = 0
    while limit is None or frames < limit:
        head = read(2)
        if len(head) < 2:
            break
        length = int.from_bytes(head, "big")
        p = read(length)
        if len(p) < length:
            break
        frames += 1
        t = chr(p[0])
        if t not in MESSAGE_LENGTH or length != MESSAGE_LENGTH[t]:
            counts["invalid"] += 1
            continue
        counts[t] += 1
        if t not in tracked:
            continue
        _, locate, _, ts = hdr.unpack_from(p, 0)
        s["ts"] += int.from_bytes(ts, "big")
        s["locate"] += locate
        body = _STRUCTS[t].unpack_from(p, 1)[3:]
        if t in "AF":
            s["add_ref"] += body[0]; s["add_shares"] += body[2]; s["add_price"] += body[4]
            s["add_buys"] += body[1] == b"B"
        elif t == "E":
            s["exec_shares"] += body[1]; s["exec_match"] += body[2]
        elif t == "C":
            s["execpx_shares"] += body[1]; s["execpx_price"] += body[4]; s["execpx_printable"] += body[3] == b"Y"
        elif t == "X":
            s["cancel_shares"] += body[1]
        elif t == "D":
            s["delete_ref"] += body[0]
        elif t == "U":
            s["replace_new_ref"] += body[1]; s["replace_shares"] += body[2]; s["replace_price"] += body[3]
        elif t == "P":
            s["trade_shares"] += body[2]; s["trade_price"] += body[4]
        elif t == "Q":
            s["cross_shares"] += body[0]; s["cross_price"] += body[2]
        elif t == "R":
            s["directory_round_lot"] += body[3]
    return {k: v & mask for k, v in s.items()}, counts
