#!/usr/bin/env python3
"""
torneko_lz_tool.py
LZ tool for PSX game 'World of Dragon Warrior - Torneko - The Last Hope (USA)'
At least some of game's graphics is compressed with LZSS scheme.
Flag byte, containing 8 bits:
	- 0: copy next byte in stream to output
	- 1: read next byte as Offset and then next byte as Len-3, copy Len bytes from previous 256 bytes buffer (previous pixels scanline)
Each 256 unpacked pixels, they are dumped to VRAM and copied as previously unpacked history buffer.


Compressor features:
 - uses positions_by_byte index for fast match-finding in 256-byte history
 - greedy match selection but with 1-token lookahead (lazy matching) to avoid common greedy mistakes
 - flag-group state persists across blocks (so repeated block-copies pack into the same flag byte)

"""
from typing import Tuple
import argparse
import os
import sys

FULL_IMAGE_SIZE = 0x100 * 0x100
BLOCK_SIZE = 256
MAX_HISTORY = 256

class PSXLZError(Exception):
    pass

def parse_offset(s: str) -> int:
    try:
        return int(s, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid offset: {s}")

# ---------------------------
# Decompression
# ---------------------------
def decompress_from_file(in_filename: str, offset: int, out_filename: str) -> None:
    if offset < 0:
        raise PSXLZError("Offset must be non-negative")
    if not os.path.isfile(in_filename):
        raise PSXLZError(f"Input file not found: {in_filename}")
    with open(in_filename, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if offset >= size:
            raise PSXLZError(f"Offset {offset} is beyond end of file (size {size})")
        f.seek(offset)
        data = f.read()
    out, consumed = decompress_bytes_to_exact(data, FULL_IMAGE_SIZE)
    with open(out_filename, "wb") as fo:
        fo.write(bytes(out))
    print(f"Decompressed 0x{consumed:x} bytes -> 0x{len(out):x} bytes in file '{out_filename}'")

def decompress_bytes_to_exact(data: bytes, target_len: int) -> Tuple[bytearray,int]:
    if target_len <= 0:
        return bytearray(), 0
    if not data:
        raise PSXLZError("No input data to decompress")

    p = 0
    src_len = len(data)
    header = data[p]; p += 1
    out = bytearray()
    history = bytearray([0] * MAX_HISTORY)
    plain_buf = bytearray()

    if header == 0x23:
        remaining = data[p:]
        if len(remaining) < target_len:
            raise PSXLZError(f"Raw mode: need {target_len} bytes but only {len(remaining)} available after header")
        out.extend(remaining[:target_len])
        consumed = 1 + target_len
        return out, consumed

    flag_byte = 0
    flag_mask = 0

    while len(out) < target_len:
        if flag_mask == 0:
            if p >= src_len:
                raise PSXLZError(f"Compressed stream ended prematurely after producing {len(out)} bytes (need {target_len})")
            flag_byte = data[p]; p += 1
            flag_mask = 0x80

        if (flag_byte & flag_mask) == 0:
            flag_mask >>= 1
            if p >= src_len:
                raise PSXLZError("Unexpected EOF while reading literal")
            b = data[p]; p += 1
            out.append(b)
            plain_buf.append(b)
            if len(plain_buf) == BLOCK_SIZE:
                history[:] = plain_buf
                plain_buf.clear()
        else:
            flag_mask >>= 1
            if p + 1 >= src_len:
                raise PSXLZError("Unexpected EOF while reading copy token")
            offset = data[p]; len_byte = data[p+1]; p += 2
            count = (len_byte + 3) & 0xFF
            if count == 0:
                count = 0x100
            idx = offset & 0xFF
            for _ in range(count):
                if len(out) >= target_len:
                    break
                b = history[idx]
                out.append(b)
                plain_buf.append(b)
                idx = (idx + 1) & 0xFF
                if len(plain_buf) == BLOCK_SIZE:
                    history[:] = plain_buf
                    plain_buf.clear()

    consumed = p
    return out, consumed

# ---------------------------
# Compressor (greedy + lazy + index)
# ---------------------------

def build_history_index(history: bytearray):
    """Return positions_by_byte: list of lists for each byte value -> history positions"""
    positions_by_byte = [[] for _ in range(256)]
    for h in range(MAX_HISTORY):
        b = history[h]
        positions_by_byte[b].append(h)
    return positions_by_byte

def find_best_match_at(history: bytearray, positions_by_byte, plain: bytes, pos: int, block_end: int):
    """
    Find best match (offset, length) in history for plain[pos:].
    Returns (best_off, best_len) with best_len>=0. If no match >=3, best_len will be 0.
    This checks only history offsets where first byte matches (fast).
    """
    first = plain[pos]
    cand_offsets = positions_by_byte[first]
    if not cand_offsets:
        return 0, 0
    remaining = block_end - pos
    max_allowed = min(256, remaining)
    best_len = 0
    best_off = 0
    # For each candidate offset extend match
    for h in cand_offsets:
        cur_len = 1
        # extend
        # use local variables for speed
        hist = history
        p = pos
        while cur_len < max_allowed and hist[(h + cur_len) & 0xFF] == plain[p + cur_len]:
            cur_len += 1
        if cur_len > best_len:
            best_len = cur_len
            best_off = h
            if best_len == max_allowed:
                break
    if best_len < 3:
        return 0, 0
    if best_len > 256:
        best_len = 256
    return best_off, best_len

def compress_bytes(plain: bytes) -> bytes:
    """
    Improved compressor:
     - pads input externally (caller ensures correct size)
     - processes block-by-block, history updated after full block flush
     - persistent flag-group across blocks
     - greedy selection of best match in history, but uses 1-token lookahead (lazy match):
         if best match at pos is len L0 >=3 and best at pos+1 is L1 > L0,
         prefer to emit literal at pos (so next token can use the longer match).
     - fast indexing via positions_by_byte for history to avoid scanning all 256 offsets every pos.
    """
    if len(plain) != FULL_IMAGE_SIZE:
        raise PSXLZError("compress_bytes expects plain exactly FULL_IMAGE_SIZE (pad before calling)")

    out = bytearray()
    out.append(0x00)  # compressed header

    history = bytearray([0] * MAX_HISTORY)
    pos = 0
    n = len(plain)

    # persistent flag-group state across blocks:
    flag_byte_pos = None
    flag_bits = 0
    flag_mask = 0  # 0 means need to start a new group
    token_buf = bytearray()

    while pos < n:
        block_start = pos
        block_end = min(block_start + BLOCK_SIZE, n)

        # build index for current history (fast)
        positions_by_byte = build_history_index(history)

        while pos < block_end:
            # start new flag group if needed
            if flag_mask == 0:
                flag_byte_pos = len(out)
                out.append(0)  # placeholder for flag byte
                flag_bits = 0
                flag_mask = 0x80
                token_buf = bytearray()

            # greedy: find best match at pos
            best_off, best_len = find_best_match_at(history, positions_by_byte, plain, pos, block_end)

            # lazy lookahead: check pos+1's best match, prefer literal now if it yields net benefit
            if best_len >= 3 and pos + 1 < block_end:
                next_off, next_len = find_best_match_at(history, positions_by_byte, plain, pos + 1, block_end)
                # heuristics: if next_len > best_len, prefer literal at pos to allow bigger match at pos+1
                # This fixes common greedy mistakes. Also if next_len == best_len and best_len is small, no change.
                if next_len > best_len:
                    # emit literal instead of copy
                    # note: we still allow copy if next_len is not sufficiently larger; this is conservative.
                    token_buf.append(plain[pos] & 0xFF)
                    pos += 1
                    flag_mask >>= 1
                else:
                    # use copy token
                    # encode
                    len_byte = (best_len - 3) & 0xFF
                    token_buf.append(best_off & 0xFF)
                    token_buf.append(len_byte)
                    flag_bits |= flag_mask
                    pos += best_len
                    flag_mask >>= 1
            elif best_len >= 3:
                # no lookahead possible (end of block) or best_len is <3 at pos+1
                len_byte = (best_len - 3) & 0xFF
                token_buf.append(best_off & 0xFF)
                token_buf.append(len_byte)
                flag_bits |= flag_mask
                pos += best_len
                flag_mask >>= 1
            else:
                # no decent match -> literal
                token_buf.append(plain[pos] & 0xFF)
                pos += 1
                flag_mask >>= 1

            # flush when group is full
            if flag_mask == 0:
                out[flag_byte_pos] = flag_bits
                flag_byte_pos = None
                out.extend(token_buf)
                token_buf = bytearray()
                # next token will start new flag group

        # finished block: update history with the block contents (blocks are full because caller pads)
        history[:] = plain[block_start:block_start + BLOCK_SIZE]

    # flush any remaining partial flag group
    if flag_mask != 0 and flag_byte_pos is not None:
        out[flag_byte_pos] = flag_bits
        out.extend(token_buf)
        flag_byte_pos = None

    return bytes(out)

def compress_file(plain_filename: str, out_filename: str) -> None:
    if not os.path.isfile(plain_filename):
        raise PSXLZError(f"Plain file not found: {plain_filename}")
    with open(plain_filename, "rb") as f:
        plain = bytearray(f.read())
    if len(plain) > FULL_IMAGE_SIZE:
        raise PSXLZError(f"Plain input too large: {len(plain)} bytes (max {FULL_IMAGE_SIZE})")
    if len(plain) < FULL_IMAGE_SIZE:
        plain.extend(b'\x00' * (FULL_IMAGE_SIZE - len(plain)))

    comp = compress_bytes(plain)
    with open(out_filename, "wb") as fo:
        fo.write(comp)
    print(f"Compressed 0x{len(plain):x} bytes -> 0x{len(comp):x} bytes in file '{out_filename}'")

# ---------------------------
# CLI 
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="PSX World of Dragon Warrior - Torneko - The Last Hope LZ tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("d", help="decompress: d <input_file> <offset> <output_file>")
    pd.add_argument("input_file")
    pd.add_argument("offset", type=parse_offset)
    pd.add_argument("output_file")

    pc = sub.add_parser("c", help="compress: c <plain_file> <output_file>")
    pc.add_argument("plain_file")
    pc.add_argument("output_file")

    args = parser.parse_args()
    try:
        if args.cmd == "d":
            decompress_from_file(args.input_file, args.offset, args.output_file)
        elif args.cmd == "c":
            compress_file(args.plain_file, args.output_file)
    except PSXLZError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
