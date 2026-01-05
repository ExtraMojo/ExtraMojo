"""Hash functions.

These are all for `BBHash`, with the exception of fnv1a.
"""
from memory import bitcast

from math import trunc

comptime m: UInt64 = 0x880355F21E6D1965


@always_inline
fn _level_key_hash(level: UInt64, key: UInt64) -> UInt64:
    """`hash` returns the hash of the current level and key."""
    return _key_hash(_level_hash(level), key)


@always_inline
fn _level_hash(level: UInt64) -> UInt64:
    """`level_hash` returns the hash of the given level."""
    return _mix(level) * m


@always_inline
fn _key_hash(level_hash: UInt64, key: UInt64) -> UInt64:
    """`key_hash` returns the hash of a key given a level hash."""
    var h = level_hash
    h ^= _mix(key)
    h *= m
    h = _mix(h)
    return h


@always_inline
fn _mix(h_in: UInt64) -> UInt64:
    """`mix` is a compression function for fast hashing."""
    var h = h_in
    h ^= h >> 23
    h *= 0x2127599BF4325C37
    h ^= h >> 47
    return h


# todo: try u128
@always_inline
fn _hash64(seed: UInt64, buffer: Span[UInt8]) -> UInt64:
    var buf = buffer[:]
    var h = seed ^ (UInt64(len(buf)) * m)

    var n = len(buf) // 8
    if n > 0:
        # TODO verify the bitcast is a no-op, or use rebind
        var data = Span(ptr=buffer.unsafe_ptr().bitcast[UInt64](), length=n)
        for v in data:
            h ^= _mix(v)
            h *= m
        buf = buf[n * 8 : len(buf)]

    var v: UInt64 = 0
    for i in range(0, len(buf)):
        v |= UInt64(buf[i]) << (8 * i)
    if len(buf) > 0:
        h ^= _mix(v)
        h *= m
    return _mix(h)


comptime FNV_OFFSET: UInt64 = 14695981039346656037
comptime FNV_PRIME: UInt64 = 1099511628211


@always_inline
fn fnv1a(buf: Span[UInt8]) -> UInt64:
    """Classic FNV1a hash function."""
    var h = FNV_OFFSET
    for b in buf:
        h ^= UInt64(b)
        h *= FNV_PRIME
    return h
