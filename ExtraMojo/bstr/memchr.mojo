import math
from bit import count_trailing_zeros
from memory import Span, pack_bits, UnsafePointer
from sys.info import simdwidthof

alias SIMD_U8_WIDTH: Int = simdwidthof[DType.uint8]()


@always_inline("nodebug")
fn memchr(haystack: Span[UInt8], chr: UInt8, start: Int = 0) -> Int:
    """
    Function to find the next occurrence of character.
    """
    if len(haystack[start:]) < SIMD_U8_WIDTH:
        for i in range(start, len(haystack)):
            if haystack[i] == chr:
                return i
        return -1

    # Do an unaligned initial read, it doesn't matter that this will overlap the next portion
    var ptr = haystack[start:].unsafe_ptr()
    var v = ptr.load[width=SIMD_U8_WIDTH]()
    var mask = v == chr

    var packed = pack_bits(mask)
    if packed:
        var index = int(count_trailing_zeros(packed))
        return index + start

    # Now get the alignment
    var offset = SIMD_U8_WIDTH - (ptr.__int__() & (SIMD_U8_WIDTH - 1))
    var aligned_ptr = ptr.offset(offset)

    # Find the last aligned end
    var haystack_len = len(haystack) - (start + offset)
    var aligned_end = math.align_down(
        haystack_len, SIMD_U8_WIDTH
    )  # relative to start + offset

    # Now do aligned reads all through
    for s in range(0, aligned_end, SIMD_U8_WIDTH):
        var v = aligned_ptr.load[width=SIMD_U8_WIDTH](s)
        var mask = v == chr
        var packed = pack_bits(mask)
        if packed:
            var index = int(count_trailing_zeros(packed))
            return s + index + offset + start

    # Finish and last bytes
    for i in range(aligned_end + start + offset, len(haystack)):
        if haystack[i] == chr:
            return i

    return -1


alias LOOP_SIZE = SIMD_U8_WIDTH * 4


@always_inline("nodebug")
fn memchr_wide(haystack: Span[UInt8], chr: UInt8, start: Int = 0) -> Int:
    """
    Function to find the next occurrence of character.
    """
    if len(haystack[start:]) < LOOP_SIZE:
        for i in range(start, len(haystack)):
            if haystack[i] == chr:
                return i
        return -1

    # Do an unaligned initial read, it doesn't matter that this will overlap the next portion
    var ptr = haystack[start:].unsafe_ptr()
    var v = ptr.load[width=SIMD_U8_WIDTH]()
    var mask = v == chr

    var packed = pack_bits(mask)
    if packed:
        var index = int(count_trailing_zeros(packed))
        return index + start

    # Now get the alignment
    var offset = SIMD_U8_WIDTH - (ptr.__int__() & (SIMD_U8_WIDTH - 1))
    var aligned_ptr = ptr.offset(offset)

    # Find the last aligned end
    var haystack_len = len(haystack) - (start + offset)
    var aligned_end = math.align_down(
        haystack_len, LOOP_SIZE
    )  # relative to start + offset

    # Now do aligned reads all through
    for s in range(0, aligned_end, LOOP_SIZE):
        var a = aligned_ptr.load[width=SIMD_U8_WIDTH](s)
        var b = aligned_ptr.load[width=SIMD_U8_WIDTH](s + 1 * SIMD_U8_WIDTH)
        var c = aligned_ptr.load[width=SIMD_U8_WIDTH](s + 2 * SIMD_U8_WIDTH)
        var d = aligned_ptr.load[width=SIMD_U8_WIDTH](s + 3 * SIMD_U8_WIDTH)
        var eqa = a == chr
        var eqb = b == chr
        var eqc = c == chr
        var eqd = d == chr
        var or1 = eqa | eqb
        var or2 = eqc | eqd
        var or3 = or1 | or2

        var packed = pack_bits(or3)
        if packed:
            # Now check each register knowing we have a match
            var packed_a = pack_bits(eqa)
            if packed_a:
                var index = int(count_trailing_zeros(packed_a))
                return s + index + offset + start
            var packed_b = pack_bits(eqb)
            if packed_b:
                var index = int(count_trailing_zeros(packed_b))
                return s + (1 * SIMD_U8_WIDTH) + index + offset + start
            var packed_c = pack_bits(eqc)
            if packed_c:
                var index = int(count_trailing_zeros(packed_c))
                return s + (2 * SIMD_U8_WIDTH) + index + offset + start

            var packed_d = pack_bits(eqd)
            var index = int(count_trailing_zeros(packed_d))
            return s + (3 * SIMD_U8_WIDTH) + index + offset + start

    # Now by single SIMD jumps
    var single_simd_end = math.align_down(
        haystack_len, SIMD_U8_WIDTH
    )  # relative to start + offset
    for s in range(aligned_end, single_simd_end, SIMD_U8_WIDTH):
        var v = aligned_ptr.load[width=SIMD_U8_WIDTH](s)
        var mask = v == chr

        var packed = pack_bits(mask)
        if packed:
            var index = int(count_trailing_zeros(packed))
            return s + index + offset + start

    # Finish and last bytes
    for i in range(single_simd_end + start + offset, len(haystack)):
        if haystack[i] == chr:
            return i

    return -1
