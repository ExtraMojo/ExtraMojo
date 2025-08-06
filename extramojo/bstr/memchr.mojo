"""Fast memchr implementations.

There are two here, `memchr` and `memchr_wide`. `memchr_wide` will do more comparisons at once,
but needs to do more loading first. If you know you have some distance between `needle`s, then
it should be faster. `memchr` is just vanilla memchr.
"""
import math
from bit import count_trailing_zeros
from memory import pack_bits, UnsafePointer
from sys.info import simdwidthof

alias SIMD_U8_WIDTH: Int = simdwidthof[DType.uint8]()


@always_inline("nodebug")
fn memchr[
    do_alignment: Bool = False
](haystack: Span[UInt8], chr: UInt8, start: Int = 0) -> Int:
    """
    Function to find the next occurrence of character.

    ```mojo
    from testing import assert_equal
    from extramojo.bstr.memchr import memchr

    assert_equal(memchr("enlivened,unleavened,Arnulfo's,Unilever's,unloved|Anouilh,analogue,analogy".as_bytes(), ord("|")), 49)
    ```

    Args:
        haystack: The bytes to search for the `chr`.
        chr: The byte to search for.
        start: The starting point to begin the search in `haystack`.

    Parameters:
        do_alignment: If True this will do an aligning read at the very start of the haystack.
                      If your haystack is very long, this may provide a marginal benefit. If the haystack is short,
                      or the needle is frequently in the first `SIMD_U8_WIDTH * 2` bytes, then skipping the
                      aligning read can be very beneficial since the aligning read check will overlap some
                      amount with the subsequent aligned read that happens next.

    Returns:
        The index of the found character, or -1 if not found.
    """
    if len(haystack[start:]) < SIMD_U8_WIDTH:
        for i in range(start, len(haystack)):
            if haystack[i] == chr:
                return i
        return -1

    # Do an unaligned initial read, it doesn't matter that this will overlap the next portion
    var ptr = haystack[start:].unsafe_ptr()

    var offset = 0

    @parameter
    if do_alignment:
        var v = ptr.load[width=SIMD_U8_WIDTH]()
        var mask = v == chr

        var packed = pack_bits(mask)
        if packed:
            var index = Int(count_trailing_zeros(packed))
            return index + start

        # Now get the alignment
        offset = SIMD_U8_WIDTH - (ptr.__int__() & (SIMD_U8_WIDTH - 1))
        # var aligned_ptr = ptr.offset(offset)
        ptr = ptr.offset(offset)

    # Find the last aligned end
    var haystack_len = len(haystack) - (start + offset)
    var aligned_end = math.align_down(
        haystack_len, SIMD_U8_WIDTH
    )  # relative to start + offset

    # Now do aligned reads all through
    for s in range(0, aligned_end, SIMD_U8_WIDTH):
        var v = ptr.load[width=SIMD_U8_WIDTH](s)
        var mask = v == chr
        var packed = pack_bits(mask)
        if packed:
            var index = Int(count_trailing_zeros(packed))
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

    This function does more unrolling and will be faster if the search if over longer distances. If in doubt use `memchr`.

    ```mojo
    from testing import assert_equal
    from extramojo.bstr.memchr import memchr_wide

    assert_equal(memchr_wide("enlivened,unleavened,Arnulfo's,Unilever's,unloved|Anouilh,analogue,analogy".as_bytes(), ord("|")), 49)
    ```

    Args:
        haystack: The bytes to search for the `chr`.
        chr: The byte to search for.
        start: The starting point to begin the search in `haystack`.

    Returns:
        The index of the found character, or -1 if not found.
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
        var index = Int(count_trailing_zeros(packed))
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
                var index = Int(count_trailing_zeros(packed_a))
                return s + index + offset + start
            var packed_b = pack_bits(eqb)
            if packed_b:
                var index = Int(count_trailing_zeros(packed_b))
                return s + (1 * SIMD_U8_WIDTH) + index + offset + start
            var packed_c = pack_bits(eqc)
            if packed_c:
                var index = Int(count_trailing_zeros(packed_c))
                return s + (2 * SIMD_U8_WIDTH) + index + offset + start

            var packed_d = pack_bits(eqd)
            var index = Int(count_trailing_zeros(packed_d))
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
            var index = Int(count_trailing_zeros(packed))
            return s + index + offset + start

    # Finish and last bytes
    for i in range(single_simd_end + start + offset, len(haystack)):
        if haystack[i] == chr:
            return i

    return -1
