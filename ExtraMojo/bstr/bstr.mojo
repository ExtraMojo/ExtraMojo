import math
from algorithm import vectorize
from memory import UnsafePointer
from sys.info import simdwidthof

from ExtraMojo.bstr.memchr import memchr


# TODO: split this all out and create similar abstractions as the Rust bstr crate


alias SIMD_U8_WIDTH: Int = simdwidthof[DType.uint8]()


@always_inline
fn find_chr_all_occurrences(haystack: Span[UInt8], chr: UInt8) -> List[Int]:
    """Find all the occurrences of `chr` in the input buffer.

    ```mojo
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import find_chr_all_occurrences

    var haystack = "ATCGACCATCGAGATCATGTTTCAT"
    var expected = List(2, 5, 6, 9, 15, 22)
    assert_equal(find_chr_all_occurrences(haystack.as_bytes(), ord("C")), expected)
    ```
    """
    var holder = List[Int]()
    # TODO alignment
    # TODO move this to memchr?

    if len(haystack) < SIMD_U8_WIDTH:
        for i in range(0, len(haystack)):
            if haystack[i] == chr:
                holder.append(i)
        return holder

    @parameter
    fn inner[simd_width: Int](offset: Int):
        var simd_vec = haystack.unsafe_ptr().load[width=simd_width](offset)
        var bool_vec = simd_vec == chr
        if bool_vec.reduce_or():
            # TODO: @unroll
            for i in range(len(bool_vec)):
                if bool_vec[i]:
                    holder.append(offset + i)

    vectorize[inner, SIMD_U8_WIDTH](len(haystack))
    return holder


alias CAPITAL_A = SIMD[DType.uint8, SIMD_U8_WIDTH](ord("A"))
alias CAPITAL_Z = SIMD[DType.uint8, SIMD_U8_WIDTH](ord("Z"))
alias LOWER_A = SIMD[DType.uint8, SIMD_U8_WIDTH](ord("a"))
alias LOWER_Z = SIMD[DType.uint8, SIMD_U8_WIDTH](ord("z"))
alias ASCII_CASE_MASK = SIMD[DType.uint8, SIMD_U8_WIDTH](
    32
)  # The diff between a and A is just the sixth bit set
alias ZERO = SIMD[DType.uint8, SIMD_U8_WIDTH](0)


@always_inline
fn is_ascii_uppercase(value: UInt8) -> Bool:
    """Check if a byte is ASCII uppercase.

    ```mojo
    from testing import assert_true, assert_false
    from ExtraMojo.bstr.bstr import is_ascii_uppercase

    for ascii_letter in range(ord("A"), ord("Z")+1):
        assert_true(is_ascii_uppercase(ascii_letter))
    for ascii_letter in range(ord("a"), ord("z")+1):
        assert_false(is_ascii_uppercase(ascii_letter))
    assert_false(is_ascii_uppercase(0))
    ```
    """
    return value >= 65 and value <= 90  # 'A' -> 'Z'


@always_inline
fn is_ascii_lowercase(value: UInt8) -> Bool:
    """Check if a byte is ASCII lowercase.

    ```mojo
    from testing import assert_true, assert_false
    from ExtraMojo.bstr.bstr import is_ascii_lowercase

    for ascii_letter in range(ord("A"), ord("Z")+1):
        assert_false(is_ascii_lowercase(ascii_letter))
    for ascii_letter in range(ord("a"), ord("z")+1):
        assert_true(is_ascii_lowercase(ascii_letter))
    assert_false(is_ascii_lowercase(0))
    ```
    """
    return value >= 97 and value <= 122  # 'a' -> 'z'


@always_inline
fn to_ascii_lowercase(mut buffer: List[UInt8, _]):
    """Lowercase all ascii a-zA-Z characters.

    ```mojo
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import to_ascii_lowercase
    var test = List("ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ".as_bytes())
    var expected = List("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz".as_bytes())
    to_ascii_lowercase(test)
    assert_equal(test, expected)
    ```
    """
    if len(buffer) < SIMD_U8_WIDTH * 3:
        for i in range(0, len(buffer)):
            # I'm not sure why casting is needed. UInt8(is_ascii_uppercase) is being seen as a Bool for some reason
            buffer[i] |= (
                UInt8(is_ascii_uppercase(buffer[i])).cast[DType.uint8]() * 32
            )
        return

    # Initial unaligned set
    var ptr = buffer.unsafe_ptr()
    var v = ptr.load[width=SIMD_U8_WIDTH]()
    _to_ascii_lowercase_vec(v)
    ptr.store(0, v)

    # Now get an aligned pointer
    var offset = SIMD_U8_WIDTH - (ptr.__int__() & (SIMD_U8_WIDTH - 1))
    var aligned_ptr = ptr.offset(offset)

    # Find the last aligned read possible
    var buffer_len = len(buffer) - offset
    var aligned_end = math.align_down(
        buffer_len, SIMD_U8_WIDTH
    )  # relative to offset

    # Now do aligned reads all through
    for s in range(0, aligned_end, SIMD_U8_WIDTH):
        var v = aligned_ptr.load[width=SIMD_U8_WIDTH](s)
        _to_ascii_lowercase_vec(v)
        aligned_ptr.store(s, v)

    for i in range(aligned_end + offset, len(buffer)):
        buffer[i] |= (
            UInt8(is_ascii_uppercase(buffer[i])).cast[DType.uint8]() * 32
        )


@always_inline
fn _to_ascii_lowercase_vec(mut v: SIMD[DType.uint8, SIMD_U8_WIDTH]):
    """Convert a vec to ascii lowercase."""
    var ge_A = v >= CAPITAL_A
    var le_Z = v <= CAPITAL_Z
    var is_upper = ge_A.__and__(le_Z)
    v |= ASCII_CASE_MASK * is_upper.cast[DType.uint8]()


@always_inline
fn to_ascii_uppercase(mut buffer: List[UInt8, _]):
    """Uppercase all ascii a-zA-Z characters.

    ```mojo
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import to_ascii_uppercase
    var test = List("ABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZABCdefgHIjklmnOPQRSTUVWXYZ".as_bytes())
    var expected = List("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ".as_bytes())
    to_ascii_uppercase(test)
    assert_equal(test, expected)
    ```
    """
    if len(buffer) < SIMD_U8_WIDTH * 3:
        for i in range(0, len(buffer)):
            buffer[i] ^= (
                UInt8(is_ascii_lowercase(buffer[i])).cast[DType.uint8]() * 32
            )
        return

    # Initial unaligned set
    var ptr = buffer.unsafe_ptr()
    var v = ptr.load[width=SIMD_U8_WIDTH]()
    _to_ascii_uppercase_vec(v)
    ptr.store(0, v)

    # Now get an aligned pointer
    var offset = SIMD_U8_WIDTH - (ptr.__int__() & (SIMD_U8_WIDTH - 1))
    var aligned_ptr = ptr.offset(offset)

    # Find the last aligned read possible
    var buffer_len = len(buffer) - offset
    var aligned_end = math.align_down(
        buffer_len, SIMD_U8_WIDTH
    )  # relative to offset

    # Now do aligned reads all through
    for s in range(0, aligned_end, SIMD_U8_WIDTH):
        var v = aligned_ptr.load[width=SIMD_U8_WIDTH](s)
        _to_ascii_uppercase_vec(v)
        aligned_ptr.store(s, v)

    for i in range(aligned_end + offset, len(buffer)):
        buffer[i] ^= (
            UInt8(is_ascii_lowercase(buffer[i])).cast[DType.uint8]() * 32
        )


@always_inline
fn _to_ascii_uppercase_vec(mut v: SIMD[DType.uint8, SIMD_U8_WIDTH]):
    """Convert a vec to ASCII upercase."""
    var ge_a = v >= LOWER_A
    var le_z = v <= LOWER_Z
    var is_lower = ge_a.__and__(le_z)
    v ^= ASCII_CASE_MASK * is_lower.cast[DType.uint8]()


fn find(haystack: Span[UInt8], needle: Span[UInt8]) -> Optional[Int]:
    """Look for the substring `needle` in the haystack.

    This is not a terribly smart find implementation. It will use `memchr` to find
    occurrences of the first byte in the `needle`, it then checks the subsequent bytes to see
    if they match the rest of the needle.

    ```mojo
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import find

    var haystack = "ABCDEFGhijklmnop".as_bytes()
    var expected = 4
    var answer = find(haystack, "EFG".as_bytes()).value()
    assert_equal(answer, expected)
    ```

    Args:
        haystack: The bytes to be searched for the `needle`.
        needle: The bytes to search for in the `haystack`.

    Returns:
        Index of the start of the first occurrence of needle.

    """
    # https://github.com/BurntSushi/bstr/blob/master/src/ext_slice.rs#L3094
    # https://github.com/BurntSushi/memchr/blob/master/src/memmem/searcher.rs

    # Naive-ish search. Use our simd accel searcher to find the first char in the needle
    # check for extension, and move forward
    var start = 0
    while start < len(haystack):
        start = memchr(haystack, needle[0], start)
        if start == -1:
            return None
        # Try extension
        var matched = True
        for i in range(1, len(needle)):
            if haystack[start + i] != needle[i]:
                matched = False
                break
        if matched:
            return start
        else:
            start = start + 1
    return None


@fieldwise_init
@register_passable
struct _StartEnd(Copyable, Movable):
    """Helper struct for tracking start/end coords in `SplitIterator`"""

    var start: Int
    var end: Int


struct SplitIterator[is_mutable: Bool, //, origin: Origin[is_mutable]](
    Copyable, ExplicitlyCopyable, Movable
):
    """
    Get an iterator the yields the splits from the input `to_split` string.

    TODO: these test run fine in the test module, but not in doctests.

    ```
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import SplitIterator
    var input = "ABCD\tEFGH\tIJKL\nMNOP".as_bytes()
    var expected = List(
        "ABCD".as_bytes(), "EFGH".as_bytes(), "IJKL\nMNOP".as_bytes()
    )
    var output = List[Span[UInt8, StaticConstantOrigin]]()
    for value in SplitIterator(input, ord("\t")):
        output.append(value)
    for i in range(len(expected)):
        assert_equal(StringSlice(unsafe_from_utf8=output[i]), StringSlice(unsafe_from_utf8=expected[i]))
    ```

    ```
    from collections.string.string_slice import StringSlice
    from testing import assert_equal
    from ExtraMojo.bstr.bstr import SplitIterator

    var input = "ABCD\tEFGH\tIJKL\nMNOP".as_bytes()
    var expected = List(
        "ABCD".as_bytes(), "EFGH".as_bytes(), "IJKL\nMNOP".as_bytes()
    )
    var iter = SplitIterator(input, ord("\t"))
    var first = iter.__next__()
    var peek = iter.peek()
    var second = iter.__next__()
    assert_equal(StringSlice(unsafe_from_utf8=peek.value()), StringSlice(unsafe_from_utf8=second))
    assert_equal(StringSlice(unsafe_from_utf8=first), StringSlice(unsafe_from_utf8=expected[0]))
    assert_equal(StringSlice(unsafe_from_utf8=second), StringSlice(unsafe_from_utf8=expected[1]))
    ```
    """

    var inner: Span[UInt8, origin]
    var split_on: UInt8
    var current: Int
    var len: Int
    var next_split: Optional[_StartEnd]

    fn __init__(out self, to_split: Span[UInt8, origin], split_on: UInt8):
        self.inner = to_split
        self.split_on = split_on
        self.current = 0
        self.len = 1
        self.next_split = None
        self._find_next_split()

    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __len__(read self) -> Int:
        return self.len

    @always_inline
    fn __has_next__(read self) -> Bool:
        return self.__len__() > 0

    fn __next__(mut self) -> Span[UInt8, origin]:
        var ret = self.next_split.value()

        self._find_next_split()
        return self.inner[ret.start : ret.end]

    fn _find_next_split(mut self):
        if self.current >= len(self.inner):
            self.next_split = None
            self.len = 0
            return

        var end = memchr(self.inner, self.split_on, self.current)

        if end != -1:
            self.next_split = _StartEnd(self.current, end)
            self.current = end + 1
        else:
            self.next_split = _StartEnd(self.current, len(self.inner))
            self.current = len(self.inner) + 1

    fn peek(read self) -> Optional[Span[UInt8, origin]]:
        """Peek ahead at the next split result."""
        if self.next_split:
            var split = self.next_split.value()
            return self.inner[split.start : split.end]
        else:
            return None
