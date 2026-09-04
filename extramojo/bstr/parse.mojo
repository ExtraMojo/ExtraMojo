"""Decimal floating-point parsing for byte strings.

Uses Zig-style digit scanning with Mojo's numeric conversion helpers. The
implementation relies on private APIs from the pinned Mojo 1.0.0 standard
library; retain the rounding regression tests when upgrading Mojo.

## References

- [Zig's float parser](https://codeberg.org/ziglang/zig/src/tag/0.16.0/lib/std/fmt/parse_float)
- [Mojo's float parser](https://github.com/modular/modular/blob/max/v26.5.0/mojo/stdlib/std/collections/string/_parsing_numbers/parsing_floats.mojo)
- [Source provenance and upstream references](https://github.com/ExtraMojo/ExtraMojo/blob/main/licenses/third-party/README.md)
"""

# Digit scanning and rounding checks adapted from Zig 0.16.0 (MIT).
# Copyright (c) Zig contributors. See licenses/third-party/zig.txt.
#
# Numeric conversion adapted from Modular's stdlib parsing_floats.mojo.
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See licenses/third-party/modular.txt.
# Modified for byte-string parsing and to correct the product-refinement mask
# and tie detection. See _lemire below.

from std.bit import byte_swap, count_leading_zeros
from std.collections.string._parsing_numbers.constants import (
    SMALLEST_POWER_OF_5,
    get_power_of_5,
)
from std.collections.string._parsing_numbers.parsing_floats import (
    can_use_clinger_fast_path,
    clinger_fast_path,
    create_float64,
    create_subnormal_float64,
    full_multiplication,
)
from std.memory import bitcast
from std.sys.info import is_big_endian


def parse_decimal(data: Span[UInt8, _]) raises -> Float64:
    """Parse an ASCII decimal byte string as a `Float64`.

    Reads the bytes directly, validating and accumulating up to eight digits
    at a time without allocating or creating a `String`. Accepts an optional
    sign, a decimal point, and a signed `e`/`E` exponent. At least one
    significand digit and, when present, one exponent digit are required.
    The entire input must be consumed.

    The complete significand (all digits with the decimal point removed) must
    fit in `UInt64`. Exponent magnitude must be at most `2147483647`.
    Leading zeros are allowed and do not count against the significand limit.
    Floating-point underflow produces signed zero; overflow produces signed
    infinity. Negative zero is preserved.

    Whitespace, hexadecimal floats, underscores, type suffixes, and NaN/Inf
    spellings are not accepted. This is not a general replacement for `atof`;
    overlong significands are rejected rather than truncated.

    ```mojo
    from std.testing import assert_equal
    from extramojo.bstr.parse import parse_decimal

    assert_equal(parse_decimal("12.5".as_bytes()), Float64(12.5))
    assert_equal(parse_decimal("-.125e2".as_bytes()), Float64(-12.5))

    var buffer = List("x=0.25,".as_bytes())
    assert_equal(parse_decimal(Span(buffer)[2:6]), Float64(0.25))
    ```

    Args:
        data: The complete ASCII decimal field to parse.

    Returns:
        The parsed `Float64`, rounded to nearest with ties to even.

    Raises:
        If the input is empty, contains invalid syntax, or exceeds the
        significand or exponent limits.
    """
    var n = len(data)
    if n == 0:
        raise Error("Empty float")

    var i = 0
    var negative = data[0] == 45  # '-'
    if negative or data[0] == 43:  # '+'
        i += 1

    # Accumulate both sides of the decimal point into the same integer.
    var w = UInt64(0)
    var start = i
    _scan_digits(data, i, w)
    var digit_count = i - start
    var q = Int64(0)
    if i < n and data[i] == 46:  # '.'
        i += 1
        start = i
        _scan_digits(data, i, w)
        q = -Int64(i - start)
        digit_count += i - start
    if digit_count == 0:
        raise Error("Expected digits")

    if i < n and (data[i] == 101 or data[i] == 69):  # 'e' / 'E'
        i += 1
        var exponent_negative = False
        if i < n and (data[i] == 45 or data[i] == 43):
            exponent_negative = data[i] == 45
            i += 1
        start = i
        var exponent = Int64(0)
        while i < n and data[i] >= 48 and data[i] <= 57:
            var digit = Int64(data[i] - 48)
            if exponent > (2147483647 - digit) // 10:
                raise Error("Exponent magnitude exceeds Int32 maximum")
            exponent = exponent * 10 + digit
            i += 1
        if i == start:
            raise Error("Expected exponent digits")
        q += -exponent if exponent_negative else exponent

    if i != n:
        raise Error("Invalid float character")

    # Reuse Mojo's numeric conversion, with the version-specific fixes below.
    var value: Float64
    if can_use_clinger_fast_path(w, q):
        value = clinger_fast_path(w, q)
    else:
        value = _lemire(w, q)
    return -value if negative else value


@always_inline
def _scan_digits(data: Span[UInt8, _], mut i: Int, mut w: UInt64) raises:
    """Consume decimal digits into `w`, checking for significand overflow."""
    # SWAR digit validation/conversion, as in Zig's parse8Digits.
    # UInt64 arithmetic wraps; overflow of the accumulated significand is
    # checked separately before each update. Loads never cross the span end.
    comptime MAX = UInt64(0xFFFFFFFFFFFFFFFF)
    while len(data) - i >= 8:
        var bytes8 = data.unsafe_ptr().unsafe_load[width=8, alignment=1](i)
        var v = bitcast[DType.uint64, 1](bytes8)
        comptime if is_big_endian():
            v = byte_swap(v)
        if (
            ((v + 0x4646464646464646) | (v - 0x3030303030303030))
            & 0x8080808080808080
        ) != 0:
            break
        v -= 0x3030303030303030
        v = v * 10 + (v >> 8)
        var a = (v & 0x000000FF000000FF) * 0x000F424000000064
        var b = ((v >> 16) & 0x000000FF000000FF) * 0x0000271000000001
        var chunk = UInt64(UInt32((a + b) >> 32))
        if w > (MAX - chunk) // 100000000:
            raise Error("Significand exceeds UInt64")
        w = w * 100000000 + chunk
        i += 8

    while i < len(data):
        var c = data[i]
        if c < 48 or c > 57:
            break
        var digit = UInt64(c - 48)
        if w > (MAX - digit) // 10:
            raise Error("Significand exceeds UInt64")
        w = w * 10 + digit
        i += 1


def _lemire(w: UInt64, q: Int64) -> Float64:
    """Mojo's Lemire structure with two checks corrected to match Zig.

    Reuses Mojo's power-of-five tables, UInt128 multiplication, and Float64
    constructors. Calling the unmodified lemire_algorithm would inherit both
    its product-refinement-mask bug and its false-positive tie detection.
    """
    if w == 0 or q < -342:
        return 0.0
    if q > 308:
        return FloatLiteral.infinity

    var leading = count_leading_zeros(w)
    var normalized = w << leading
    var index = Int(2 * (q - SMALLEST_POWER_OF_5))
    var product = full_multiplication(normalized, get_power_of_5(index))
    # We retain 55 high-word bits: test the OTHER 9 bits for all ones.
    # Mojo 1.0.0 tests (1 << 55) - 1 here, missing required refinements.
    if (product.high & 0x1FF) == 0x1FF:
        var second = full_multiplication(normalized, get_power_of_5(index + 1))
        product.low += second.high
        if product.low < second.high:
            product.high += 1

    var upper = product.most_significant_bit()
    var shift = upper + 9
    var m = product.high >> shift
    var p = ((Int64(217706) * q) >> 16) + 63 - Int64(leading) + Int64(upper)

    if p <= -1086:
        return 0.0
    if p < -1022:
        m >>= UInt64(-1022 - p)
        m += m & 1
        m >>= 1
        if m >= UInt64(1 << 52):
            return create_float64(m, -1022)
        return create_subnormal_float64(m)

    # A true halfway case requires EVERY discarded high-word bit to be zero.
    # pop_count(product.high // m) == 1 (Mojo 1.0.0) loses that information.
    if (
        product.low <= 1
        and Int64(-4) <= q <= Int64(23)
        and (m & 3) == 1
        and (m << shift) == product.high
    ):
        m &= ~UInt64(1)
    m += m & 1
    m >>= 1
    if m == UInt64(1 << 53):
        m >>= 1
        p += 1
    if p > 1023:
        return FloatLiteral.infinity
    return create_float64(m, p)
