"""Floating-point parsing for byte strings, with Zig-compatible Float64 syntax.

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

from ._parse_float import _decimal_fallback


def parse_float(data: Span[UInt8, _]) raises -> Float64:
    """Parse an ASCII byte string using Zig 0.16's `Float64` input syntax.

    Reads the bytes directly, validating and accumulating up to eight digits
    at a time on the common decimal path. Accepts an optional sign, a decimal
    point, decimal `e`/`E` exponents, and hexadecimal numbers prefixed with
    `0x`/`0X` with optional binary `p`/`P` exponents. Underscores may separate
    digits. Also accepts case-insensitive `nan`, `inf`, and `infinity`.
    NaN has Zig's canonical positive quiet-NaN representation, even for `-nan`.

    Significands and exponents have no fixed input-length limit. An exact
    fallback handles long decimals with bounded temporary integer storage;
    ordinary decimals and hexadecimal values do not allocate temporary strings.
    Floating-point underflow produces signed zero; overflow produces signed
    infinity. Negative zero is preserved, and rounding is nearest, ties to even.

    The entire input must be consumed. Whitespace, decimal type suffixes,
    NaN payloads, and malformed separators are rejected. The supported result
    type is `Float64`, not Zig's other floating-point types.

    ```mojo
    from std.testing import assert_equal
    from extramojo.bstr.parse import parse_float

    assert_equal(parse_float("12.5".as_bytes()), Float64(12.5))
    assert_equal(parse_float("-.125e2".as_bytes()), Float64(-12.5))
    assert_equal(parse_float("0x1.8p+1".as_bytes()), Float64(3))

    var buffer = List("x=0.25,".as_bytes())
    assert_equal(parse_float(Span(buffer)[2:6]), Float64(0.25))
    ```

    Args:
        data: The complete ASCII floating-point field to parse.

    Returns:
        The parsed `Float64`, rounded to nearest with ties to even.

    Raises:
        If the input is empty or contains invalid syntax.
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
    if not _scan_digits(data, i, w):
        return _parse_extended(data)
    var digit_count = i - start
    var q = Int64(0)
    if i < n and data[i] == 46:  # '.'
        i += 1
        start = i
        if not _scan_digits(data, i, w):
            return _parse_extended(data)
        q = -Int64(i - start)
        digit_count += i - start
    if digit_count == 0:
        return _parse_extended(data)

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
                return _parse_extended(data)
            exponent = exponent * 10 + digit
            i += 1
        if i == start:
            raise Error("Expected exponent digits")
        q += -exponent if exponent_negative else exponent

    if i != n:
        return _parse_extended(data)

    # Reuse Mojo's numeric conversion, with the version-specific fixes below.
    var value: Float64
    if can_use_clinger_fast_path(w, q):
        value = clinger_fast_path(w, q)
    else:
        value = _lemire(w, q)
    return -value if negative else value


@always_inline
def parse_decimal(data: Span[UInt8, _]) raises -> Float64:
    """Compatibility alias for `parse_float`, including its extended syntax.

    Args:
        data: The complete ASCII floating-point field to parse.

    Returns:
        The parsed `Float64`.

    Raises:
        If the input is empty or contains invalid syntax.
    """
    return parse_float(data)


@always_inline
def _scan_digits(data: Span[UInt8, _], mut i: Int, mut w: UInt64) -> Bool:
    """Consume decimal digits; return False before overflowing `w`."""
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
            return False
        w = w * 100000000 + chunk
        i += 8

    while i < len(data):
        var c = data[i]
        if c < 48 or c > 57:
            break
        var digit = UInt64(c - 48)
        if w > (MAX - digit) // 10:
            return False
        w = w * 10 + digit
        i += 1
    return True


@always_inline
def _digit(c: UInt8, hexadecimal: Bool = False) -> Int:
    if UInt8(48) <= c <= UInt8(57):
        return Int(c - 48)
    var lower = c | 32
    if hexadecimal and UInt8(97) <= lower <= UInt8(102):
        return Int(lower - 97) + 10
    return -1


def _equal_fold(data: Span[UInt8, _], word: StringSlice) -> Bool:
    if len(data) != word.byte_length():
        return False
    var expected = word.as_bytes()
    for i in range(len(data)):
        if (data[i] | 32) != expected[i]:
            return False
    return True


@no_inline
def _parse_extended(data: Span[UInt8, _]) raises -> Float64:
    """Validate extended syntax; keep the ordinary decimal path small."""
    var n = len(data)
    var i = 0
    var negative = data[0] == 45
    if negative or data[0] == 43:
        i += 1
    if i == n:
        raise Error("Expected digits")
    var unsigned = data[i:]
    if _equal_fold(unsigned, "nan"):
        return bitcast[DType.float64](UInt64(0x7FF8000000000000))
    if _equal_fold(unsigned, "inf") or _equal_fold(unsigned, "infinity"):
        return -Float64(
            FloatLiteral.infinity
        ) if negative else FloatLiteral.infinity

    var hexadecimal = i + 1 < n and data[i] == 48 and (data[i + 1] | 32) == 120
    if hexadecimal:
        i += 2
    var start = i
    var w = UInt64(0)
    var kept = 0
    var significant_digits = 0
    var fraction_digits = 0
    var saw_digit = False
    var saw_dot = False
    var previous_digit = False
    var sticky = False
    var limit = 16 if hexadecimal else 19
    while i < n:
        var c = data[i]
        var digit = _digit(c, hexadecimal)
        if digit >= 0:
            saw_digit = True
            previous_digit = True
            if saw_dot:
                fraction_digits += 1
            if significant_digits != 0 or digit != 0:
                significant_digits += 1
                if kept < limit:
                    w = w * UInt64(16 if hexadecimal else 10) + UInt64(digit)
                    kept += 1
                else:
                    sticky = sticky or digit != 0
        elif c == 46 and not saw_dot:
            saw_dot = True
            previous_digit = False
        elif c == 95:
            if (
                not previous_digit
                or i + 1 == n
                or _digit(data[i + 1], hexadecimal) < 0
            ):
                raise Error("Invalid digit separator")
            previous_digit = False
        else:
            break
        i += 1
    if not saw_digit:
        raise Error("Expected digits")
    var end = i
    # More than 4*n + 4096 cannot be cancelled by this significand, even in
    # hexadecimal. Int128 keeps length/exponent arithmetic safe for any span.
    var exponent = Int128(0)
    var exponent_limit = Int128(n) * 4 + 4096
    var marker = UInt8(112) if hexadecimal else UInt8(101)
    if i < n and (data[i] | 32) == marker:
        i += 1
        var exponent_negative = False
        if i < n and (data[i] == 43 or data[i] == 45):
            exponent_negative = data[i] == 45
            i += 1
        previous_digit = False
        var exponent_digit = False
        while i < n:
            var digit = _digit(data[i])
            if digit >= 0:
                exponent_digit = True
                previous_digit = True
                if exponent < exponent_limit:
                    exponent = exponent * 10 + Int128(digit)
            elif data[i] == 95:
                if not previous_digit or i + 1 == n or _digit(data[i + 1]) < 0:
                    raise Error("Invalid exponent separator")
                previous_digit = False
            else:
                break
            i += 1
        if not exponent_digit:
            raise Error("Expected exponent digits")
        if exponent_negative:
            exponent = -exponent
    if i != n:
        raise Error("Invalid float character")

    var value: Float64
    if w == 0:
        value = 0.0
    elif hexadecimal:
        var q = exponent + 4 * Int128(
            significant_digits - kept - fraction_digits
        )
        if q < -4096:
            value = 0.0
        elif q > 4096:
            value = FloatLiteral.infinity
        else:
            value = _hex_float(w, Int64(q), sticky)
    else:
        var point = exponent + Int128(significant_digits - fraction_digits)
        if point < -323:
            value = 0.0
        elif point > 309:
            value = FloatLiteral.infinity
        else:
            var q = Int64(point) - Int64(kept)
            if not sticky and can_use_clinger_fast_path(w, q):
                value = clinger_fast_path(w, q)
            else:
                var lower = _lemire(w, q)
                if not sticky or lower == _lemire(w + 1, q):
                    value = lower
                else:
                    value = _decimal_fallback(data[start:end], Int64(point))
    return -value if negative else value


def _hex_float(w: UInt64, q: Int64, sticky: Bool) -> Float64:
    """Round retained hexadecimal bits plus sticky remainder to binary64."""
    var bits = 64 - Int(count_leading_zeros(w))
    var exponent = q + Int64(bits) - 1
    if exponent > 1023:
        return FloatLiteral.infinity
    var shift = Int64(bits - 53) if exponent >= -1022 else -q - 1074
    var m: UInt64
    if shift <= 0:
        m = w << UInt64(-shift)
    elif shift > 64:
        return 0.0
    else:
        var halfway = UInt64(1) << UInt64(shift - 1)
        var remainder: UInt64
        if shift == 64:
            m = 0
            remainder = w
        else:
            m = w >> UInt64(shift)
            remainder = w & ((UInt64(1) << UInt64(shift)) - 1)
        if remainder > halfway or (
            remainder == halfway and (sticky or (m & 1) != 0)
        ):
            m += 1
    if exponent < -1022:
        return bitcast[DType.float64](m)
    if m == UInt64(1 << 53):
        m >>= 1
        exponent += 1
    if exponent > 1023:
        return FloatLiteral.infinity
    return create_float64(m, exponent)


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
