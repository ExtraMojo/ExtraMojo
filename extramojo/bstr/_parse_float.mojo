"""Exact rounding for long decimal Float64 inputs.

Every finite binary64 rounding boundary has at most 768 significant decimal
digits: even a midpoint with denominator 2**1075 has a numerator with at most
54 binary bits, so multiplying it by 5**1075 needs at most 768 decimal digits.
Keeping 800 digits and a sticky bit therefore distinguishes every boundary.
The integers below stay bounded regardless of the length of the input.
"""

from std.bit import count_leading_zeros
from std.collections.string._parsing_numbers.parsing_floats import (
    create_float64,
)
from std.memory import bitcast


struct _BigUInt(Copyable, Movable):
    """Little-endian base-2**32 integer for the exact decimal fallback."""

    var words: List[UInt32]

    def __init__(out self, value: UInt32 = 0):
        self.words = []
        if value != 0:
            self.words.append(value)

    def multiply_add(mut self, factor: UInt32, digit: UInt32 = 0):
        var carry = UInt64(digit)
        for i in range(len(self.words)):
            carry += UInt64(self.words[i]) * UInt64(factor)
            self.words[i] = UInt32(carry)
            carry >>= 32
        if carry != 0:
            self.words.append(UInt32(carry))

    def bit_length(self) -> Int:
        if len(self.words) == 0:
            return 0
        return (
            (len(self.words) - 1) * 32
            + 32
            - Int(count_leading_zeros(self.words[len(self.words) - 1]))
        )

    def shifted(self, bits: Int) -> Self:
        var result = Self()
        if len(self.words) == 0:
            return result^
        for _ in range(bits // 32):
            result.words.append(0)
        var carry = UInt64(0)
        for word in self.words:
            var value = (UInt64(word) << UInt64(bits % 32)) | carry
            result.words.append(UInt32(value))
            carry = value >> 32
        if carry != 0:
            result.words.append(UInt32(carry))
        return result^

    def compare(self, other: Self) -> Int:
        if len(self.words) != len(other.words):
            return 1 if len(self.words) > len(other.words) else -1
        for i in range(len(self.words) - 1, -1, -1):
            if self.words[i] != other.words[i]:
                return 1 if self.words[i] > other.words[i] else -1
        return 0

    def subtract(mut self, other: Self):
        # The caller has already established self >= other.
        var borrow = UInt64(0)
        for i in range(len(self.words)):
            var word = UInt64(other.words[i]) if i < len(other.words) else 0
            var difference = UInt64(self.words[i]) - word - borrow
            self.words[i] = UInt32(difference)
            borrow = difference >> 63
        self._trim()

    def shift_right_one(mut self):
        var carry = UInt32(0)
        for i in range(len(self.words) - 1, -1, -1):
            var word = self.words[i]
            self.words[i] = (word >> 1) | carry
            carry = word << 31
        self._trim()

    def _trim(mut self):
        while len(self.words) != 0 and self.words[len(self.words) - 1] == 0:
            _ = self.words.pop()


@no_inline
def _decimal_fallback(digits: Span[UInt8, _], point: Int64) -> Float64:
    """Round a validated positive decimal, with -323 <= point <= 309.

    `digits` excludes the exponent but may contain a dot and underscores.
    `point` is the decimal point position relative to the first nonzero digit.
    Only an ambiguous, long significand needs this allocation-backed path.
    """
    var numerator = _BigUInt()
    var kept = 0
    var started = False
    var sticky = False
    for c in digits:
        if c == 46 or c == 95:
            continue
        var digit = UInt32(c - 48)
        if not started and digit == 0:
            continue
        started = True
        if kept < 800:
            numerator.multiply_add(10, digit)
            kept += 1
        else:
            sticky = sticky or digit != 0

    var denominator = _BigUInt(1)
    var q = Int(point) - kept
    if q >= 0:
        for _ in range(q):
            numerator.multiply_add(10)
    else:
        for _ in range(-q):
            denominator.multiply_add(10)

    # Find floor(log2(numerator / denominator)) using exact comparisons.
    var exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator.compare(denominator.shifted(exponent)) < 0:
            exponent -= 1
    elif numerator.shifted(-exponent).compare(denominator) < 0:
        exponent -= 1

    # Scale to an integer significand: 53 bits for normals, or units of
    # 2**-1074 for subnormals. The quotient has at most 53 bits.
    var scale = 1074 if exponent < -1022 else 52 - exponent
    if scale >= 0:
        numerator = numerator.shifted(scale)
    else:
        denominator = denominator.shifted(-scale)

    var quotient = UInt64(0)
    var shift = numerator.bit_length() - denominator.bit_length()
    if shift >= 0:
        var divisor = denominator.shifted(shift)
        for bit in range(shift, -1, -1):
            if numerator.compare(divisor) >= 0:
                numerator.subtract(divisor)
                quotient |= UInt64(1) << UInt64(bit)
            divisor.shift_right_one()

    var halfway = numerator.shifted(1).compare(denominator)
    if halfway > 0 or (halfway == 0 and (sticky or (quotient & 1) != 0)):
        quotient += 1
    if exponent < -1022:
        # Carry into bit 52 naturally produces the smallest normal.
        return bitcast[DType.float64](quotient)
    if quotient == UInt64(1 << 53):
        quotient >>= 1
        exponent += 1
    if exponent > 1023:
        return FloatLiteral.infinity
    return create_float64(quotient, Int64(exponent))
