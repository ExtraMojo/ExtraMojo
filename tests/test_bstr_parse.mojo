from std.memory import bitcast
from std.testing import assert_equal, assert_raises, TestSuite

from extramojo.bstr.parse import _scan_digits, parse_decimal


def _assert_parse(value: String, expected: Float64) raises:
    assert_equal(
        bitcast[DType.uint64](parse_decimal(value.as_bytes())),
        bitcast[DType.uint64](expected),
        "Mismatch for: " + value,
    )


def _assert_parse_bits(value: String, expected: UInt64) raises:
    assert_equal(
        bitcast[DType.uint64](parse_decimal(value.as_bytes())),
        expected,
        "Mismatch for: " + value,
    )


def test_parse_decimal() raises:
    var cases: List[Tuple[String, Float64]] = [
        ("1", 1.0),
        ("+1", 1.0),
        ("-1", -1.0),
        ("12.5", 12.5),
        ("-.125e2", -12.5),
        (".5", 0.5),
        ("+.5", 0.5),
        ("1.", 1.0),
        ("1.e2", 100.0),
        ("1.25E-2", 0.0125),
        ("+1.25e+2", 125.0),
        ("1.25e0002", 125.0),
        ("0.1234567890", 0.1234567890),
        ("0.9999999999", 0.9999999999),
        ("1e22", 1e22),
        ("1e23", 1e23),
        ("1e-22", 1e-22),
        ("1e-23", 1e-23),
        ("000123.4500", 123.45),
    ]
    for value, expected in cases:
        _assert_parse(value, expected)


def test_parse_decimal_doc_example() raises:
    assert_equal(parse_decimal("12.5".as_bytes()), Float64(12.5))
    assert_equal(parse_decimal("-.125e2".as_bytes()), Float64(-12.5))
    var buffer = List("x=0.25,".as_bytes())
    assert_equal(parse_decimal(Span(buffer)[2:6]), Float64(0.25))


def test_parse_decimal_signed_zero() raises:
    var positive: List[String] = [
        "0",
        "+0",
        "0.",
        ".0",
        "0.00000000",
        "0e9999",
        "0e-9999",
        "0e2147483647",
        "0e-2147483647",
        "1e-5000",
    ]
    for value in positive:
        _assert_parse_bits(value, 0)
        _assert_parse_bits("-" + value.removeprefix("+"), 0x8000000000000000)


def test_parse_decimal_numeric_limits() raises:
    var cases: List[Tuple[String, UInt64]] = [
        ("18446744073709551615", UInt64(0x43F0000000000000)),
        (".18446744073709551615e20", UInt64(0x43F0000000000000)),
        ("9007199254740993", UInt64(0x4340000000000000)),
        ("9007199254740995", UInt64(0x4340000000000002)),
        ("2.2250738585072011e-308", UInt64(0x000FFFFFFFFFFFFF)),
        ("2.2250738585072013e-308", UInt64(0x0010000000000000)),
        ("2.2250738585072014e-308", UInt64(0x0010000000000000)),
        ("5e-324", UInt64(0x0000000000000001)),
        ("2.4703282292062327e-324", UInt64(0)),
        ("2.4703282292062328e-324", UInt64(1)),
        ("1e-342", UInt64(0)),
        ("1e-343", UInt64(0)),
        ("1.7976931348623157e308", UInt64(0x7FEFFFFFFFFFFFFF)),
        ("1.7976931348623159e308", UInt64(0x7FF0000000000000)),
        ("1e309", UInt64(0x7FF0000000000000)),
        ("1e5000", UInt64(0x7FF0000000000000)),
        ("1e2147483647", UInt64(0x7FF0000000000000)),
        ("1e-2147483647", UInt64(0)),
    ]
    for value, expected in cases:
        _assert_parse_bits(value, expected)
        _assert_parse_bits("-" + value, expected | 0x8000000000000000)


def test_parse_decimal_rounding_regressions() raises:
    # Exact bit patterns verified against Zig 0.16.0 and Ruby's Float parser.
    # Mojo 1.0.0's atof rounds these one ULP too low. Do not use it as an oracle.
    var cases: List[Tuple[String, UInt64]] = [
        # False-positive halfway detection.
        ("1154050619.162411961e10", UInt64(0x43E40503A795FE75)),
        (".1659421546721668875e20", UInt64(0x43ECC94FE18537CB)),
        ("63409.4969512038461e14", UInt64(0x43D5FFE509B91169)),
        # Missing product refinement.
        ("50.48026509774068137e202", UInt64(0x6A39C2D7FBB5A68F)),
        ("324712100448695786.1e-42", UInt64(0x3AD91F95F4C5A1EF)),
        ("9016829116255082.147e236", UInt64(0x743F7C0B42DAE41C)),
    ]
    for value, expected in cases:
        _assert_parse_bits(value, expected)
        _assert_parse_bits("+" + value, expected)
        _assert_parse_bits("-" + value, expected | 0x8000000000000000)


def test_parse_decimal_invalid() raises:
    var cases: List[String] = [
        "",
        "+",
        "-",
        ".",
        "+.",
        "-.",
        "e1",
        ".e1",
        "1e",
        "1E",
        "1e+",
        "1e-",
        "1e+-2",
        "1e--2",
        "1e2.0",
        "1e2e3",
        "1.2.3",
        "++1",
        "--1",
        "+-1",
        "1+",
        "1-",
        "1 2",
        " 1",
        "1 ",
        "1\n",
        "\t1",
        "1,",
        "12x",
        "1_2",
        "0x1p2",
        "NaN",
        "nan",
        "Inf",
        "+inf",
        "-Infinity",
        "1f",
        "1F",
        "1\x00",
        "1\x001",
        "1\u00e9",
        "1234567x",
        "12345678x",
        "123456789x",
        "1234567890123456x",
    ]
    for value in cases:
        with assert_raises():
            _ = parse_decimal(value.as_bytes())


def test_parse_decimal_overflow() raises:
    var cases: List[String] = [
        "18446744073709551616",
        "-18446744073709551616",
        "1844674407370955161.6",
        "184467440737095516150",
        "18446744073709551616e-100",
        "999999999999999999999999",
        "1e2147483648",
        "1e-2147483648",
        "0e2147483648",
        "1e99999999999999999999999999999999999999999999999999",
    ]
    for value in cases:
        with assert_raises():
            _ = parse_decimal(value.as_bytes())


def test_parse_decimal_leading_zeros() raises:
    var zeros = String()
    for _ in range(128):
        zeros += "0"
    _assert_parse_bits(zeros, 0)
    _assert_parse_bits("-" + zeros, 0x8000000000000000)
    _assert_parse(zeros + "1", 1.0)
    _assert_parse(zeros + "." + zeros + "1e129", 1.0)
    _assert_parse_bits(zeros + "18446744073709551615", 0x43F0000000000000)


def test_parse_decimal_unaligned_slices() raises:
    var cases: List[Tuple[String, Float64]] = [
        ("1", 1.0),
        ("1234567", 1234567.0),
        ("12345678", 12345678.0),
        ("123456789", 123456789.0),
        ("123456789012345", 123456789012345.0),
        ("1234567890123456", 1234567890123456.0),
        ("12345678901234567", 12345678901234567.0),
        (".12345678", 0.12345678),
        ("0.1234567890", 0.1234567890),
        ("12345678.5", 12345678.5),
        ("-12345678e-8", -0.12345678),
    ]
    for offset in range(16):
        for value, expected in cases:
            var buffer = List[UInt8]()
            for _ in range(offset):
                buffer.append(255)
            buffer.extend(value.as_bytes())
            var end = len(buffer)
            buffer.append(255)
            var actual = parse_decimal(Span(buffer)[offset:end])
            assert_equal(
                bitcast[DType.uint64](actual),
                bitcast[DType.uint64](expected),
                "Unaligned input: " + value,
            )


def test_parse_decimal_non_ascii_bytes() raises:
    for invalid in range(128, 256):
        for offset in range(9):
            var buffer = List("123456789".as_bytes())
            buffer[offset] = UInt8(invalid)
            with assert_raises():
                _ = parse_decimal(Span(buffer))


def test_scan_digits_all_byte_values() raises:
    # Check SWAR classification against a scalar scan for every possible byte
    # at every position in an eight-byte load, including bytes above ASCII.
    for position in range(8):
        for replacement in range(256):
            var bytes8: List[UInt8] = [49, 50, 51, 52, 53, 54, 55, 56]
            bytes8[position] = UInt8(replacement)
            var expected_i = 0
            var expected_w = UInt64(0)
            while expected_i < 8 and UInt8(48) <= bytes8[expected_i] <= 57:
                expected_w = expected_w * 10 + UInt64(bytes8[expected_i] - 48)
                expected_i += 1
            var i = 0
            var w = UInt64(0)
            _scan_digits(Span(bytes8), i, w)
            assert_equal(i, expected_i)
            assert_equal(w, expected_w)


def test_scan_digits_accumulation() raises:
    var cases: List[String] = [
        "",
        "0",
        "1",
        "1234567",
        "12345678",
        "123456789",
        "1234567890123456",
        "1234567890123456789",
    ]
    for value in cases:
        var data = value.as_bytes()
        var expected = UInt64(0)
        for c in data:
            expected = expected * 10 + UInt64(c - 48)
        var w = UInt64(0)
        var i = 0
        _scan_digits(data, i, w)
        assert_equal(i, len(data))
        assert_equal(w, expected)


def test_parse_decimal_generated_fast_path() raises:
    var state = UInt64(0x123456789ABCDEF0)
    for i in range(5000):
        state = state * 6364136223846793005 + 1442695040888963407
        var w = state >> 11
        var q = (i % 45) - 22
        var scale = Float64(1)
        for _ in range(-q if q < 0 else q):
            scale *= 10
        # Both w and scale are exact Float64 integers in this range, so the
        # single division/multiplication supplies an independent scan oracle.
        var expected = Float64(w) / scale if q < 0 else Float64(w) * scale
        var value = String(w) + "e" + String(q)
        _assert_parse(value, expected)
        _assert_parse("-" + value, -expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
