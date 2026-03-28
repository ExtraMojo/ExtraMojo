from std.testing import assert_equal, TestSuite

from extramojo.math.ops import saturating_add, saturating_sub


def _sat_add[dtype: DType where dtype.is_integral(), width: Int]() raises:
    comptime MIN = Scalar[dtype].MIN
    comptime MAX = Scalar[dtype].MAX

    var lhs = SIMD[dtype, width](MAX)
    var rhs = SIMD[dtype, width](1)
    var expected = SIMD[dtype, width](MAX)
    assert_equal(saturating_add[dtype, width](lhs, rhs), expected)

    expected = SIMD[dtype, width](2)
    assert_equal(saturating_add[dtype, width](rhs, rhs), expected)


def test_saturating_add() raises:
    comptime dtypes = (
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
    )
    comptime widths = (4, 8, 16, 32, 64, 128)

    comptime for i in range(0, len(dtypes)):
        comptime for j in range(0, len(widths)):
            comptime dtype = dtypes[i]
            comptime width = widths[j]
            comptime if dtype.is_integral():
                _sat_add[dtype, width]()


def _sat_sub[dtype: DType where dtype.is_integral(), width: Int]() raises:
    comptime MIN = Scalar[dtype].MIN
    comptime MAX = Scalar[dtype].MAX

    var lhs = SIMD[dtype, width](MIN)
    var rhs = SIMD[dtype, width](1)
    var expected = SIMD[dtype, width](MIN)
    assert_equal(saturating_sub(lhs, rhs), expected)

    expected = SIMD[dtype, width](0)
    assert_equal(saturating_sub(rhs, rhs), expected)


def test_saturating_sub() raises:
    comptime dtypes = [
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
    ]
    comptime widths = [4, 8, 16, 32, 64, 128]

    comptime for i in range(0, len(dtypes)):
        comptime for j in range(0, len(widths)):
            comptime dtype = dtypes[i]
            comptime width = widths[j]
            comptime if dtype.is_integral():
                _sat_sub[dtype, width]()


# def test_fastmod():
#     comptime types = [
#         DType.uint8,
#         DType.uint16,
#         DType.uint32,
#         DType.uint64,
#         DType.uint128,
#     ]

#     @parameter
#     for i in range(0, len(types)):
#         var x = ~Scalar[types[i]](0)
#         assert_equal(fastmod(x, 2), x % 2)


# def test_fastmod_specific():
#     assert_equal(fastmod(UInt64(17771040589687592464), UInt64(64)), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
