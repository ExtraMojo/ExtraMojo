from sys import llvm_intrinsic


@always_inline
fn saturating_sub[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://llvm.org/docs/LangRef.html#llvm-usub-sat-intrinsics
    https://llvm.org/docs/LangRef.html#llvm-ssub-sat-intrinsics
    """
    constrained[data.is_integral()]()

    @parameter
    if data.is_unsigned():
        return llvm_intrinsic["llvm.usub.sat", type_of(lhs)](lhs, rhs)
    else:
        return llvm_intrinsic["llvm.ssub.sat", type_of(lhs)](lhs, rhs)


@always_inline
fn saturating_add[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD addition.

    https://llvm.org/docs/LangRef.html#llvm-uadd-sat-intrinsics
    https://llvm.org/docs/LangRef.html#llvm-sadd-sat-intrinsics
    """
    constrained[data.is_integral()]()

    @parameter
    if data.is_unsigned():
        return llvm_intrinsic["llvm.uadd.sat", type_of(lhs)](lhs, rhs)
    else:
        return llvm_intrinsic["llvm.sadd.sat", type_of(lhs)](lhs, rhs)


# @always_inline
# fn fastmod[
#     dtype: DType
# ](hash: Scalar[dtype], n: Scalar[dtype]) -> Scalar[dtype]:
#     """https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
#     """
#     constrained[dtype.is_unsigned(), "dtype must be unsigned."]()
#     constrained[
#         dtype is not DType.uint256, "dtype must be smaller than UInt256"
#     ]()

#     comptime upsized = (
#         DType.uint16 if dtype
#         is DType.uint8 else DType.uint32 if dtype
#         is dtype.uint16 else DType.uint64 if dtype
#         is DType.uint32 else DType.uint128 if dtype
#         is DType.uint64 else DType.uint256
#     )

#     return (
#         ((hash.cast[upsized]()) * (n.cast[upsized]())) >> bit_width_of[dtype]()
#     ).cast[dtype]()
