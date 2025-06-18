"""Provides a growable bitfield.

Optimized for space (1 bit per element) and speed (O(1) operations).
Offers set/clear/test/toggle and fast population count. The underlying
storage grows automatically but does not shrink unless `shrink_to_fit`
is called (not implemented yet).

Example:
```mojo
    var bv = BitVec(length=128, fill=False) # 128-bit set, all clear
    bv.set(42)                  # Mark value 42 as present.
    if bv[42]:                  # Check membership, could use bv.test(42)
        print("hit")            # Prints "hit".
    bv.clear(42)                # Remove 42.
    print(bv.count_set_bits())  # Prints 0.
```
"""
from algorithm import vectorize
from bit import log2_floor, pop_count
from math import ceildiv
from memory import UnsafePointer, pack_bits, memcpy, memset, memset_zero
from os import abort
from sys.info import is_gpu, simdwidthof


@always_inline
fn _check_index_bounds[operation_name: StaticString](idx: UInt, max_size: Int):
    """Checks if the index is within bounds for a BitVec operation.

    Parameters:
        operation_name: The name of the operation for error reporting.

    Args:
        idx: The index to check.
        max_size: The maximum size of the BitVec.
    """
    debug_assert(
        idx < max_size,
        "BitVec index out of bounds when ",
        operation_name,
        " bit: ",
        idx,
        " >= ",
        max_size,
    )


@always_inline
fn _elts[dtype: DType](bits: UInt) -> UInt:
    """Compute the number of elements needed to hold the given number of bits.
    """
    alias bitwidth = dtype.bitwidth()

    @parameter
    if bitwidth == 0:
        return 0
    else:
        return (bits + bitwidth - 1) // bitwidth


@always_inline
fn _word_index[dtype: DType](idx: UInt) -> UInt:
    """Computes the 0-based index of the Self.WORD_DTYPE word containg bit `idx`
    """
    alias _WORD_BITS_LOG2 = log2_floor(dtype.bitwidth())
    return Int(idx >> _WORD_BITS_LOG2)


@always_inline
fn _bit_mask[dtype: DType](idx: UInt) -> Scalar[dtype]:
    """Returns a UInt64 mask with only the bit corresponding to `idx` set."""
    alias _WORD_BITS = dtype.bitwidth()
    return Scalar[dtype](1) << Scalar[dtype]((idx & (_WORD_BITS - 1)))


struct BitVec(Boolable, Copyable, ExplicitlyCopyable, Movable, Sized, Writable):
    """A growable bitfield.

    This uses one bit per bool for storage.

    The bits are stored in `Self.WORD_DTYPE` words. This is optimized
    for compactness and speed.
    """

    alias WORD_DTYPE = DType.uint64 if not is_gpu() else DType.uint32
    alias WORD_BYTEWIDTH = Self.WORD_DTYPE.bitwidth() // 8
    alias WORD = Scalar[Self.WORD_DTYPE]

    var data: UnsafePointer[Self.WORD, alignment = Self.WORD_BYTEWIDTH]
    """The data storage."""

    var _len: UInt
    """The current length in bits."""

    var _capacity: UInt
    """The capacity in words."""

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    @always_inline
    fn __init__(out self):
        self.data = UnsafePointer[self.WORD, alignment = Self.WORD_BYTEWIDTH]()
        self._len = 0  # in bits
        self._capacity = 0  # in words

    @always_inline
    fn __init__(out self, *, capacity: UInt):
        """Capacity measured in bits."""
        # TODO: alignment of 16 on GPU?
        # TODO: allow for using `external_memory` for GPU?
        var word_cap = _elts[Self.WORD_DTYPE](capacity)
        if capacity:
            self.data = UnsafePointer[
                self.WORD, alignment = self.WORD_BYTEWIDTH
            ].alloc(word_cap)
            memset_zero(self.data, word_cap)
        else:
            self.data = UnsafePointer[
                self.WORD, alignment = self.WORD_BYTEWIDTH
            ]()
        self._len = 0
        self._capacity = word_cap

    @always_inline
    fn __init__(out self, *, length: UInt, fill: Bool = False):
        """Create a new bitvec with a known length and fill.

        Args:
            length: Known length in bits.
            fill: The value to fill the bitvec with.
        """

        self = Self()
        self.resize(length, fill)

    @always_inline
    fn __init__(out self, owned *values: Bool, __list_literal__: () = ()):
        """Constructs a BitVec from the given values.

        Args:
            values: The values to populate the BitVec with.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        self = Self(elements=values^)

    fn __init__(out self, *, owned elements: VariadicListMem[Bool, _]):
        """Constructs a BitVec from the given values.

        Args:
            elements: The values to populate the list with.
        """

        self = Self(capacity=len(elements))

        for i in range(0, len(elements)):
            self.append(elements[i])

    # --------------------------------------------------------------------- #
    # Lifecycle methods
    # --------------------------------------------------------------------- #

    @always_inline
    fn copy(self) -> Self:
        var copy = Self(capacity=self._len)
        memcpy(copy.data, self.data, self._capacity)
        copy._len = self._len
        return copy^

    @always_inline
    fn __moveinit__(out self, owned other: Self):
        self.data = other.data
        self._len = other._len
        self._capacity = other._capacity

    @always_inline
    fn __del__(owned self):
        self.data.free()

    # --------------------------------------------------------------------- #
    # Capacity queries
    # --------------------------------------------------------------------- #

    @always_inline
    fn __len__(self) -> Int:
        """The number of bits in the bitvec."""
        return self._len

    @always_inline
    fn capacity(read self) -> UInt:
        """Returns the capacity in bits."""
        return self._capacity * self.WORD_DTYPE.bitwidth()

    @always_inline
    fn word_len(read self) -> UInt:
        """Get the number of words that have been set."""
        return _elts[self.WORD_DTYPE](self._len)

    @always_inline
    fn is_empty(self) -> Bool:
        """Checks if the BitVec has any values stored in it.

        Equivalent to `len(self) == 0`. Note that this checks the logical
        size, not the allocated capacity.

        Returns:
            True if no values are stored in the BitVec.
        """
        return len(self) == 0

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the bitset is non-empty (contains at least one value).

        Equivalent to `len(self) != 0` or `not self.is_empty()`.

        Returns:
            True if at least one value is in BitVec., False otherwise.
        """
        return not self.is_empty()

    # --------------------------------------------------------------------- #
    # Allocations
    # --------------------------------------------------------------------- #

    fn _realloc(mut self, new_capacity: UInt):
        """Reallocate to for new_capacity, which is in words."""
        var new_data = UnsafePointer[
            self.WORD, alignment = self.WORD_BYTEWIDTH
        ].alloc(new_capacity)
        var current_words = _elts[self.WORD_DTYPE](self._len)
        memset_zero(
            new_data.offset(current_words), new_capacity - current_words
        )
        memcpy(new_data, self.data, current_words)

        if self.data:
            self.data.free()
        self.data = new_data
        self._capacity = new_capacity

    fn resize(mut self, new_size: UInt, fill: Bool):
        """Resize the bitvec, filling any new size with fill.

        Args:
            new_size: The new size in bits.
            fill: The value to use to populate new elements.
        """

        if new_size <= self._len:
            self.shrink(new_size)
            return

        var old_words = _elts[self.WORD_DTYPE](self._len)
        var old_len = self._len
        self.reserve(new_size)

        # set new mem
        if self._capacity > old_words:
            memset(
                self.data.offset(old_words),
                0xFF if fill else 0x00,
                self._capacity - old_words,
            )

        # Set the bits in the last word of the old values
        var bit_offset = old_len % self.WORD_DTYPE.bitwidth()
        if bit_offset != 0:
            var mask = (1 << bit_offset) - 1
            if fill:
                # Fill upper bits in word to 1
                self.data[old_words - 1] |= ~mask
            else:
                # Clear upper bits
                self.data[old_words - 1] &= mask

        # Set the bits in the last word
        bit_offset = new_size % self.WORD_DTYPE.bitwidth()
        if bit_offset != 0 and fill:
            var mask = (1 << bit_offset) - 1
            # clear upper bits
            self.data[self._capacity - 1] &= mask
        self._len = new_size

    @always_inline
    fn shrink(mut self, new_size: UInt):
        """Resizes to the given new size (in bits) which must be <= the current size.

        Args:
            new_size: The new size in bits.

        Notes:
            With no new value provided, the new size must be smaller than or equal to the
            current one. Elements at the end are discarded.
        """
        if self._len < new_size:
            abort(
                "You are calling BitVec.resize with a new_size bigger than the"
                " current size. If you want to make the BitVec bigger, provide"
                " a value to fill the new slots with. If not, make sure the new"
                " size is smaller than the current size."
            )

        self._len = new_size
        self.reserve(new_size)

    @always_inline
    fn reserve(mut self, new_capacity: UInt):
        """Reserves the requested capacity (in bits).

        Args:
            new_capacity: The new capacity, in bits.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the data is moved.
        """
        var word_cap = _elts[self.WORD_DTYPE](new_capacity)
        if self._capacity >= word_cap:
            return
        self._realloc(word_cap)

    # --------------------------------------------------------------------- #
    # Dunder Methods
    # --------------------------------------------------------------------- #

    @always_inline
    fn __getitem__(read self, idx: UInt) -> Bool:
        """Get the bit at the given index.

        Args:
            idx: The index of the bit.
        Returns:
            A bool, True if the bit was set, False if it was not.
        """
        _check_index_bounds["__getitem__"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        return Bool(self.data[w] & _bit_mask[self.WORD_DTYPE](idx))

    @always_inline
    fn __setitem__(mut self, idx: UInt, value: Bool):
        """Set the bit at the given index.

        Args:
            idx: The index of the bit to set.
            value: The value to set the bit to.
        """
        _check_index_bounds["__setitem__"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        if value:
            self.data[w] |= _bit_mask[self.WORD_DTYPE](idx)
        else:
            self.data[w] &= ~_bit_mask[self.WORD_DTYPE](idx)

    @always_inline
    fn __or__(read self, read other: Self) -> Self:
        return self.union(other)

    @always_inline
    fn __ior__(mut self, read other: Self):
        self.union_update(other)

    @always_inline
    fn __sub__(read self, read other: Self) -> Self:
        return self.difference(other)

    @always_inline
    fn __isub__(mut self, read other: Self):
        self.difference_update(other)

    @always_inline
    fn __and__(read self, read other: Self) -> Self:
        return self.intersection(other)

    @always_inline
    fn __iand__(mut self, read other: Self):
        self.intersection_update(other)

    fn __eq__(read self, read other: Self) -> Bool:
        alias width = simdwidthof[Scalar[self.WORD_DTYPE]]()

        if len(self) != len(other):
            return False

        var equal = True

        @parameter
        @always_inline
        fn equality[simd_width: Int](offset: Int):
            # TODO: is there a better way to skip this work?
            if equal:
                var lhs = SIMD[self.WORD_DTYPE, simd_width]()
                var rhs = SIMD[self.WORD_DTYPE, simd_width]()

                equal = Bool(Scalar[self.WORD_DTYPE](pack_bits(lhs == rhs)))

        # Leave the last word off to handle specially
        var words = _elts[self.WORD_DTYPE](len(self))
        vectorize[equality, width](words - 1)

        if equal:
            var mask = len(self) % self.WORD_DTYPE.bitwidth()
            equal = (self.data[words - 1] & mask) == (
                other.data[words - 1] & mask
            )

        return equal

    fn __ne__(read self, read other: Self) -> Bool:
        return not (self == other)

    # --------------------------------------------------------------------- #
    # Methods
    # --------------------------------------------------------------------- #
    @always_inline
    fn test(self, idx: UInt) -> Bool:
        """Tests if the bit at the specified index `idx` is set (is 1).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`.

        Args:
            idx: The non-negative index of the bit to test (must be < `size`).

        Returns:
            True if the bit at `idx` is set, False otherwise.
        """
        _check_index_bounds["testing"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        return (self.data[w] & _bit_mask[self.WORD_DTYPE](idx)) != 0

    @always_inline
    fn clear(mut self):
        """Clear the BitVec.

        This sets the length to 0.
        """
        self._len = 0

    @always_inline
    fn zero_all(mut self):
        """Set all bits to zero."""
        memset_zero(self.data, _elts[self.WORD_DTYPE](self._len))

    @always_inline
    fn set_and_check(mut self, idx: UInt) -> Bool:
        """Set the bit at the given index. If the value was already set, return False,
        otherwise True.

        Args:
            idx: The index of the bit to set.

        Returns:
            False if the bit was already set, True if it was not.
        """
        _check_index_bounds["set"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        var mask = _bit_mask[self.WORD_DTYPE](idx)

        var ret = (self.data[w] & mask) == 0

        self.data[w] |= mask
        return ret

    @always_inline
    fn set(mut self, idx: UInt):
        """Set the bit at the given index to 1.

        Args:
            idx: The index of the bit to set.
        """
        _check_index_bounds["set"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        self.data[w] |= _bit_mask[self.WORD_DTYPE](idx)

    @always_inline
    fn clear_and_check(mut self, idx: UInt) -> Bool:
        """Clear the bit at the given index. If the value was already clear (0, False),
        return False, otherwise True.

        Args:
            idx: The index of the bit to clear.

        Returns:
            False if the bit was already clear, True if it was not.
        """
        _check_index_bounds["set"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        var mask = _bit_mask[self.WORD_DTYPE](idx)

        var ret = (self.data[w] & mask) != 0

        self.data[w] &= ~mask
        return ret

    @always_inline
    fn clear(mut self, idx: UInt):
        """Clear the bit at the given index (set to 0).

        Args:
            idx: The index of the bit to clear.
        """
        _check_index_bounds["clear"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        self.data[w] &= ~_bit_mask[self.WORD_DTYPE](idx)

    @always_inline
    fn toggle(mut self, idx: UInt):
        """Toggles (inverts) the bit at the specified index `idx`.

        Args:
            idx: The non-negative index of the bit to toggle (must be < len(BitVec)).
        """
        _check_index_bounds["toggling"](idx, self._len)
        var w = _word_index[self.WORD_DTYPE](idx)
        self.data[w] ^= _bit_mask[self.WORD_DTYPE](idx)

    @always_inline
    fn append(mut self, value: Bool):
        """Append an item to the end of the BitVec.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets to 1.
        """
        if self._len >= self.capacity():
            self._realloc(self._capacity * 2 | Int(self._capacity == 0))
        # N.B. incr first to avoid debug assert that checks length in set
        self._len += 1
        self[self._len - 1] = value

    @always_inline
    fn append_true(mut self):
        """Append a set bit to the end of the BitVec.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets to 1.
        """
        if self._len >= self.capacity():
            self._realloc(self._capacity * 2 | Int(self._capacity == 0))
        # N.B. incr first to avoid debug assert that checks length in set
        self._len += 1
        self.set(self._len - 1)

    @always_inline
    fn append_false(mut self):
        """Append a cleared bit to the end of the BitVec.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets to 1.
        """
        if self._len >= self.capacity():
            self._realloc(self._capacity * 2 | Int(self._capacity == 0))
        # N.B. incr first to avoid debug assert that checks length in set
        self._len += 1
        self.clear(self._len - 1)

    @always_inline
    fn pop_back(mut self) -> Bool:
        """Remove and return the last item in the BitVec."""
        debug_assert(
            not self.is_empty(), "Called `pop_back` on an empty BitVec"
        )
        var ret = self[self._len - 1]
        self._len -= 1
        return ret

    @always_inline
    fn _count_set_bits(read self, *, up_to: UInt) -> UInt:
        """Count the total number of set bits where index < up_to.

        Args:
            up_to: The index to count up to. This index is not included in the count.
        """
        # Plus one on the check here because this is an exclusive range
        _check_index_bounds["count_set_bits"](up_to, self._len + 1)
        alias width = simdwidthof[Scalar[Self.WORD_DTYPE]]()
        var total = 0

        @parameter
        @always_inline
        fn count[simd_width: Int](offset: Int):
            var vec = self.data.offset(offset).load[width=simd_width]()
            total += Int(pop_count(vec).reduce_add())

        var num_words = _elts[self.WORD_DTYPE](up_to)
        if num_words == 0:
            return 0

        vectorize[count, width](num_words - 1)

        # Now add in the last bits
        var bit_offset = up_to % self.WORD_DTYPE.bitwidth()
        if bit_offset != 0:
            var mask = (1 << bit_offset) - 1
            total += Int(pop_count(mask & self.data[num_words - 1]))
        else:
            # We count everything in the word
            total += Int(pop_count(self.data[num_words - 1]))
        return total

    @always_inline
    fn count_set_bits(read self) -> UInt:
        """Count the total number of set bits."""
        return self._count_set_bits(up_to=len(self))

    @always_inline
    fn rank(read self, bit_idx: UInt) -> UInt:
        """Count the total number of set bits up to (but not including) `bit_idx`.

        Args:
            bit_idx: Index to get the rank for.

        TODO: implement another struct that builds a rank/select index.
        # References:

        - https://rob-p.github.io/CMSC858D/static_files/presentations/CMSC858D-Lec08.pdf

        """
        return self._count_set_bits(up_to=bit_idx)

    @always_inline
    fn count_clear_bits(read self) -> UInt:
        """Count the total number of clear bits."""
        return len(self) - self.count_set_bits()

    # --------------------------------------------------------------------- #
    # Set operations
    # --------------------------------------------------------------------- #
    @always_inline
    @staticmethod
    fn _vectorize_apply[
        func: fn[simd_width: Int] (
            SIMD[Self.WORD_DTYPE, simd_width],
            SIMD[Self.WORD_DTYPE, simd_width],
        ) capturing -> SIMD[Self.WORD_DTYPE, simd_width],
        lhs_zero_out: Bool = False,
    ](read left: Self, read right: Self) -> Self:
        """Applies a vectorized binary operation between two BitVecs.

        This internal utility function optimizes set operations by processing
        multiple words in parallel using SIMD instructions when possible. It
        applies the provided function to corresponding words from both BitVecs
        and returns a new BitVec with the results.

        The vectorized operation is applied to each word in the BitVec but only
        if the number of words in the BitVecs is greater than or equal to the
        SIMD width.

        Parameters:
            func: A function that takes two SIMD vectors of Self.WORD_DTYPE
                values and returns a SIMD vector with the result of the operation.
                The function should implement the desired set operation (e.g.,
                union, intersection).
            lhs_zero_out: Decides how to handle the case where the lhs is longer than
                the rhs. If it is set to True, then the remainder of the lhs, will be
                cleared. If it is set to False, then the lhs bits will be left alone.

        Args:
            left: The first BitVec operand.
            right: The second BitVec operand.

        Returns:
            A new BitVec containing the result of applying the function to each
            corresponding pair of words from the input BitVecs.

        Notes:
            The length of left must be >= length of right.
        """
        alias width = simdwidthof[Self.WORD_DTYPE]()
        debug_assert(
            len(left) >= len(right), "Length of left must be >= length of right"
        )
        var res = Self(length=len(left), fill=False)

        # Define a vectorized operation that processes multiple words at once
        @parameter
        @always_inline
        fn _intersect[simd_width: Int](offset: Int):
            # Initialize SIMD vectors to hold multiple words from each bitset
            var left_vec = SIMD[Self.WORD_DTYPE, simd_width]()
            var right_vec = SIMD[Self.WORD_DTYPE, simd_width]()

            # Load a batch of words from both bitsets into SIMD vectors
            left_vec = left.data.offset(offset).load[width=simd_width]()
            right_vec = right.data.offset(offset).load[width=simd_width]()

            # Apply the provided operation (union, intersection, etc.) to the
            # vectors
            var result_vec = func(left_vec, right_vec)

            # Store the results back into the result bitset
            res.data.offset(offset).store[width=simd_width](result_vec)

        var lhs_len = _elts[Self.WORD_DTYPE](len(left))
        var rhs_len = _elts[Self.WORD_DTYPE](len(right))
        vectorize[_intersect, width](min(lhs_len, rhs_len))

        if lhs_len > rhs_len:
            var bit_offset = Self.WORD_DTYPE.bitwidth() - (
                len(right) % Self.WORD_DTYPE.bitwidth()
            )

            if bit_offset != 0:
                var word_idx = rhs_len - 1
                var mask = (1 << bit_offset) - 1

                @parameter
                if lhs_zero_out:
                    res.data[word_idx] &= mask  # clear high bits
                else:
                    # copy left's word and preserve the low bits
                    res.data[word_idx] = (res.data[word_idx] & ~mask) | (
                        left.data[word_idx] & mask
                    )

            var remaining_words = lhs_len - rhs_len
            if remaining_words > 0:

                @parameter
                if lhs_zero_out:
                    memset_zero(res.data.offset(rhs_len), remaining_words)
                else:
                    memcpy(
                        res.data.offset(rhs_len),
                        left.data.offset(rhs_len),
                        remaining_words,
                    )

        return res^

    fn union(self, other: Self) -> Self:
        """Returns a new bitset that is the union of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        |: 1 1 1 1 1 1 1 0
        ```

        Args:
            other: The bitset to union with.

        Returns:
            A new bitset containing all elements from both sets.
        """

        @parameter
        @always_inline
        fn _union[
            simd_width: Int
        ](
            left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ) -> SIMD[self.WORD_DTYPE, simd_width]:
            return left | right

        if len(self) >= len(other):
            return Self._vectorize_apply[_union, False](self, other)
        else:
            return Self._vectorize_apply[_union, False](other, self)

    fn intersection(self, other: Self) -> Self:
        """Returns a new bitset that is the intersection of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        &: 0 0 1 0 0 0 0 0
        ```

        Args:
            other: The bitset to intersect with.

        Returns:
            A new bitset containing only the elements present in both sets.
        """

        @parameter
        @always_inline
        fn _intersection[
            simd_width: Int
        ](
            left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ) -> SIMD[self.WORD_DTYPE, simd_width]:
            return left & right

        if len(self) >= len(other):
            return Self._vectorize_apply[_intersection, True](self, other)
        else:
            return Self._vectorize_apply[_intersection, True](other, self)

    fn difference(self, other: Self) -> Self:
        """Returns a new bitset that is the difference of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        -: 0 0 0 1 1 1 1 0
        ```

        Args:
            other: The bitset to subtract from `self`.

        Returns:
            A new bitset containing elements from `self` that are not in `other`.
        """

        @parameter
        @always_inline
        fn _difference[
            simd_width: Int
        ](
            left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ) -> SIMD[self.WORD_DTYPE, simd_width]:
            return left & ~right

        if len(self) >= len(other):
            return Self._vectorize_apply[_difference, False](self, other)
        else:
            return Self._vectorize_apply[_difference, False](other, self)

    # --------------------------------------------------------------------- #
    # Inplace Set operations
    # --------------------------------------------------------------------- #

    @always_inline
    @staticmethod
    fn _mut_vectorize_apply[
        func: fn[simd_width: Int] (
            mut SIMD[Self.WORD_DTYPE, simd_width],
            SIMD[Self.WORD_DTYPE, simd_width],
        ) capturing,
        lhs_zero_out: Bool = False,
    ](mut left: Self, right: Self):
        """Applies a vectorized binary operation between two BitVecs.

        This internal utility function optimizes set operations by processing
        multiple words in parallel using SIMD instructions when possible. It
        applies the provided function to corresponding words from both BitVecs
        and returns a new BitVec with the results.

        The vectorized operation is applied to each word in the BitVec but only
        if the number of words in the BitVecs is greater than or equal to the
        SIMD width.

        Parameters:
            func: A function that takes two SIMD vectors of Self.WORD_DTYPE
                values and returns a SIMD vector with the result of the operation.
                The function should implement the desired set operation (e.g.,
                union, intersection).
            lhs_zero_out: Decides how to handle the case where the lhs is longer than
                the rhs. If it is set to True, then the remainder of the lhs, will be
                cleared. If it is set to False, then the lhs bits will be left alone.

        Args:
            left: The first BitVec operand.
            right: The second BitVec operand.

        """
        alias width = simdwidthof[Self.WORD_DTYPE]()

        # Define a vectorized operation that processes multiple words at once
        @parameter
        @always_inline
        fn _intersect[simd_width: Int](offset: Int):
            # Initialize SIMD vectors to hold multiple words from each bitset
            var left_vec = SIMD[Self.WORD_DTYPE, simd_width]()
            var right_vec = SIMD[Self.WORD_DTYPE, simd_width]()

            # Load a batch of words from both bitsets into SIMD vectors
            left_vec = left.data.offset(offset).load[width=simd_width]()
            right_vec = right.data.offset(offset).load[width=simd_width]()

            # Apply the provided operation (union, intersection, etc.) to the
            # vectors
            func(left_vec, right_vec)

            # Store the results back into the result bitset
            left.data.offset(offset).store[width=simd_width](left_vec)

        var lhs_len = _elts[Self.WORD_DTYPE](len(left))
        var rhs_len = _elts[Self.WORD_DTYPE](len(right))
        if lhs_len < rhs_len:
            left.resize(len(right), False)
            lhs_len = rhs_len
        vectorize[_intersect, width](min(lhs_len, rhs_len))

        if lhs_len > rhs_len:
            var bit_offset = Self.WORD_DTYPE.bitwidth() - (
                len(right) % Self.WORD_DTYPE.bitwidth()
            )

            if bit_offset != 0:
                var word_idx = rhs_len - 1
                var mask = (1 << bit_offset) - 1

                @parameter
                if lhs_zero_out:
                    left.data[word_idx] &= mask  # clear high bits

            var remaining_words = lhs_len - rhs_len
            if remaining_words > 0:

                @parameter
                if lhs_zero_out:
                    memset_zero(left.data.offset(rhs_len), remaining_words)

    fn union_update(mut self, other: Self):
        """Modifies `self` to be the union of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        |= 1 1 1 1 1 1 1 0
        ```

        If `len(self)`  < `len(other)`, `self` will be resized to match
        the size of `other` by filling with `0`s

        Args:
            other: The bitset to union with.

        Notes:
            This retains `self`s length.
        """

        @parameter
        @always_inline
        fn _union[
            simd_width: Int
        ](
            mut left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ):
            left |= right

        return Self._mut_vectorize_apply[_union, False](self, other)

    fn intersection_update(mut self, other: Self):
        """Modifies `self` to be the intersection of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        &= 0 0 1 0 0 0 0 0
        ```

        If `len(self)`  < `len(other)`, `self` will be resized to match
        the size of `other` by filling with `0`s.

        Args:
            other: The bitset to intersect with.

        Notes:
            This retains `self`s length.
        """

        @parameter
        @always_inline
        fn _intersection[
            simd_width: Int
        ](
            mut left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ):
            left &= right

        return Self._mut_vectorize_apply[_intersection, True](self, other)

    fn difference_update(mut self, other: Self):
        """Modifies `self` to be the difference of `self` and `other`.

        ```
        A: 0 0 1 1 1 1 1 0
        B: 1 1 1 0 0
        -= 0 0 0 1 1 1 1 0
        ```

        If `len(self)`  < `len(other)`, `self` will be resized to match
        the size of `other` by filling with `0`s.

        Args:
            other: The bitset to subtract from `self`.

        Notes:
            This retains `self`s length.
        """

        @parameter
        @always_inline
        fn _difference[
            simd_width: Int
        ](
            mut left: SIMD[self.WORD_DTYPE, simd_width],
            right: SIMD[self.WORD_DTYPE, simd_width],
        ):
            left &= ~right

        return Self._mut_vectorize_apply[_difference, False](self, other)

    fn write_to[W: Writer](read self, mut writer: W):
        writer.write(
            "BitVec{length=", len(self), " ,words=", self.word_len(), "}\n\t"
        )
        for i in range(0, len(self)):
            writer.write(Int(self[i]))
        writer.write("\n")
