"""Provides a BBHash implementation.

## Examples

```mojo
from testing import assert_true, assert_false
from ExtraMojo.collections.bbhash.bbhash import BBHash

var keys: List[String] = ["fox", "cat", "dog", "mouse", "frog"]
var bbset = BBHash(keys^, gamma=1.0)
assert_true(bbset.find(String("fox")))
assert_false(bbset.find(String("muffin")))
```

# References:

- https://github.com/relab/bbhash
- https://arxiv.org/abs/1702.03154
"""

from hashlib.hash import hash
from os import abort

from ExtraMojo.collections.bitvec import BitVec, _word_index, _bit_mask, _elts
from ExtraMojo.collections.bbhash.hash import (
    _level_hash,
    _key_hash,
    _level_key_hash,
)


alias MAX_ITERS = 100


@fieldwise_init
struct _BCVec(Writable):
    """Combined bit and collision vector."""

    var v: BitVec
    var c: BitVec

    fn __init__(out self, *, length: UInt):
        """Create with the given number of bits."""
        # Expand out to word boundary, no reason not to
        var full_len = (
            _elts[BitVec.WORD_DTYPE](length) * BitVec.WORD_DTYPE.bitwidth()
        )
        self.v = BitVec(length=full_len, fill=False)
        self.c = BitVec(length=full_len, fill=False)

    fn update(mut self, h: UInt64):
        """`update` sets the bit for the given hash h, and records a collision if
        the bit was already set. The bit position is determined by h modulo the
        size of the vector.
        """
        # TODO arguably fastmod should be fine here?
        # var x = fastmod(h, len(self.v))
        var x = h % len(self.v)
        var w = _word_index[BitVec.WORD_DTYPE](UInt(x))
        var mask = _bit_mask[BitVec.WORD_DTYPE](UInt(x))

        if self.v.data[w] & mask != 0:
            # Found one or more collisions at index; update collision vector
            self.c.data[w] |= mask
            return
        # No collisions at index; set bit
        self.v.data[w] |= mask

    fn unset_collision(mut self, h: UInt64) -> Bool:
        var x = h % len(self.v)
        var w = _word_index[BitVec.WORD_DTYPE](UInt(x))
        var mask = _bit_mask[BitVec.WORD_DTYPE](UInt(x))
        if self.c.data[w] & mask != 0:
            # found collision at index i; unset bit
            self.v.data[w] &= ~mask
            return True
        # No collisions at index i
        return False

    fn next_level(mut self, new_size: Int):
        """Setup for the next level, resize and set to zero."""
        var full_len = (
            _elts[BitVec.WORD_DTYPE](new_size) * BitVec.WORD_DTYPE.bitwidth()
        )
        self.c.resize(full_len, fill=False)
        self.c.zero_all()
        self.v.resize(full_len, fill=False)
        self.v.zero_all()

    fn write_to[W: Writer](read self, mut writer: W):
        writer.write("bitvec:    ", self.v)
        writer.write("collisions:", self.c)


struct BBHash:
    var bits: List[BitVec]
    var ranks: List[UInt64]
    var reverse_map: List[UInt64]

    fn __init__[
        K: Copyable & Movable & Hashable
    ](out self, owned keys: List[K], *, gamma: Float64 = 1.0):
        """Create a `BBHash`.

        Args:
            keys: The keys to create a minimal perfect hash for.
            gamma: Tunable param for optimizing between storage size and creation/lookup speed.
                1.0 will give the best size, larger will result in faster lookups and creation.
                See publication for details.
        """
        self.bits = []
        self.ranks = []
        self.reverse_map = []
        self._compute(keys, gamma)

    # TODO: use wyhash instead of builtin hash / fnv1a, compare them all
    # TODO: switch to fastmod from lemiere?
    # TODO: add the parallel version on the bitvec of atomics
    fn _compute[
        K: Copyable & Movable & Hashable
    ](mut self, owned keys: List[K], owned gamma: Float64):
        """Compute the minimal perfect hash function.

        Args:
            keys: The hashable keys.
            gamma: The gamma factor that can be between >= 1.0
                tuning data structure size, vs lookup and construction perf.

        Notes:
            - If a minimal perfect hash can't be created within MAX_ITERS this will abort.
        """
        # TODO: log when gamma < 1.0
        gamma = max(gamma, 1.0)
        var size = len(keys)
        var redo = List[K](
            capacity=size // 2
        )  # heuristic: only 1/2 of the keys will collide

        # bit vectors for current level: A and C in the paper
        level_vec = _BCVec(length=UInt(Int(gamma * size)))

        # loop exits when there are no more keys to re-hash
        var lvl = 0
        while True:
            # precompute the level hash to speed up the key hashing
            var lvl_hash = _level_hash(UInt64(lvl))

            # find colliding keys and possible bit vector positions for non-colliding keys
            for k in keys:
                var h = _key_hash(lvl_hash, hash(k))
                # update the bit and collision vectors for the current level
                level_vec.update(h)

            # remove bit vector position assignments for colliding keys and add them to the redo set
            # TODO: this look feels like it could be faster?
            for i in range(0, len(keys)):
                var h = _key_hash(lvl_hash, hash(keys[i]))
                # Unset the bit vec position for the key if there was a collision
                if level_vec.unset_collision(h):
                    # Add the key to redos
                    redo.append(keys[i])

            # save the current bit vector for the current level
            self.bits.append(level_vec.v.copy())

            size = len(redo)
            if size == 0:
                break

            # move to the next level and compute the set of keys to re-hash (that had collisions)
            swap(keys, redo)
            redo.clear()
            level_vec.next_level(UInt(Int(gamma * size)))

            if lvl > MAX_ITERS:
                abort("Unable to find max mph after " + String(lvl) + " tries")
            lvl += 1
        self._compute_level_ranks()

    fn _compute_level_ranks(mut self):
        """Computes the total rank of each level.

        The total rank is the rank for all levels up to and including the current level.
        """
        # init to rank 1, since 0 idx is reserved for not-found
        var rank: UInt64 = 1
        self.ranks = List[UInt64](length=len(self.bits), fill=0)
        for i in range(0, len(self.bits)):
            self.ranks[i] = rank
            rank += self.bits[i].count_set_bits()

    fn find[K: Hashable](read self, key: K) -> UInt64:
        """Find returns a unique index representing the key in the minimal hash set.

        The return value is meaningful ONLY for keys in the original key set
        (provided at the time of construction of the minimal hash set).

        If the key is in the original key set, the return value is guaranteed to be
        in the range [1, len(keys)].

        If the key is not in the original key set, two things can happen:
        1. The return value is 0, representing that the key was not in the original key set.
        2. The return value is in the expected range [1, len(keys)], but is a false positive.

        Args:
            key: The hashable key to check membership for.

        Returns:
            The unique index of the key, or 0 if it isn't in the set.
        """
        for lvl in range(0, len(self.bits)):
            var i = _level_key_hash(UInt64(lvl), hash(key)) % len(
                self.bits[lvl]
            )
            if self.bits[lvl].test(UInt(i)):
                return self.ranks[lvl] + UInt64(self.bits[lvl].rank(UInt(i)))
        return 0
