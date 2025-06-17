from ExtraMojo.collections.bitvec import BitVec
from ExtraMojo.collections.bbhash.hash import level_hash, key_hash, fnv1a

# TODO: Call shrink_to_fit?


trait Spanable(Copyable, Movable):
    fn as_span(ref self) -> Span[UInt8, __origin_of(self)]:
        # will need rebind like
        # return rebind[Span[UInt8, __origin_of(self)]](
        #     self.seq.buffer_to_search()
        # )
        ...


@fieldwise_init
struct BCVec:
    """Combined bit and collision vector."""

    var v: BitVec
    var c: BitVec

    fn __init__(out self, *, length: UInt):
        """Create with the given number of bits."""
        self.v = BitVec(length=length, fill=False)
        self.c = BitVec(length=length, fill=False)


struct BBHash:
    var bits: List[BitVec]
    var ranks: List[UInt64]
    var reverse_map: List[UInt64]

    fn __init__(out self):
        self.bits = []
        self.ranks = []
        self.reverse_map = []

    fn compute[S: Spanable](read self, keys: List[S], gamma: Float64):
        var size = len(keys)
        var redo = List[UInt64](
            capacity=size // 2
        )  # heuristic: only 1/2 of the keys will collide

        # bit vectors for current level: A and C in the paper
        level_vec = BCVec(length=UInt(gamma * size))

        # loop exits when there are no more keys to re-hash
        var lvl = 0
        while True:
            # precompute the level hash to speed up the key hashing
            var lvl_hash = level_hash(UInt64(lvl))

            # find colliding keys and possible bit vector positions for non-colliding keys

            lvl += 1
