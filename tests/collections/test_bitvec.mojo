from testing import assert_equal, assert_true, assert_false

from ExtraMojo.collections.bitvec import BitVec


def test_bitvec_from_list_literal():
    var bv: BitVec = [True, True, False, False, False, True]
    assert_equal(len(bv), 6)
    assert_equal(bv.count_set_bits(), 3)


def test_bitvec_init_with_capacity():
    var bv = BitVec(capacity=1024)
    assert_equal(bv.capacity(), 1024)


def test_bitvec_capacity_alignment():
    var bv = BitVec(capacity=130)  # Should allocate ceil(130 / word_size)
    assert_true(bv.capacity() >= 130)
    assert_equal(len(bv), 0)


def test_bitvec_init_length_fill_false():
    var bv = BitVec(length=65, fill=False)
    assert_equal(len(bv), 65)

    # Ensure bits are cleared
    for i in range(0, bv.word_len()):
        assert_equal(bv.data[i], 0)


def test_bitvec_init_length_fill_true():
    var bv = BitVec(length=65, fill=True)
    assert_equal(len(bv), 65)

    for i in range(0, bv.word_len()):
        assert_equal(bv.data[i], ~0)

    var last_bits = 65 % bv.WORD_DTYPE.bitwidth()
    var mask = (1 << last_bits) - 1
    assert_equal(bv.data[bv._capacity - 1] & mask, mask)


def test_bitvec_resize_grow_fill_false():
    var bv = BitVec(length=10, fill=True)
    bv.resize(130, fill=False)
    assert_equal(len(bv), 130)

    # Check bits from original words
    var first_word = bv.data[0]
    var first_word_mask = (1 << 10) - 1
    assert_equal(first_word & first_word_mask, first_word_mask)

    # Check new words are cleared
    for i in range(1, bv.word_len()):
        assert_equal(bv.data[i], 0)


def test_bitvec_resize_grow_fill_true():
    var bv = BitVec(length=10, fill=False)
    bv.resize(130, fill=True)
    assert_equal(len(bv), 130)

    # Check original bits are zero
    assert_equal(bv.data[0] & ((1 << 10) - 1), 0)

    # Check that upper bits of first word are set
    var upper_mask = ~((1 << 10) - 1)
    assert_true((bv.data[0] & upper_mask) == upper_mask)

    # Check new words are set to 0xFFFFFFFF
    for i in range(1, bv.word_len()):
        assert_equal(bv.data[i], ~0)


def test_bitvec_resize_shrink():
    var bv = BitVec(length=130, fill=True)
    bv.resize(64, fill=False)  # should shrink and mask the last word
    assert_equal(len(bv), 64)

    var mask = Scalar[bv.WORD_DTYPE].MAX
    assert_equal(bv.data[0], mask)


def test_bitvec_shrink_to_same_size():
    var bv = BitVec(length=64, fill=True)
    bv.shrink(64)  # no-op
    assert_equal(len(bv), 64)
    assert_equal(bv.data[0], ~0)


def test_bitvec_clear():
    var bv = BitVec(length=514, fill=True)
    assert_true(bv.data[1] != 0)
    assert_true(len(bv) == 514)
    bv.clear()
    assert_true(bv.data[1] == 0)
    assert_true(len(bv) == 0)


def test_bitvec_empty():
    var bv = BitVec(length=1, fill=False)
    assert_true(Bool(bv))
    assert_false(bv.is_empty())


def test_bitvec_getitem():
    var bv = BitVec(length=600, fill=True)
    for i in range(0, 600):
        assert_true(bv[i])


def test_bitvec_setitem():
    var bv = BitVec(length=1200, fill=False)
    bv[600] = True
    assert_true(bv[600])
    bv[0] = True
    assert_true(bv[0])
    bv[1199] = True
    assert_true(bv[1199])


def test_bitvec_set():
    var bv = BitVec(length=1200, fill=False)
    bv.set(600)
    assert_true(bv[600])
    bv.set(0)
    assert_true(bv[0])
    bv.set(1199)
    assert_true(bv[1199])


def test_bitvec_clear_one():
    var bv = BitVec(length=1200, fill=True)
    bv.clear(600)
    assert_false(bv[600])
    bv.clear(0)
    assert_false(bv[0])
    bv.clear(1199)
    assert_false(bv[1199])


def test_bitvec_toggle():
    var bv = BitVec(length=1200, fill=True)
    bv.toggle(600)
    assert_false(bv[600])
    bv.toggle(0)
    assert_false(bv[0])
    bv.toggle(1199)
    assert_false(bv[1199])

    bv.toggle(600)
    assert_true(bv[600])
    bv.toggle(0)
    assert_true(bv[0])
    bv.toggle(1199)
    assert_true(bv[1199])


def test_bitvec_append():
    var bv = BitVec()
    for _i in range(0, 100):
        bv.append(True)

    assert_equal(len(bv), 100)
    for i in range(0, 100):
        assert_true(bv[i])

    bv.clear()
    assert_true(bv.is_empty())
    for _i in range(0, 1050):
        bv.append_true()

    assert_equal(len(bv), 1050)
    for i in range(0, 1050):
        assert_true(bv[i])

    bv = BitVec(length=800, fill=True)
    for _i in range(0, 200):
        bv.append_false()
    assert_equal(len(bv), 1000)

    for i in range(0, 800):
        assert_true(bv[i])
    for i in range(800, 1000):
        assert_false(bv[i])


def test_bitvec_pop_back():
    var bv = BitVec(length=1000, fill=True)
    for _i in range(0, 1000):
        assert_true(bv.pop_back())
    assert_true(bv.is_empty())


def test_bitvec_set_and_check():
    var bv = BitVec(length=1000, fill=True)
    assert_false(bv.set_and_check(512))
    bv.append(False)
    assert_true(bv.set_and_check(1000))


def test_bitvec_clear_and_check():
    var bv = BitVec(length=1000, fill=True)
    assert_true(bv.clear_and_check(512))
    bv.append(False)
    assert_false(bv.clear_and_check(1000))


def test_bitvec_union():
    # left size longer
    var bvA = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bvA.set(i)
    var bvB = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvB.set(i)
    var bvC = bvA | bvB
    assert_equal(bvC[499], True)
    assert_equal(len(bvC), 600)
    assert_equal(bvC.count_set_bits(), 6)

    # left size shorter
    var bvD = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bvD.set(i)
    var bvE = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvE.set(i)
    var bvF = bvE | bvD
    assert_equal(len(bvF), 600)
    assert_equal(bvF.count_set_bits(), 6)

    # Basic case
    var bv1 = BitVec(length=128, fill=False)
    bv1.set(1)
    bv1.set(2)
    bv1.set(3)

    var bv2 = BitVec(length=128, fill=False)
    bv2.set(3)
    bv2.set(4)
    bv2.set(5)

    var bv3 = bv1.union(bv2)

    assert_equal(len(bv3), len(bv2), msg="Union: Length remains the same.")
    assert_equal(bv3.count_set_bits(), 5, msg="Union: Basic case count")
    assert_true(bv3.test(1), msg="Union: Basic case bit 1")
    assert_true(bv3.test(2), msg="Union: Basic case bit 2")
    assert_true(bv3.test(3), msg="Union: Basic case bit 3")
    assert_true(bv3.test(4), msg="Union: Basic case bit 4")
    assert_true(bv3.test(5), msg="Union: Basic case bit 5")

    # Union with empty set
    var bv_empty = BitVec(length=128, fill=False)
    var bv4 = bv1.union(bv_empty)
    assert_equal(bv4.count_set_bits(), 3, msg="Union: With empty set count")
    assert_true(bv4.test(1), msg="Union: With empty set bit 1")
    assert_true(bv4.test(2), msg="Union: With empty set bit 2")
    assert_true(bv4.test(3), msg="Union: With empty set bit 3")
    assert_false(bv4.test(4), msg="Union: With empty set bit 4")

    var bv5 = bv_empty.union(bv1)
    assert_equal(
        bv5.count_set_bits(), 3, msg="Union: Empty with non-empty set count"
    )
    assert_true(bv5.test(1), msg="Union: Empty with non-empty set bit 1")
    assert_true(bv5.test(2), msg="Union: Empty with non-empty set bit 2")
    assert_true(bv5.test(3), msg="Union: Empty with non-empty set bit 3")

    # Union of identical sets
    var bv6 = bv1.union(bv1)
    assert_equal(bv6.count_set_bits(), 3, msg="Union: Identical sets count")
    assert_true(bv6.test(1), msg="Union: Identical sets bit 1")
    assert_true(bv6.test(2), msg="Union: Identical sets bit 2")
    assert_true(bv6.test(3), msg="Union: Identical sets bit 3")

    # Union of disjoint sets
    var bv7 = BitVec(length=128, fill=False)
    bv7.set(10)
    bv7.set(20)
    var bv8 = bv1.union(bv7)
    assert_equal(bv8.count_set_bits(), 5, msg="Union: Disjoint sets count")
    assert_true(bv8.test(1), msg="Union: Disjoint sets bit 1")
    assert_true(bv8.test(2), msg="Union: Disjoint sets bit 2")
    assert_true(bv8.test(3), msg="Union: Disjoint sets bit 3")
    assert_true(bv8.test(10), msg="Union: Disjoint sets bit 10")
    assert_true(bv8.test(20), msg="Union: Disjoint sets bit 20")

    # Union across word boundaries
    var bv9 = BitVec(length=128, fill=False)
    bv9.set(60)
    bv9.set(65)
    var bv10 = BitVec(length=128, fill=False)
    bv10.set(63)
    bv10.set(70)
    var bv11 = bv9.union(bv10)
    assert_equal(bv11.count_set_bits(), 4, msg="Union: Across words count")
    assert_true(bv11.test(60), msg="Union: Across words bit 60")
    assert_true(bv11.test(63), msg="Union: Across words bit 63")
    assert_true(bv11.test(65), msg="Union: Across words bit 65")
    assert_true(bv11.test(70), msg="Union: Across words bit 70")


def test_bitvec_intersection():
    # left size longer
    var bvA = BitVec(length=600, fill=False)
    for i in [0, 25, 550]:
        bvA.set(i)
    var bvB = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvB.set(i)
    var bvC = bvA & bvB
    assert_equal(len(bvC), 600)
    assert_equal(bvC.count_set_bits(), 1)

    # left size shorter
    var bvD = BitVec(length=600, fill=False)
    for i in [0, 25, 550]:
        bvD.set(i)
    var bvE = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvE.set(i)
    var bvF = bvE & bvD
    assert_equal(len(bvF), 600)
    assert_equal(bvF.count_set_bits(), 1)

    # Basic case
    var bv1 = BitVec(length=128, fill=False)
    bv1.set(1)
    bv1.set(2)
    bv1.set(3)

    var bv2 = BitVec(length=128, fill=False)
    bv2.set(3)
    bv2.set(4)
    bv2.set(5)

    var bv3 = bv1.intersection(bv2)
    assert_equal(bv3.count_set_bits(), 1, msg="Intersection: Basic case count")
    assert_true(bv3.test(3), msg="Intersection: Basic case bit 3")
    assert_false(bv3.test(1), msg="Intersection: Basic case bit 1")
    assert_false(bv3.test(2), msg="Intersection: Basic case bit 2")
    assert_false(bv3.test(4), msg="Intersection: Basic case bit 4")
    assert_false(bv3.test(5), msg="Intersection: Basic case bit 5")

    # Intersection with empty set
    var bv_empty = BitVec(length=128, fill=False)
    var bv4 = bv1.intersection(bv_empty)
    assert_equal(
        bv4.count_set_bits(), 0, msg="Intersection: With empty set count"
    )

    var bv5 = bv_empty.intersection(bv1)
    assert_equal(
        bv5.count_set_bits(),
        0,
        msg="Intersection: Empty with non-empty set count",
    )

    # Intersection of identical sets
    var bv6 = bv1.intersection(bv1)
    assert_equal(
        bv6.count_set_bits(), 3, msg="Intersection: Identical sets count"
    )
    assert_true(bv6.test(1), msg="Intersection: Identical sets bit 1")
    assert_true(bv6.test(2), msg="Intersection: Identical sets bit 2")
    assert_true(bv6.test(3), msg="Intersection: Identical sets bit 3")

    # Intersection of disjoint sets
    var bv7 = BitVec(length=128, fill=False)
    bv7.set(10)
    bv7.set(20)
    var bv8 = bv1.intersection(bv7)
    assert_equal(
        bv8.count_set_bits(), 0, msg="Intersection: Disjoint sets count"
    )

    # Intersection across word boundaries
    var bv9 = BitVec(length=128, fill=False)
    bv9.set(60)
    bv9.set(65)
    bv9.set(70)
    var bv10 = BitVec(length=128, fill=False)
    bv10.set(63)
    bv10.set(65)
    bv10.set(75)
    var bv11 = bv9.intersection(bv10)
    assert_equal(
        bv11.count_set_bits(), 1, msg="Intersection: Across words count"
    )
    assert_true(bv11.test(65), msg="Intersection: Across words bit 65")
    assert_false(bv11.test(60), msg="Intersection: Across words bit 60")
    assert_false(bv11.test(63), msg="Intersection: Across words bit 63")
    assert_false(bv11.test(70), msg="Intersection: Across words bit 70")
    assert_false(bv11.test(75), msg="Intersection: Across words bit 75")


def test_bitvec_difference():
    # left size longer
    var bvA = BitVec(length=600, fill=False)
    for i in [0, 25, 550]:
        bvA.set(i)
    var bvB = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvB.set(i)
    var bvC = bvA - bvB
    assert_equal(len(bvC), 600)
    assert_equal(bvC.count_set_bits(), 2)

    # left size shorter
    var bvD = BitVec(length=600, fill=False)
    for i in [0, 25, 550]:
        bvD.set(i)
    var bvE = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bvE.set(i)
    var bvF = bvE - bvD
    assert_equal(len(bvF), 600)
    assert_equal(bvF.count_set_bits(), 2)

    # Basic case (bv1 - bv2)
    var bv1 = BitVec(length=128, fill=False)
    bv1.set(1)
    bv1.set(2)
    bv1.set(3)

    var bv2 = BitVec(length=128, fill=False)
    bv2.set(3)
    bv2.set(4)
    bv2.set(5)

    var bv3 = bv1.difference(bv2)
    assert_equal(
        bv3.count_set_bits(), 2, msg="Difference: Basic case (bv1-bv2) count"
    )
    assert_true(bv3.test(1), msg="Difference: Basic case (bv1-bv2) bit 1")
    assert_true(bv3.test(2), msg="Difference: Basic case (bv1-bv2) bit 2")
    assert_false(bv3.test(3), msg="Difference: Basic case (bv1-bv2) bit 3")
    assert_false(bv3.test(4), msg="Difference: Basic case (bv1-bv2) bit 4")

    # Basic case (bv2 - bv1)
    var bv4 = bv2.difference(bv1)
    assert_equal(
        bv4.count_set_bits(), 2, msg="Difference: Basic case (bv2-bv1) count"
    )
    assert_true(bv4.test(4), msg="Difference: Basic case (bv2-bv1) bit 4")
    assert_true(bv4.test(5), msg="Difference: Basic case (bv2-bv1) bit 5")
    assert_false(bv4.test(1), msg="Difference: Basic case (bv2-bv1) bit 1")
    assert_false(bv4.test(3), msg="Difference: Basic case (bv2-bv1) bit 3")

    # Difference with empty set
    var bv_empty = BitVec(length=128, fill=False)
    var bv5 = bv1.difference(bv_empty)
    assert_equal(
        bv5.count_set_bits(), 3, msg="Difference: With empty set count"
    )
    assert_true(bv5.test(1), msg="Difference: With empty set bit 1")
    assert_true(bv5.test(2), msg="Difference: With empty set bit 2")
    assert_true(bv5.test(3), msg="Difference: With empty set bit 3")

    var bv6 = bv_empty.difference(bv1)
    assert_equal(
        bv6.count_set_bits(),
        0,
        msg="Difference: Empty with non-empty set count",
    )

    # Difference of identical sets
    var bv7 = bv1.difference(bv1)
    assert_equal(
        bv7.count_set_bits(), 0, msg="Difference: Identical sets count"
    )

    # Difference of disjoint sets
    var bv8 = BitVec(length=128, fill=False)
    bv8.set(10)
    bv8.set(20)
    var bv9 = bv1.difference(bv8)  # bv1 - bv8
    assert_equal(
        bv9.count_set_bits(), 3, msg="Difference: Disjoint sets (bv1-bv8) count"
    )
    assert_true(bv9.test(1), msg="Difference: Disjoint sets (bv1-bv8) bit 1")
    assert_true(bv9.test(2), msg="Difference: Disjoint sets (bv1-bv8) bit 2")
    assert_true(bv9.test(3), msg="Difference: Disjoint sets (bv1-bv8) bit 3")
    assert_false(bv9.test(10), msg="Difference: Disjoint sets (bv1-bv8) bit 10")

    var bv10 = bv8.difference(bv1)  # bv8 - bv1
    assert_equal(
        bv10.count_set_bits(),
        2,
        msg="Difference: Disjoint sets (bv8-bv1) count",
    )
    assert_true(bv10.test(10), msg="Difference: Disjoint sets (bv8-bv1) bit 10")
    assert_true(bv10.test(20), msg="Difference: Disjoint sets (bv8-bv1) bit 20")
    assert_false(bv10.test(1), msg="Difference: Disjoint sets (bv8-bv1) bit 1")

    # Difference across word boundaries
    var bv11 = BitVec(length=128, fill=False)
    bv11.set(60)
    bv11.set(65)
    bv11.set(70)
    var bv12 = BitVec(length=128, fill=False)
    bv12.set(63)
    bv12.set(65)
    bv12.set(75)
    var bv13 = bv11.difference(bv12)  # bv11 - bv12
    assert_equal(
        bv13.count_set_bits(),
        2,
        msg="Difference: Across words (bv11-bv12) count",
    )
    assert_true(
        bv13.test(60), msg="Difference: Across words (bv11-bv12) bit 60"
    )
    assert_true(
        bv13.test(70), msg="Difference: Across words (bv11-bv12) bit 70"
    )
    assert_false(
        bv13.test(63), msg="Difference: Across words (bv11-bv12) bit 63"
    )
    assert_false(
        bv13.test(65), msg="Difference: Across words (bv11-bv12) bit 65"
    )
    assert_false(
        bv13.test(75), msg="Difference: Across words (bv11-bv12) bit 75"
    )

    var bv14 = bv12.difference(bv11)  # bv12 - bv11
    assert_equal(
        bv14.count_set_bits(),
        2,
        msg="Difference: Across words (bv12-bv11) count",
    )
    assert_true(
        bv14.test(63), msg="Difference: Across words (bv12-bv11) bit 63"
    )
    assert_true(
        bv14.test(75), msg="Difference: Across words (bv12-bv11) bit 75"
    )
    assert_false(
        bv14.test(60), msg="Difference: Across words (bv12-bv11) bit 60"
    )
    assert_false(
        bv14.test(65), msg="Difference: Across words (bv12-bv11) bit 65"
    )
    assert_false(
        bv14.test(70), msg="Difference: Across words (bv12-bv11) bit 70"
    )


def test_bitvec_union_update():
    var bv1 = BitVec(length=12, fill=False)
    for i in [2, 4, 6, 8, 10, 11]:
        bv1.set(i)
    var bv2 = BitVec(length=10, fill=False)
    for i in [1, 3, 5, 7, 9]:
        bv2.set(i)
    bv1 |= bv2
    assert_equal(len(bv1), 12)
    assert_equal(bv1.count_set_bits(), 11)

    # LHS Longer
    var bv3 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv3.set(i)
    var bv4 = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bv4.set(i)
    bv3 |= bv4
    assert_equal(len(bv3), 600)
    assert_equal(bv3.count_set_bits(), 6)

    # RHS Longer
    var bv5 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv5.set(i)
    var bv6 = BitVec(length=500, fill=False)
    for i in [1, 25, 499]:
        bv6.set(i)
    bv6 |= bv5
    assert_equal(len(bv6), 600)
    assert_equal(bv6.count_set_bits(), 6)


def test_bitvec_intersection_update():
    var bv1 = BitVec(length=12, fill=False)
    for i in [2, 3, 6, 8, 10, 11]:
        bv1.set(i)
    var bv2 = BitVec(length=10, fill=False)
    for i in [1, 3, 5, 7, 9]:
        bv2.set(i)
    bv1 &= bv2
    assert_equal(len(bv1), 12)
    assert_equal(bv1.count_set_bits(), 1)

    # LHS Longer
    var bv3 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv3.set(i)
    var bv4 = BitVec(length=500, fill=False)
    for i in [1, 64, 499]:
        bv4.set(i)
    bv3 &= bv4
    assert_equal(len(bv3), 600)
    assert_equal(bv3.count_set_bits(), 1)

    # RHS Longer
    var bv5 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv5.set(i)
    var bv6 = BitVec(length=500, fill=False)
    for i in [1, 64, 499]:
        bv6.set(i)
    bv6 &= bv5
    assert_equal(len(bv6), 600)
    assert_equal(bv6.count_set_bits(), 1)


def test_bitvec_difference_update():
    var bv1 = BitVec(length=12, fill=False)
    for i in [2, 3, 6, 8, 10, 11]:
        bv1.set(i)
    var bv2 = BitVec(length=10, fill=False)
    for i in [1, 3, 5, 7, 9]:
        bv2.set(i)
    bv1 -= bv2
    assert_equal(len(bv1), 12)
    assert_equal(bv1.count_set_bits(), 5)

    # LHS Longer
    var bv3 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv3.set(i)
    var bv4 = BitVec(length=500, fill=False)
    for i in [1, 64, 499]:
        bv4.set(i)
    bv3 -= bv4
    assert_equal(len(bv3), 600)
    assert_equal(bv3.count_set_bits(), 2)

    # RHS Longer
    var bv5 = BitVec(length=600, fill=False)
    for i in [0, 64, 550]:
        bv5.set(i)
    var bv6 = BitVec(length=500, fill=False)
    for i in [1, 64, 499]:
        bv6.set(i)
    bv6 -= bv5
    assert_equal(len(bv6), 600)
    assert_equal(bv6.count_set_bits(), 2)


def test_bitvec_partial_union_preserves_tail():
    var a = BitVec(length=130, fill=False)
    a.set(0)
    a.set(129)  # tail bit

    var b = BitVec(length=125, fill=False)
    b.set(63)
    b.set(124)

    var c = a | b  # uses __or__, lhs_zero_out = False
    assert_equal(len(c), 130)
    assert_equal(c[0], True)
    assert_equal(c[63], True)
    assert_equal(c[124], True)
    assert_equal(c[129], True)  # tail preserved


def test_bitvec_union_update_preserves_tail():
    var a = BitVec(length=130, fill=False)
    a.set(5)
    a.set(129)

    var b = BitVec(length=64, fill=False)
    b.set(0)
    b.set(5)

    a.union_update(b)
    assert_equal(len(a), 130)
    assert_equal(a[0], True)
    assert_equal(a[5], True)
    assert_equal(a[129], True)  # tail preserved


def test_bitvec_full_overlap_union():
    var a = BitVec(length=64, fill=False)
    var b = BitVec(length=64, fill=False)
    b.set(7)

    var c = a | b
    assert_equal(len(c), 64)
    assert_equal(c[7], True)


def test_bitvec_adhoc():
    alias example = "As the quick brown fox jumped over the fence a moon was rising in the distance. Then the moon exploded. The End.".as_bytes()
    var periods = BitVec(length=len(example), fill=False)
    var spaces = BitVec(length=len(example), fill=False)
    var ts = BitVec(length=len(example), fill=False)
    var none = BitVec(length=len(example), fill=False)

    for i in range(0, len(example)):
        if example[i] == ord("."):
            periods.set(i)
        elif example[i] == ord(" "):
            spaces.set(i)
        elif example[i] == ord("T"):
            ts.set(i)
        else:
            none.set(i)

    var either = periods | spaces
    assert_equal(either.count_set_bits(), 24)
    either |= ts
    assert_equal(either.count_set_bits(), 26)
    assert_equal(none.count_set_bits() + either.count_set_bits(), len(example))
    assert_true(periods != spaces)

    var test = either & none
    assert_equal(test.count_set_bits(), 0)
    assert_equal(test.count_clear_bits(), len(example))

    test &= either
    assert_equal(test.count_set_bits(), 0)
    assert_equal(test.count_clear_bits(), len(example))

    either -= test
    assert_equal(either.count_set_bits(), 26)
    test = either - ts
    assert_equal(test.count_set_bits(), 24)


def test_bitvec_equal():
    var bv1 = BitVec(length=65, fill=False)
    var bv2 = BitVec(length=65, fill=False)
    bv1.set(64)
    bv2.set(64)

    assert_true(bv1 == bv2)

    var bv3 = BitVec(length=65, fill=False)
    var bv4 = BitVec(length=65, fill=False)
    bv3.set(64)

    assert_true(bv3 != bv4)


def main():
    test_bitvec_from_list_literal()
    test_bitvec_init_with_capacity()
    test_bitvec_capacity_alignment()
    test_bitvec_init_length_fill_false()
    test_bitvec_init_length_fill_true()
    test_bitvec_resize_grow_fill_false()
    test_bitvec_resize_grow_fill_true()
    test_bitvec_resize_shrink()
    test_bitvec_shrink_to_same_size()
    test_bitvec_clear()
    test_bitvec_empty()
    test_bitvec_getitem()
    test_bitvec_setitem()
    test_bitvec_set()
    test_bitvec_clear_one()
    test_bitvec_toggle()
    test_bitvec_append()
    test_bitvec_pop_back()
    test_bitvec_set_and_check()
    test_bitvec_clear_and_check()
    test_bitvec_union()
    test_bitvec_intersection()
    test_bitvec_difference()
    test_bitvec_union_update()
    test_bitvec_intersection_update()
    test_bitvec_difference_update()
    test_bitvec_partial_union_preserves_tail()
    test_bitvec_union_update_preserves_tail()
    test_bitvec_full_overlap_union()
    test_bitvec_adhoc()
