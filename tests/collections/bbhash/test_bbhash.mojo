from collections import Set
from hashlib.hash import hash
from testing import assert_equal, assert_true, assert_false, TestSuite

from extramojo.collections.bbhash.bbhash import BBHash


def test_bbhash_simple():
    var some_star_wars_characters: List[String] = [
        "4-LOM",
        "Admiral Thrawn",
        "Senator Bail Organa",
        "Ben Skywalker",
        "Bib Fortuna",
        "Boba Fett",
        "C-3PO",
        "Cad Bane",
        "Cade Skywalker",
        "Captain Rex",
        "Chewbacca",
        "Clone Commander Cody",
        "Darth Vader",
        "General Grievous",
        "General Veers",
        "Greedo",
        "Han Solo",
        "IG 88",
        "Jabba The Hutt",
        "Luke Skywalker",
        "Mara Jade",
        "Mission Vao",
        "Obi-Wan Kenobi",
        "Princess Leia",
        "PROXY",
        "Qui-Gon Jinn",
        "R2-D2",
        "Revan",
        "Wedge Antilles",
        "Yoda",
    ]
    var gamma_values: List[Float64] = [1.0, 1.5, 2.0]

    for g in gamma_values:
        var keys = some_star_wars_characters.copy()
        var bb = BBHash(keys^, gamma=g)
        var key_set: Set[UInt] = {}

        for i in range(0, len(some_star_wars_characters)):
            var idx = bb.find(some_star_wars_characters[i])
            assert_true(
                idx != 0,
                "can't find key in set: " + some_star_wars_characters[i],
            )
            assert_true(
                idx <= len(some_star_wars_characters),
                "key mapping out of bounds",
            )
            assert_true(
                UInt(idx) not in key_set,
                String("idx: {} ({}), was found in set already").format(
                    idx, some_star_wars_characters[i]
                ),
            )
            key_set.add(UInt(idx))


def test_bbhash_example():
    var keys: List[String] = ["fox", "cat", "dog", "mouse", "frog"]
    var bbset = BBHash(keys^, gamma=1.0)
    assert_true(bbset.find(String("fox")))
    assert_false(bbset.find(String("muffin")))


def test_bbhash_revmap():
    var keys: List[String] = ["fox", "cat", "dog", "mouse", "frog"]
    var bbset = BBHash[True](keys^, gamma=1.0)
    var idx = bbset.find(String("fox"))
    var key_hash = bbset.key(idx)
    assert_true(key_hash)
    assert_equal(key_hash.value(), hash(String("fox")))

    idx = bbset.find(String("muffin"))
    key_hash = bbset.key(idx)
    assert_false(key_hash)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()


# def main():
#     test_bbhash_simple()
#     test_bbhash_example()
#     test_bbhash_revmap()
