from ExtraMojo.collections.bbhash.hash import (
    hash64,
    level_hash,
    key_hash,
    hash,
    fnv1a,
)

from testing import assert_equal, assert_true, assert_false


@fieldwise_init
struct HashCases(Copyable, Movable):
    var key: String
    var want: UInt64

    @staticmethod
    fn hash64_cases() -> List[Self]:
        return [
            {"0", 7749275010220701263},
            {"0", 7749275010220701263},
            {"01", 10872400197931041294},
            {"012", 15346974464947871338},
            {"0123", 4981619345643336833},
            {"01234", 13689749573412676407},
            {"012345", 12368623765455266986},
            {"0123456", 12533847505615611486},
            {"01234567", 7278149109652352471},
            {"012345678", 732621452303077734},
            {"0123456789", 11460214490870572832},
            {"01234567890", 17034508007883094207},
            {"012345678901", 2613408247548219540},
            {"0123456789012", 9592538707932359806},
            {"01234567890123", 8026059100768657110},
            {"012345678901234", 10326712318572838968},
            {"0123456789012345", 8542297193267372205},
            {"01234567890123456", 9007936939456149167},
            {"012345678901234567", 4883792730719261022},
            {"0123456789012345678", 13174688786326369759},
            {"01234567890123456789", 4588891807043452244},
        ]

    @staticmethod
    fn fnv1a_cases() -> List[Self]:
        return [
            {"", 14695981039346656037},
            {"a", 12638187200555641996},
            {"hello", 11831194018420276491},
            {"world", 5717881983045765875},
            {"foo", 15902901984413996407},
            {"bar", 16101355973854746},
            {"foobar", 9625390261332436968},
            {
                "The quick brown fox jumps over the lazy dog",
                17580284887202820368,
            },
        ]


def test_hash64():
    for kase in HashCases.hash64_cases():
        var got = hash64(0, kase.key.as_bytes())
        assert_equal(got, kase.want, "hash64: Mismatch for key: " + kase.key)


def test_fnv1a():
    for kase in HashCases.fnv1a_cases():
        var got = fnv1a(kase.key.as_bytes())
        assert_equal(got, kase.want, "fnv1a: Mismatch for key: " + kase.key)


def test_hash():
    for lvl in range(0, 5):
        var lvl_hash = level_hash(UInt64(lvl))
        for key in range(0, 5):
            var slow_hash = hash(lvl, key)
            var fast_hash = key_hash(lvl_hash, key)
            assert_equal(
                slow_hash,
                fast_hash,
                String("hash({}, {}) != key_hahs({}, {})").format(
                    lvl, key, lvl_hash, key
                ),
            )


def main():
    test_hash64()
    test_fnv1a()
    test_hash()
