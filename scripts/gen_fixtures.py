# Generates marshal fixtures under REAL Python 2.7. Run with:
#   python2 scripts/gen_fixtures.py
# Writes one .bin per value into tests/fixtures/. The Rust tests parse these
# back, so CI needs no Python interpreter -- the bytes are checked in.
import marshal
import os
import sys

if sys.version_info[0] != 2:
    sys.exit("must run under Python 2.7 (real marshal format)")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "tests", "fixtures")

# A code object whose layout matches what read.rs decodes.
CODE = compile("x = 1\nprint(x)\n", "<fixture>", "exec")

FIXTURES = [
    ("none", None),
    ("bool_true", True),
    ("bool_false", False),
    ("int", 7),
    ("int_neg", -7),
    ("int_big", 2 ** 100),
    ("int_big_neg", -(2 ** 100)),
    ("float", 1.5),
    ("float_neg", -1.5),
    ("complex", complex(2.0, -3.0)),
    ("str_bytes", b"Andr\xe8 Previn"),
    ("unicode", u"Andr\xe8 Previn"),
    ("unicode_empty", u""),
    ("tuple", (1, 2, u"abc")),
    ("tuple_empty", ()),
    ("list", [True, None, 3]),
    ("dict", {u"a": 1, u"b": 2}),
    ("nested", (1, [2, 3], {u"k": (4, 5)})),
    ("set", set([1, 2, 3])),
    ("frozenset", frozenset([u"a", u"b"])),
    ("dict_tuple_key", {(u"a", u"b"): u"c"}),
    ("code", CODE),
    # Two code objects sharing interned strings: exercises the reader's
    # FLAG_REF / TYPE_REF backreference table.
    ("code_tuple", (CODE, CODE)),
]


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    for name, value in FIXTURES:
        path = os.path.join(OUT, name + ".bin")
        with open(path, "wb") as f:
            f.write(marshal.dumps(value))
    print("wrote %d fixtures to %s" % (len(FIXTURES), OUT))


if __name__ == "__main__":
    main()
