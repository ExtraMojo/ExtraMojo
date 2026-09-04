# ⚡ extramojo ⚡

Extra functionality to extend the Mojo std lib.

## Documentation and Examples

[ExtraMojo Docs](https://extramojo.github.io/ExtraMojo/)

Also see any of the tests in the `test_*.mojo` files.

## Install / Usage

Add `https://repo.prefix.dev/modular-community` to your project channels:

```
# mojoproject.toml or pixi.toml

[project]
channels = ["conda-forge", "https://conda.modular.com/max", "https://repo.prefix.dev/modular-community"]
description = "Add a short description here"
name = "my-mojo-project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
mojo = "=1.0.0"
```

then run:

```bash
pixi add extramojo
```

Or directly by following these instructions.

See docs for [numojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/tree/v0.3?tab=readme-ov-file#how-to-install) and just do that for this package until Mojo has true package / library support.

tl;dr;

In your project `mojo run -I "../extramojo" my_example_file.mojo`.
Note the bit about how to add this project to your LSP so things resolve in VSCode.


## Notable Functionality

- Buffered Reading and writing for files
- CLI opts parser
- For bytestrings (`List/Span[UInt8]`), SIMD version of: memchr, to_uppercase, to_lowercase, SplitIterator
- Float parsing directly from byte spans: `extramojo.bstr.parse.parse_float`
- BBHash
- BitVec
- `saturating_add` and `saturating_sub` for integral types
- reservoir sampling

See [docs for details](https://extramojo.github.io/ExtraMojo/)

### Parsing floating-point byte strings

```mojo
from extramojo.bstr.parse import parse_float

var value = parse_float("-12.5e2".as_bytes())
var hexadecimal = parse_float("0x1.8p+1".as_bytes())
```

The parser accepts Zig 0.16's `Float64` input syntax: signed decimals with
`e`/`E` exponents, `0x`/`0X` hexadecimal numbers with optional `p`/`P` exponents,
underscores between digits, and case-insensitive `nan`, `inf`, and `infinity`.
It consumes the entire input, rejects whitespace and malformed syntax, and
handles arbitrarily long significands and exponent spellings. Rounding is
nearest, ties to even; overflow and underflow produce signed infinity and zero.

Ordinary decimals use the eight-digit fast path. Ambiguous long decimals use
an exact fallback with bounded temporary integer storage. `parse_decimal` is
retained as a compatibility alias and accepts the same expanded syntax.


## Tasks

```
pixi run t
```

## Attribution

- Much of the first draft of the File and Tensor code was taken from [here](https://github.com/MoSafi2/MojoFastTrim/tree/restructed), which has now moved [here](https://github.com/MoSafi2/BlazeSeq).
- The byte float parser adapts digit scanning and rounding checks from Zig (MIT;
  [license](licenses/third-party/zig.txt)) and numeric conversion from Modular's Mojo standard
  library (Apache-2.0 with LLVM exceptions; [license](licenses/third-party/modular.txt)).
  See the [provenance notes](licenses/third-party/README.md) for source versions and changes.
