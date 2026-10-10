# ⚡ extramojo ⚡

Extra functionality to extend the Mojo std lib.

## Documentation and Examples

[ExtraMojo Docs](https://extramojo.github.io/ExtraMojo/)

Also see any of the tests in the `test_*.mojo` files.

## Install / Usage

ExtraMojo 0.24.0 targets **Mojo 1.1.0**.

Add `https://repo.prefix.dev/modular-community` to your project channels:

```
# pixi.toml

[workspace]
channels = ["conda-forge", "https://conda.modular.com/max", "https://repo.prefix.dev/modular-community"]
description = "Add a short description here"
name = "my-mojo-project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
mojo = "=1.1.0"
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
- BBHash
- BitVec
- `saturating_add` and `saturating_sub` for integral types
- reservoir sampling

See [docs for details](https://extramojo.github.io/ExtraMojo/)


## Migrating to 0.24.0

`for_each_line` now takes its callback as a runtime argument:
`for_each_line(path, callback, buf_size=...)`, rather than
`for_each_line[callback](path, buf_size=...)`. Define callbacks with explicit
capture lists (`{imm ...}` / `{mut ...}`), without `@parameter`.

## Tasks

```
pixi run t
```

## Attribution

- Much of the first draft of the File and Tensor code was taken from [here](https://github.com/MoSafi2/MojoFastTrim/tree/restructed), which has now moved [here](https://github.com/MoSafi2/BlazeSeq).
