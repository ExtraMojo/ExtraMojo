# ⚡ extramojo ⚡

Extra functionality to extend the Mojo std lib.

## Documentation and Examples

[ExtraMojo Docs](https://extramojo.github.io/ExtraMojo/)

Also see any of the tests in the `test_*.mojo` files.

## Install / Usage

This checkout targets **Mojo 1.2.0**. Use a source checkout or a published
package built for the same compiler version.

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
mojo = "=1.2.0"
```

then run:

```bash
pixi add extramojo
```

To build and install a local checkout instead, use Pixi 0.80 or newer and add
the following to your consumer's manifest (keeping the stable Modular and
conda-forge channels above):

```toml
[workspace]
preview = ["pixi-build"]
# ...channels and platforms...

[dependencies]
mojo = "=1.2.0"
extramojo = { path = "../ExtraMojo" }
```

Run `pixi install`, then `pixi run mojo run my_example_file.mojo`. No source
include path is needed: `pixi-build-mojo` installs the compiled library into
the environment's `lib/mojo` directory. The package pins `mojo-compiler` to
1.2.0 for build, host, and runtime compatibility. ExtraMojo remains self-contained;
its buffered I/O and byte-string APIs do not require `mojo-io` or `mojo-bstr`.


## Notable Functionality

- Buffered Reading and writing for files
- CLI opts parser
- For bytestrings (`List/Span[UInt8]`), SIMD version of: memchr, to_uppercase, to_lowercase, SplitIterator
- BBHash
- BitVec
- `saturating_add` and `saturating_sub` for integral types
- reservoir sampling

See [docs for details](https://extramojo.github.io/ExtraMojo/)


## Line callbacks

`for_each_line(path, callback, buf_size=...)` takes a runtime callback.
Define callbacks with explicit
capture lists (`{imm ...}` / `{mut ...}`), without `@parameter`.

## Tasks

```bash
pixi install --locked
pixi run build   # Precompile the library for local development
pixi run t       # All existing tests; also used by `hat test`
pixi run t-asan  # AddressSanitizer tests
pixi run rbuild  # Build a Conda archive in dist/ with pixi-build-mojo
pixi run rtest   # Build, install into a temporary consumer, run all tests
```

The consumer test copies only `tests/` outside the checkout, installs the exact
archive from `dist/`, and runs without `-I .` or a source import path. CI runs
source and installed-package tests on Linux x86-64, Linux ARM64, and macOS ARM64,
plus AddressSanitizer on Linux x86-64. Neither packaging task uploads a release.

`pixi.toml` defines the package build. `LICENSE` includes both license texts;
see also `LICENSE-MIT`, `UNLICENSE`, and source attribution notices.

## Attribution

- File and Tensor code includes work from [MojoFastTrim](https://github.com/MoSafi2/MojoFastTrim/tree/restructed), maintained as [BlazeSeq](https://github.com/MoSafi2/BlazeSeq).
