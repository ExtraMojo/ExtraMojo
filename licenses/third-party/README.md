# Third-party code in the decimal byte parser

ExtraMojo's original code remains available under its existing MIT / Unlicense
terms. The adapted portions of `extramojo/bstr/parse.mojo` retain the following
upstream licenses; the package metadata describes their combined requirements.

## Zig

- Source: [Zig 0.16.0](https://codeberg.org/ziglang/zig/src/tag/0.16.0/lib/std/fmt/parse_float),
  specifically `parse.zig` (`parse8Digits` and digit scanning), `common.zig`
  (`isEightDigits`), and `convert_eisel_lemire.zig` (refinement and rounding checks).
- Copyright (c) Zig contributors.
- License: MIT (Expat), reproduced unchanged in [zig.txt](zig.txt).
- Changes: translated the relevant operations to Mojo, used byte spans, and added
  checked significand accumulation and a restricted decimal grammar.
- Upstream credits Johnny Lee's
  [Fast numeric string to int](https://johnnylee-sde.github.io/Fast-numeric-string-to-int/)
  for eight-digit conversion, and Daniel Lemire's
  [Number Parsing at a Gigabyte per Second](https://arxiv.org/abs/2101.11408)
  for numeric conversion.

## Modular

- Source: [`parsing_floats.mojo` at `max/v26.5.0`](https://github.com/modular/modular/blob/b4497b7ce9ba96331c72c637ad41b44bab374f33/mojo/stdlib/std/collections/string/_parsing_numbers/parsing_floats.mojo)
  (Mojo 1.0.0), specifically the Lemire conversion and product-refinement logic.
- Copyright (c) 2026, Modular Inc. All rights reserved.
- License: Apache-2.0 WITH LLVM-exception, reproduced unchanged from that revision's
  root `LICENSE` in [modular.txt](modular.txt).
- Changes: combined the relevant conversion steps, corrected the product-refinement
  mask and tie detection, and reused stdlib tables and numeric helpers by import.
- Upstream credits Daniel Lemire's paper above, Noble Mushtak and Daniel Lemire's
  [Fast Number Parsing Without Fallback](https://arxiv.org/abs/2212.06644), and
  [Carl Verret's csFastFloat](https://github.com/CarlVerret/csFastFloat) as references.

The LLVM exception is part of Modular's source license, not an added LLVM library
dependency. These notices concern the adapted parser source, not redistribution
of the Mojo compiler or SDK. The package recipe includes this provenance note
and both upstream license texts.
