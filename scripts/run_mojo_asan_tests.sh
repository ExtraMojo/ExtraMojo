#!/usr/bin/env bash
set -euo pipefail

target_features=()
if [[ -n "${MOJO_TARGET_FEATURES:-}" ]]; then
    target_features=(--target-features "$MOJO_TARGET_FEATURES")
fi

test_bin="${TMPDIR:-/tmp}/extramojo_test_bin"

while IFS= read -r -d "" test_file; do
    pixi run mojo build "${target_features[@]}" \
        --sanitize address \
        -debug-level=line-tables \
        -I . \
        -D ASSERT=all \
        "$test_file" \
        -o "$test_bin"
    "$test_bin"
done < <(find ./tests -name "test_*.mojo" -print0)
