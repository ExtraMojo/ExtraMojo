#!/usr/bin/env bash
set -euo pipefail

target_features=()
if [[ -n "${MOJO_TARGET_FEATURES:-}" ]]; then
    target_features=(--target-features "$MOJO_TARGET_FEATURES")
fi

while IFS= read -r -d "" test_file; do
    pixi run mojo run "${target_features[@]}" \
        -debug-level=line-tables \
        -I . \
        -D ASSERT=all \
        "$test_file"
done < <(find ./tests -name "test_*.mojo" -print0)
