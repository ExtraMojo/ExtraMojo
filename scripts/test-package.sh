#!/usr/bin/env bash
# Run the existing suite against an installed archive, never the source tree.
set -euo pipefail

root=$(pwd)
if [ "$#" -eq 0 ]; then
    echo "Usage: bash scripts/test-package.sh dist/extramojo-*.conda" >&2
    exit 1
fi

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) platform=linux-64 ;;
    Linux-aarch64) platform=linux-aarch64 ;;
    Darwin-arm64) platform=osx-arm64 ;;
    *) echo "Unsupported test platform" >&2; exit 1 ;;
esac

test_args=(-D ASSERT=all)
if [ "$platform" = linux-64 ]; then
    # Match the existing package tests on hosts without AVX-512.
    test_args+=(--target-features=-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512bitalg,-avx512vpopcntdq,-avx512fp16,-avx512bf16)
fi

consumer=$(mktemp -d)
trap 'rm -rf "$consumer"' EXIT

for package in "$@"; do
    test -f "$package"
    case "$package" in
        /*) ;;
        *) package="$root/$package" ;;
    esac
    if command -v sha256sum >/dev/null; then
        digest=$(sha256sum "$package" | cut -d ' ' -f 1)
    else
        digest=$(shasum -a 256 "$package" | cut -d ' ' -f 1)
    fi
    # Each archive gets its own environment and only the test sources.
    work=$(mktemp -d "$consumer/package.XXXXXX")
    cp -R "$root/tests" "$work/tests"
    cat > "$work/pixi.toml" <<EOF
[workspace]
name = "extramojo-consumer"
channels = ["https://prefix.dev/conda-forge", "https://conda.modular.com/max"]
platforms = ["$platform"]

[dependencies]
mojo = "=1.2.0"
extramojo = { url = "file://$package", sha256 = "$digest" }
EOF
    (
        cd "$work"
        # Do not inherit the source workspace's import path or active environment.
        unset MOJO_PATH MODULAR_MOJO_IMPORT_PATH PIXI_PROJECT_MANIFEST CONDA_PREFIX
        # Pixi 0.80 can reuse extracted same-version packages despite a new digest.
        export PIXI_CACHE_DIR="$work/cache"
        pixi install --manifest-path "$work/pixi.toml"
        pixi run --manifest-path "$work/pixi.toml" mojo --version
        count=0
        while IFS= read -r test; do
            echo "Installed-package test: $test"
            pixi run --manifest-path "$work/pixi.toml" mojo run \
                "${test_args[@]}" "$test"
            count=$((count + 1))
        done < <(find tests -name 'test_*.mojo' | sort)
        test "$count" -gt 0
        echo "Passed $count installed-package test files: $package"
    )
done
