#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYO3_PYTHON="${PYO3_PYTHON:-python3}"
case "$(uname -s)" in
    Darwin) built_extension=target/release/lib_rs.dylib ;;
    Linux) built_extension=target/release/lib_rs.so ;;
    *) echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac
cargo build --release --no-default-features --features multiple-pymethods,lfm2,pyo3/extension-module
installed_extension=$("$PYO3_PYTHON" - <<'PY'
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader, FileFinder
from pathlib import Path

directory = Path("vllm_vulkan").resolve()
finder = FileFinder(str(directory), (ExtensionFileLoader, EXTENSION_SUFFIXES))
spec = finder.find_spec("_rs")
print(spec.origin if spec else directory / "_rs.so")
PY
)
staged_extension=$(mktemp "${installed_extension}.XXXXXX")
trap 'rm -f -- "$staged_extension"' EXIT
cp -p "$built_extension" "$staged_extension"
"$PYO3_PYTHON" - "$staged_extension" <<'PY'
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
loader = importlib.machinery.ExtensionFileLoader("vllm_vulkan._rs", str(path))
spec = importlib.util.spec_from_file_location(loader.name, path, loader=loader)
if spec is None:
    raise ImportError(f"Cannot load staged extension: {path}")
module = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = module
loader.exec_module(module)
from vllm_vulkan import _rs

assert hasattr(_rs, "Lfm2Model"), "staged extension is missing Lfm2Model"
print(_rs.enumerate_devices())
PY
mv -- "$staged_extension" "$installed_extension"
