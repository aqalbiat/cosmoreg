#!/usr/bin/env bash
# A self-contained XeLaTeX build; run this script through bash inside WSL.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v xelatex >/dev/null
command -v python3 >/dev/null
mkdir -p build
for pass in 1 2 3 4; do
    echo "XeLaTeX pass ${pass}"
    if ! xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
        -file-line-error report.tex > "build/pass-${pass}.stdout.txt" 2>&1; then
        tail -n 65 "build/pass-${pass}.stdout.txt"
        exit 1
    fi
done
python3 verify_build.py
