#!/bin/zsh
# 打包 catcher-asr-host 為可發布的 tar.gz 並產生 SHA-256。
# 產物路徑:dist/catcher-asr-host-v<version>-macos-arm64.tar.gz(.sha256)
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p catcher-asr-host

version=$(cargo pkgid -p catcher-asr-host | sed 's/.*[@#]//')
stage=$(mktemp -d)
name="catcher-asr-host-v${version}-macos-arm64"
mkdir -p "dist" "${stage}/${name}"

cp target/release/catcher-asr-host "${stage}/${name}/"
cp docs/protocol/asr-host-v1.md "${stage}/${name}/PROTOCOL.md"
tar -czf "dist/${name}.tar.gz" -C "${stage}" "${name}"
shasum -a 256 "dist/${name}.tar.gz" | tee "dist/${name}.tar.gz.sha256"
rm -rf "${stage}"
echo "done: dist/${name}.tar.gz"
