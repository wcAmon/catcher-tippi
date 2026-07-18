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

# MLX 在找不到烤入建置機絕對路徑的 default metallib 時,會 fallback 去找
# 「與執行檔同目錄」下的 mlx.metallib(colocated fallback)——這正是
# apps/tippi/scripts/build-app.sh:20-22 對 Tippi.app 打包時依賴的行為
# (同一手法搬過來給 catcher-asr-host 的發布包用)。v0.1.0 沒有隨附這份
# colocated metallib,導致在建置機以外的機器上 MLX 印出
# "Failed to load the default metallib." 然後失敗——這就是本次 v0.1.1
# 要修的缺陷。
METALLIB=$(find target/release/build -path '*/out/build/lib/mlx.metallib' -print -quit)
test -n "${METALLIB}"
cp "${METALLIB}" "${stage}/${name}/mlx.metallib"

tar -czf "dist/${name}.tar.gz" -C "${stage}" "${name}"
(cd dist && shasum -a 256 "${name}.tar.gz" | tee "${name}.tar.gz.sha256")
rm -rf "${stage}"
echo "done: dist/${name}.tar.gz"
