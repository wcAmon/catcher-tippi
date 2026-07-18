#!/bin/zsh
# 打包 tomato-ears 配方與 env-base 為可發布的 tar.gz 並產生 SHA-256。
# 產物路徑: dist/tomato-ears-recipe-v0.1.0.tar.gz(.sha256)
set -euo pipefail
cd "$(dirname "$0")/.."

name="tomato-ears-recipe-v0.1.0"
mkdir -p "dist"

# 使用 git archive + prefix 確保只包含已追蹤的檔案
# (排除 untracked/ignored 檔案，保證 bundle == committed tree)
git archive --prefix="${name}/" HEAD recipes/ | gzip > "dist/${name}.tar.gz"

# 產生 SHA-256 (bare-filename LF 格式)
(cd dist && shasum -a 256 "${name}.tar.gz" | tee "${name}.tar.gz.sha256")

# 驗證: 解包並比對原樹是否一致
verify_dir=$(mktemp -d)
extract_dir="${verify_dir}/extracted"
mkdir -p "${extract_dir}"

# 解包
tar -xzf "dist/${name}.tar.gz" -C "${extract_dir}"

# 從 git archive 提取原始檔案作為對照
git_archive_dir="${verify_dir}/git-archive"
mkdir -p "${git_archive_dir}"
git archive --prefix="${name}/" HEAD recipes/ | tar -xz -C "${git_archive_dir}"

# 比對內容是否一致
echo "Verifying bundle integrity..."
if diff -r "${extract_dir}/${name}" "${git_archive_dir}/${name}"; then
  echo "✓ Bundle content matches git archive"
else
  echo "✗ Bundle content differs from git archive!"
  rm -rf "${verify_dir}"
  exit 1
fi

# 驗證 SHA-256
echo "Verifying SHA-256 checksum..."
if (cd dist && shasum -c "${name}.tar.gz.sha256"); then
  echo "✓ SHA-256 checksum valid"
else
  echo "✗ SHA-256 checksum verification failed!"
  rm -rf "${verify_dir}"
  exit 1
fi

# 驗證 .sha256 檔案最後一個位元組是 0x0A (LF)
last_byte=$(tail -c 1 "dist/${name}.tar.gz.sha256" | od -An -tx1 | tr -d ' ')
if [ "$last_byte" = "0a" ]; then
  echo "✓ SHA-256 file ends with LF (0x0A)"
else
  echo "✗ SHA-256 file does not end with LF!"
  rm -rf "${verify_dir}"
  exit 1
fi

# 清理
rm -rf "${verify_dir}"

# 輸出摘要
bundle_size=$(ls -lh "dist/${name}.tar.gz" | awk '{print $5}')
file_count=$(tar -tzf "dist/${name}.tar.gz" | wc -l)
sha256_value=$(cat "dist/${name}.tar.gz.sha256" | awk '{print $1}')

echo ""
echo "✓ Bundle created successfully!"
echo "  File: dist/${name}.tar.gz"
echo "  Size: ${bundle_size}"
echo "  Files: ${file_count}"
echo "  SHA-256: ${sha256_value}"
