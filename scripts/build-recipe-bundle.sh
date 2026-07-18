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

# 自我完整性檢查:配方包本身必須自我完整——使用者的 agent 只會複製
# recipes/tomato-ears/ 到自己機器上的 app 目錄,repo 根目錄的
# .superpowers/、crates/、apps/、scripts/ 等其他目錄不會被帶過去,任何
# bundle 內文件/程式碼如果依賴這些路徑實際存在,對使用者來說就是死路徑。
#
# 這裡選擇「印出警告、不中斷建構」而非直接 FAIL:要精準區分「這一行是
# 需要修掉的死連結」還是「刻意保留的出處紀錄(見 PLAN.md 引言的引用慣例
# 說明)」需要逐行語意判讀——同一份文件裡兩種引用會混在一起,機械規則
# (例如「這行有沒有某個關鍵字」)很容易誤殺合法的 provenance 說明,或者
# 反過來放過真正的死連結。改成 WARNING 讓建構者人工複核全部命中行,比
# 自動 FAIL/自動放行都更不容易出錯,同時仍然保證命中內容不會被靜默忽略。
echo ""
echo "Checking bundle self-containedness (source-repo-only path references)..."
containedness_hits=$(grep -rnE 'docs/superpowers|\.superpowers|crates/|apps/|scripts/bootstrap' "${extract_dir}/${name}" || true)
if [ -n "${containedness_hits}" ]; then
  echo "⚠ WARNING: bundle contains references to source-repo-only paths:"
  echo "${containedness_hits}" | while IFS= read -r hit; do
    if echo "${hit}" | grep -q '源 repo\|_baseUrlNote\|_deviations'; then
      echo "  [provenance note, OK] ${hit}"
    else
      echo "  [needs manual review]  ${hit}"
    fi
  done
  echo "  -> lines marked [provenance note, OK] cite the source repo path for"
  echo "     traceability (allowed, see PLAN.md intro); anything marked"
  echo "     [needs manual review] should be checked by hand before shipping."
  echo "     Build NOT failed automatically — see this script's comment for why."
else
  echo "✓ No source-repo-only path references found in bundle"
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
