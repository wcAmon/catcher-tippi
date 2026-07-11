#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
TIPPI_DIR=${SCRIPT_DIR:h}
ROOT=${TIPPI_DIR:h:h}
APP=${TIPPI_DIR}/build/Tippi.app
CONTENTS=${APP}/Contents

cd ${ROOT}
cargo build -p catcher-ffi --release
install_name_tool -id @rpath/libcatcher_ffi.dylib target/release/libcatcher_ffi.dylib
swift build --package-path apps/tippi -c release --product Tippi
BIN_PATH=$(swift build --package-path apps/tippi -c release --show-bin-path)

rm -rf ${APP}
mkdir -p ${CONTENTS}/MacOS ${CONTENTS}/Frameworks
cp ${BIN_PATH}/Tippi ${CONTENTS}/MacOS/Tippi
cp target/release/libcatcher_ffi.dylib ${CONTENTS}/Frameworks/
METALLIB=$(find target/release/build -path '*/out/build/lib/mlx.metallib' -print -quit)
test -n "${METALLIB}"
cp ${METALLIB} ${CONTENTS}/Frameworks/mlx.metallib
cp apps/tippi/Resources/Info.plist ${CONTENTS}/Info.plist
print -n 'APPL????' > ${CONTENTS}/PkgInfo

if otool -l ${CONTENTS}/MacOS/Tippi | grep -q "${ROOT}/target/release"; then
    install_name_tool -delete_rpath "${ROOT}/target/release" ${CONTENTS}/MacOS/Tippi
fi

codesign --force --sign - ${CONTENTS}/Frameworks/libcatcher_ffi.dylib
codesign --force --sign - ${CONTENTS}/Frameworks/mlx.metallib
codesign --force --sign - --entitlements apps/tippi/Resources/Tippi.entitlements ${APP}
${SCRIPT_DIR}/verify-app.sh
