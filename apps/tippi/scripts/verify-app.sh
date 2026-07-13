#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
TIPPI_DIR=${SCRIPT_DIR:h}
APP=${TIPPI_DIR}/build/Tippi.app
EXECUTABLE=${APP}/Contents/MacOS/Tippi
LIBRARY=${APP}/Contents/Frameworks/libcatcher_ffi.dylib
METALLIB=${APP}/Contents/Frameworks/mlx.metallib
PLIST=${APP}/Contents/Info.plist
NOTICE=${APP}/Contents/Resources/THIRD_PARTY_NOTICES.md

test -x ${EXECUTABLE}
test -f ${LIBRARY}
test -f ${METALLIB}
test -f ${PLIST}
test -f ${NOTICE}
test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' ${PLIST})" = "Tippi"
test "$(/usr/libexec/PlistBuddy -c 'Print :NSMicrophoneUsageDescription' ${PLIST})" != ""
file ${EXECUTABLE} | grep -q 'arm64'
otool -L ${EXECUTABLE} | grep -q '@rpath/libcatcher_ffi.dylib'
otool -l ${EXECUTABLE} | grep -q '@executable_path/../Frameworks'
if otool -L ${EXECUTABLE} | tail -n +2 | grep -q '/.worktrees/'; then
    print -u2 "Tippi contains a worktree dylib path"
    exit 1
fi
codesign --verify --deep --strict ${APP}
ENTITLEMENTS=$(codesign -d --entitlements - ${APP} 2>&1)
if print -- "${ENTITLEMENTS}" | grep -q 'com.apple.security.app-sandbox'; then
    print -u2 "Tippi must not be sandboxed for cross-app injection"
    exit 1
fi
print "Verified ${APP}"
