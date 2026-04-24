#!/bin/bash
set -euo pipefail

# 仅在 Claude Code on the web 远程环境中运行
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "=== Session Start: syncing latest code from GitHub ===" >&2

cd "${CLAUDE_PROJECT_DIR}"

# 拉取所有分支的最新提交
git fetch origin 2>&1

# 拉取当前分支最新代码
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH" >&2

git pull origin "$CURRENT_BRANCH" 2>&1 || echo "Pull skipped (branch may not exist on remote yet)" >&2

echo "=== Sync complete ===" >&2
