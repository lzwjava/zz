#!/bin/bash
# Fix missing source files in LibreChat shallow git clone on RunPod
# Run this after cloning LibreChat but before building

HOST="root@213.173.110.200"
PORT="23634"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT -i $KEY"

echo "=== Restore empty files from git ==="
$SSH "cd /workspace/LibreChat && \
  git checkout -- packages/data-provider/src/actions.ts && \
  echo 'actions.ts restored (\$(wc -l < packages/data-provider/src/actions.ts) lines)'"

echo ""
echo "=== Build packages (skip client if already built) ==="
$SSH "cd /workspace/LibreChat && NODE_OPTIONS='--max-old-space-size=8192' npm run build:packages 2>&1 | tail -5"

echo ""
echo "=== Client build (Vite) ==="
$SSH "cd /workspace/LibreChat/client && rm -rf dist && NODE_OPTIONS='--max-old-space-size=8192' npx vite build 2>&1 | tail -10"