#!/bin/bash
# Finish LibreChat setup on RunPod server
# Run after SGLang server is already serving on port 30000

set -e

HOST="root@213.173.110.200"
PORT="23634"
KEY="$HOME/.ssh/id_ed25519"
LIBRECHAT="/workspace/LibreChat"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT -i $KEY"

echo "=== 1. Fix empty source files from shallow clone ==="
$SSH "cd $LIBRECHAT && \
  curl -sL 'https://raw.githubusercontent.com/danny-avila/LibreChat/main/packages/data-provider/src/actions.ts' -o packages/data-provider/src/actions.ts && \
  echo 'actions.ts fixed' && \
  curl -sL 'https://raw.githubusercontent.com/danny-avila/LibreChat/main/packages/data-provider/src/accessPermissions.ts' -o packages/data-provider/src/accessPermissions.ts && \
  echo 'accessPermissions.ts fixed'"

echo "=== 2. Build frontend ==="
$SSH "cd $LIBRECHAT && npm run frontend 2>&1" | tail -20
echo "Frontend build exit: $?"

echo "=== 3. Start backend in background ==="
$SSH "cd $LIBRECHAT && nohup npm run backend > /workspace/librechat.log 2>&1 & echo \"PID=\$!\""
sleep 5
echo "=== 4. Check backend status ==="
$SSH "tail -20 /workspace/librechat.log"
echo "=== 5. Test LibreChat API ==="
$SSH "curl -s http://127.0.0.1:3080/api/health 2>/dev/null | head -5" || echo "Not ready yet (wait 30s)"
sleep 30
$SSH "curl -s http://127.0.0.1:3080/api/health 2>/dev/null | head -5" || echo "Not ready yet"

echo ""
echo "=== DONE ==="
echo "SGLang: http://127.0.0.1:30000/v1/models"
echo "LibreChat: http://0.0.0.0:3080"
echo "To access from outside, check RunPod dashboard for public IP"