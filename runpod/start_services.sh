#!/bin/bash
# Start all services on RunPod for LibreChat + SGLang
# Server: 213.173.110.200:23634
#
# Usage: bash runpod/start_services.sh

set -e

HOST="root@213.173.110.200"
PORT="23634"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT -i $KEY"

echo "=== 1. Start MongoDB ==="
$SSH "mongod --dbpath /workspace/mongodb --logpath /workspace/mongodb/mongod.log --fork 2>/dev/null && \
  echo 'MongoDB started' || echo 'MongoDB already running'"
$SSH "mongosh --quiet --eval 'db.runCommand({ping:1}).ok' 2>/dev/null" && echo "MongoDB: OK"

echo ""
echo "=== 2. Check SGLang (Qwen 14B) ==="
$SSH "curl -s http://127.0.0.1:30000/v1/models | head -50" && echo "SGLang: OK" || echo "SGLang: NOT RUNNING — start manually"

echo ""
echo "=== 3. Kill anything on port 8888 ==="
$SSH "fuser -k 8888/tcp 2>/dev/null; sleep 1; echo 'Port 8888 freed'"

echo ""
echo "=== 4. Rebuild native modules (Node v22 compat) ==="
$SSH "cd /workspace/LibreChat && npm rebuild mongodb 2>&1 | tail -3"

echo ""
echo "=== 5. Start LibreChat on 8888 ==="
$SSH "cd /workspace/LibreChat && setsid node api/server/index.js > /workspace/librechat.log 2>&1 & disown"
sleep 5

echo ""
echo "=== 6. Verify ==="
$SSH "ss -tlnp | grep 8888" && echo "LibreChat: LISTENING on 8888"
$SSH "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888" && echo " LibreChat: OK"

echo ""
echo "=== DONE ==="
echo "Access at: http://213.173.110.200:8888"
echo "Logs: $SSH 'tail -50 /workspace/librechat.log'"