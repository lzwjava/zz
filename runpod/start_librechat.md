# RunPod — LibreChat + SGLang Chatbot

**Server:** `213.173.110.200` port `23634` (key: `~/.ssh/id_ed25519`)
**Access:** http://213.173.110.200:8888

## Services

| Service | Port | What |
|---------|------|------|
| LibreChat | **8888** | Chatbot UI (login → pick Qwen model) |
| SGLang | 30000 | Qwen 2.5 14B AWQ inference |
| MongoDB | 27017 | LibreChat data store |

## Quick Start

```bash
# Start everything
bash runpod/start_services.sh

# Or manually via SSH
ssh root@213.173.110.200 -p 23634 -i ~/.ssh/id_ed25519
cd /workspace/LibreChat && node api/server/index.js &
```

## Rebuilding (if needed)

```bash
bash runpod/build_librechat.sh
```

Or SSH in and run individual steps:
```bash
# Fix empty files from shallow clone
cd /workspace/LibreChat && git checkout -- packages/data-provider/src/actions.ts

# Build packages
NODE_OPTIONS='--max-old-space-size=8192' npm run build:packages

# Build client app
cd client && NODE_OPTIONS='--max-old-space-size=8192' npx vite build
```

## What Was Done

| Step | Detail |
|------|--------|
| Port | Changed LibreChat from 3080 → 8888 in `.env` |
| Jupyter | Killed on 8888 to free the port |
| Native modules | `npm rebuild mongodb` for Node v22.23.1 ABI |
| Empty files | `actions.ts` was 0 bytes — restored from git |
| Client | Vite build compiled to `client/dist/` |
| Running | LibreChat PID 20534 on 0.0.0.0:8888 (HTTP 200) |

## Verification

```bash
ssh root@213.173.110.200 -p 23634 -i ~/.ssh/id_ed25519
curl http://127.0.0.1:8888          # should return HTML
curl http://127.0.0.1:30000/v1/models  # should return model list
mongosh --quiet --eval 'db.runCommand({ping:1}).ok'  # should return 1
```
