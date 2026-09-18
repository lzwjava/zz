#!/usr/bin/env bash
# Create the project venv (inherits the system torch+transformers) and install GRPO deps.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
"$PY" -m venv --system-site-packages .venv

# --system-site-packages lets us reuse the working torch 2.10+cu128 / transformers 5.x install
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not visible in venv'; \
print('torch', torch.__version__, '->', torch.cuda.get_device_name(0))"

.venv/bin/pip install -q -r requirements.txt
.venv/bin/python - <<'EOF'
import trl, peft, bitsandbytes, transformers
print("trl", trl.__version__, "| peft", peft.__version__,
      "| bnb", bitsandbytes.__version__, "| transformers", transformers.__version__)
EOF
echo "OK: run './.venv/bin/python train_grpo.py --smoke'"
