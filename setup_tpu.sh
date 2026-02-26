#!/bin/bash
set -e

echo "=== TPU VM Setup for HyperscaleES ==="

# 1. Install Node.js + Claude Code
if ! command -v node &> /dev/null; then
    echo ">>> Installing Node.js via nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install 22
fi
if ! command -v claude &> /dev/null; then
    echo ">>> Installing Claude Code..."
    npm install -g @anthropic-ai/claude-code
fi

# 2. Upgrade pip/setuptools and install JAX for TPU
echo ">>> Upgrading pip and installing JAX for TPU..."
pip install --upgrade pip setuptools
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. Clone repo
echo ">>> Cloning HyperscaleES..."
mkdir -p ~/SNLP
cd ~/SNLP
BRANCH="${1:-warming-up}"
if [ -d "HyperscaleES" ]; then
    echo "    Repo already exists, pulling latest..."
    cd HyperscaleES
    git fetch --all
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone -b "$BRANCH" https://github.com/shr1ram/HyperscaleES.git
    cd HyperscaleES
fi

# 3. Install package + dependencies
echo ">>> Installing HyperscaleES and dependencies..."
pip install -e .

# 4. Verify TPU
echo ">>> Verifying TPU devices..."
python3 -c "import jax; devs = jax.devices(); print(f'Found {len(devs)} TPU devices: {devs}')"

# 5. wandb login reminder
echo ""
echo "=== Setup complete! ==="
echo "Run 'wandb login' if you haven't already."
echo "Then start training with e.g.:"
echo "  python -m llm_experiments.general_do_evolution --model_choice tl1.1B --task fastzero --num_epochs 3 --parallel_generations_per_gpu 256 --sigma 1e-3 --lr_scale 1.0 --noiser eggroll --track --wandb_project HyperscaleExp --wandb_name tpu_test"
