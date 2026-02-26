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

# 2. Install Python 3.11 if not available
if ! python3.11 --version &> /dev/null; then
    echo ">>> Installing Python 3.11..."
    sudo apt update -y
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# 3. Clone repo first
echo ">>> Cloning HyperscaleES..."
mkdir -p ~/SNLP
cd ~/SNLP
BRANCH="${1:-warming-up}"
if [ -d "HyperscaleES/.git" ]; then
    echo "    Repo already exists, pulling latest..."
    cd HyperscaleES
    git fetch --all
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    rm -rf HyperscaleES
    git clone -b "$BRANCH" https://github.com/shr1ram/HyperscaleES.git
    cd HyperscaleES
fi

# 4. Create venv with Python 3.11 inside the repo
if [ ! -d ".venv" ]; then
    echo ">>> Creating Python 3.11 venv..."
    python3.11 -m venv .venv
fi
source .venv/bin/activate

# 5. Upgrade pip/setuptools and install JAX for TPU
echo ">>> Upgrading pip and installing JAX for TPU..."
python3 -m pip install --upgrade pip setuptools
python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 6. Install package + dependencies
echo ">>> Installing HyperscaleES and dependencies..."
python3 -m pip install -e .

# 7. Verify TPU
echo ">>> Verifying TPU devices..."
python3 -c "import jax; devs = jax.devices(); print(f'Found {len(devs)} TPU devices: {devs}')"

# 8. Done
echo ""
echo "=== Setup complete! ==="
echo "Run 'wandb login' if you haven't already."
echo "Then activate the venv and start training:"
echo "  source ~/SNLP/HyperscaleES/.venv/bin/activate"
echo "  cd ~/SNLP/HyperscaleES"
echo "  python -m llm_experiments.general_do_evolution --model_choice tl1.1B --task fastzero --num_epochs 3 --parallel_generations_per_gpu 256 --sigma 1e-3 --lr_scale 1.0 --noiser eggroll --track --wandb_project HyperscaleExp --wandb_name tpu_test"
