#!/bin/bash
# Respawn a preempted TPU VM and update local SSH config.
# Run from your Mac: ./tpu_respawn.sh [name] [zone] [type]
#
# Examples:
#   ./tpu_respawn.sh                                    # defaults: hyperscale-v6e, europe-west4-a, v6e-8
#   ./tpu_respawn.sh hyperscale-v4 us-central2-b v4-8   # custom config

set -euo pipefail

NAME="${1:-hyperscale-v6e}"
ZONE="${2:-europe-west4-a}"
TYPE="${3:-v6e-8}"
PROJECT="statistical-nlp"
SSH_CONFIG="$HOME/.ssh/config"
SSH_HOST="gcp-tpu-v6e"

echo "=== TPU Respawn: $NAME ($TYPE) in $ZONE ==="

# 1. Check current status
STATUS=$(gcloud compute tpus tpu-vm describe "$NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --format='get(state)' 2>/dev/null || echo "NOT_FOUND")

echo "Current status: $STATUS"

if [ "$STATUS" = "READY" ]; then
    echo "TPU is already running."
    IP=$(gcloud compute tpus tpu-vm describe "$NAME" \
      --zone="$ZONE" --project="$PROJECT" \
      --format='get(networkEndpoints[0].accessConfig.externalIp)')
    echo "IP: $IP"
elif [ "$STATUS" = "PREEMPTED" ] || [ "$STATUS" = "STOPPED" ] || [ "$STATUS" = "NOT_FOUND" ]; then
    # 2. Delete if preempted/stopped (can't restart spot VMs, must recreate)
    if [ "$STATUS" != "NOT_FOUND" ]; then
        echo ">>> Deleting $STATUS VM..."
        gcloud compute tpus tpu-vm delete "$NAME" \
          --zone="$ZONE" --project="$PROJECT" --quiet
    fi

    # 3. Pick the correct runtime version for the TPU generation
    case "$TYPE" in
        v6e-*) RUNTIME_VERSION="v2-alpha-tpuv6e" ;;
        v5e-*) RUNTIME_VERSION="v2-alpha-tpuv5e" ;;
        v4-*)  RUNTIME_VERSION="tpu-ubuntu2204-base" ;;
        *)     RUNTIME_VERSION="tpu-ubuntu2204-base" ;;
    esac

    # 4. Create new VM
    echo ">>> Creating $TYPE spot TPU VM (runtime: $RUNTIME_VERSION)..."
    gcloud compute tpus tpu-vm create "$NAME" \
      --zone="$ZONE" --project="$PROJECT" \
      --accelerator-type="$TYPE" \
      --version="$RUNTIME_VERSION" \
      --spot

    # 4. Get new IP
    IP=$(gcloud compute tpus tpu-vm describe "$NAME" \
      --zone="$ZONE" --project="$PROJECT" \
      --format='get(networkEndpoints[0].accessConfig.externalIp)')
    echo "New IP: $IP"
else
    echo "Unexpected status: $STATUS. Check the GCP console."
    exit 1
fi

# 5. Update SSH config with new IP
if grep -q "Host $SSH_HOST" "$SSH_CONFIG" 2>/dev/null; then
    # Replace the HostName line after the matching Host block
    # Uses awk to find the right Host block and update only its HostName
    awk -v host="$SSH_HOST" -v ip="$IP" '
        /^Host / { in_block = ($2 == host) }
        in_block && /HostName/ { $2 = ip; in_block = 0 }
        { print }
    ' "$SSH_CONFIG" > "${SSH_CONFIG}.tmp" && mv "${SSH_CONFIG}.tmp" "$SSH_CONFIG"
    echo ">>> Updated SSH config ($SSH_HOST -> $IP)"
else
    echo ">>> WARNING: No '$SSH_HOST' entry found in $SSH_CONFIG"
    echo "    Add manually or run: gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE"
fi

# 6. Clear old host keys
ssh-keygen -R "$IP" 2>/dev/null || true

# 7. Push SSH keys via gcloud (required for fresh VMs)
echo ">>> Propagating SSH keys..."
gcloud compute tpus tpu-vm ssh "$NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --command="echo 'SSH key propagation successful'" 2>&1

# 8. Test direct SSH
echo ">>> Testing direct SSH..."
if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$SSH_HOST" "echo 'Direct SSH works'" 2>/dev/null; then
    echo ""
    echo "=== TPU ready! ==="
    echo "  ssh $SSH_HOST"
    echo ""
    echo "Run setup on the TPU:"
    echo "  curl -sL -H 'Accept: application/vnd.github.v3.raw' \\"
    echo "    'https://api.github.com/repos/shr1ram/HyperscaleES/contents/setup_tpu.sh?ref=warming-up' | bash"
else
    echo ""
    echo "Direct SSH failed. Try: gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE"
fi
