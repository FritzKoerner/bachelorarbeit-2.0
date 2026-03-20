#!/bin/bash
# ==============================================================================
# Node Diagnostics — GPU, driver, Vulkan, and Genesis compatibility check
#
# Run on any HPC node to collect driver/GPU info for debugging.
# Output is structured for easy parsing.
#
# Usage:
#   source ~/genesis_v04/hpc/setup_env.sh --load && bash ~/genesis_v04/hpc/check_node.sh
#   # or inside an interactive SLURM session:
#   srun --partition=paula --gres=gpu:1 --time=00:10:00 --pty bash -c \
#       "source ~/genesis_v04/hpc/setup_env.sh --load && bash ~/genesis_v04/hpc/check_node.sh"
# ==============================================================================

set -euo pipefail

# ── Header ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "  NODE DIAGNOSTICS"
echo "  $(date -Iseconds)"
echo "========================================"
echo ""

# ── 1. Node identity ────────────────────────────────────────────────────────
echo "--- NODE ---"
echo "[NODE] hostname: $(hostname)"
echo "[NODE] kernel: $(uname -r)"
echo "[NODE] arch: $(uname -m)"
if command -v lsb_release &>/dev/null; then
    echo "[NODE] os: $(lsb_release -ds 2>/dev/null || echo 'unknown')"
elif [ -f /etc/os-release ]; then
    echo "[NODE] os: $(. /etc/os-release && echo "$PRETTY_NAME")"
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[SLURM] job_id: $SLURM_JOB_ID"
    echo "[SLURM] partition: ${SLURM_JOB_PARTITION:-unknown}"
    echo "[SLURM] nodelist: ${SLURM_JOB_NODELIST:-unknown}"
    echo "[SLURM] gpus: ${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES:-unknown}}"
fi
echo ""

# ── 2. NVIDIA driver + GPU ──────────────────────────────────────────────────
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    echo "[GPU] nvidia-smi: found"
    echo "[GPU] driver_version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"

    # Per-GPU details
    gpu_idx=0
    while IFS=, read -r name cc mem_total mem_free; do
        name=$(echo "$name" | xargs)
        cc=$(echo "$cc" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
        echo "[GPU:$gpu_idx] name: $name"
        echo "[GPU:$gpu_idx] compute_capability: $cc"
        echo "[GPU:$gpu_idx] memory_total_mb: $mem_total"
        echo "[GPU:$gpu_idx] memory_free_mb: $mem_free"

        # BatchRenderer compatibility (Turing+ = cc >= 7.5)
        major="${cc%%.*}"
        minor="${cc#*.}"
        if [ "$major" -gt 7 ] || { [ "$major" -eq 7 ] && [ "$minor" -ge 5 ]; }; then
            echo "[GPU:$gpu_idx] batch_renderer_compatible: YES (cc $cc >= 7.5)"
        else
            echo "[GPU:$gpu_idx] batch_renderer_compatible: NO (cc $cc < 7.5, Turing+ required)"
        fi
        gpu_idx=$((gpu_idx + 1))
    done < <(nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free --format=csv,noheader,nounits)
    echo "[GPU] gpu_count: $gpu_idx"
else
    echo "[GPU] nvidia-smi: NOT FOUND"
fi
echo ""

# ── 3. CUDA ─────────────────────────────────────────────────────────────────
echo "--- CUDA ---"
if command -v nvcc &>/dev/null; then
    echo "[CUDA] nvcc: $(nvcc --version 2>&1 | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
else
    echo "[CUDA] nvcc: not found (toolkit may not be in PATH)"
fi
# nvidia-smi reports the max supported CUDA version
if command -v nvidia-smi &>/dev/null; then
    cuda_ver=$(nvidia-smi 2>&1 | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
    echo "[CUDA] driver_max_cuda: $cuda_ver"
fi
echo ""

# ── 4. Vulkan ────────────────────────────────────────────────────────────────
echo "--- VULKAN ---"
if command -v vulkaninfo &>/dev/null; then
    echo "[VULKAN] vulkaninfo: found"
    vk_api=$(vulkaninfo 2>/dev/null | grep -m1 'apiVersion' | awk '{print $NF}' || echo "error")
    echo "[VULKAN] api_version: $vk_api"
    vk_devs=$(vulkaninfo 2>/dev/null | grep -c 'GPU id' || echo "0")
    echo "[VULKAN] device_count: $vk_devs"
    # List device names
    vulkaninfo 2>/dev/null | grep 'deviceName' | while IFS= read -r line; do
        echo "[VULKAN] device: $(echo "$line" | sed 's/.*= //')"
    done
else
    echo "[VULKAN] vulkaninfo: not found"
fi
# Check for Vulkan ICD loaders
echo "[VULKAN] VK_ICD_FILENAMES: ${VK_ICD_FILENAMES:-unset}"
echo "[VULKAN] VK_DRIVER_FILES: ${VK_DRIVER_FILES:-unset}"
for icd_dir in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do
    if [ -d "$icd_dir" ]; then
        echo "[VULKAN] icd_dir: $icd_dir -> $(ls "$icd_dir" 2>/dev/null | tr '\n' ' ')"
    fi
done
# Check for libvulkan
libvulkan=$(ldconfig -p 2>/dev/null | grep libvulkan || echo "not found")
echo "[VULKAN] libvulkan: $libvulkan"
echo ""

# ── 5. Loaded modules ───────────────────────────────────────────────────────
echo "--- MODULES ---"
if command -v module &>/dev/null; then
    echo "[MODULES] loaded: $(module list 2>&1 | grep -v 'Currently Loaded' | tr '\n' ' ')"
else
    echo "[MODULES] module command: not available"
fi
echo ""

# ── 6. Python / Conda ───────────────────────────────────────────────────────
echo "--- PYTHON ---"
echo "[PYTHON] which: $(which python 2>/dev/null || echo 'not found')"
echo "[PYTHON] version: $(python --version 2>&1 || echo 'not found')"
echo "[CONDA] env: ${CONDA_DEFAULT_ENV:-not active}"
echo "[CONDA] prefix: ${CONDA_PREFIX:-none}"
echo ""

# ── 7. PyTorch CUDA ──────────────────────────────────────────────────────────
echo "--- PYTORCH ---"
python -c "
import sys
try:
    import torch
    print(f'[PYTORCH] version: {torch.__version__}')
    print(f'[PYTORCH] cuda_available: {torch.cuda.is_available()}')
    print(f'[PYTORCH] cuda_version: {torch.version.cuda}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
            print(f'[PYTORCH:GPU{i}] name: {name}')
            print(f'[PYTORCH:GPU{i}] capability: {cap[0]}.{cap[1]}')
            print(f'[PYTORCH:GPU{i}] memory_gb: {mem:.1f}')
except Exception as e:
    print(f'[PYTORCH] error: {e}')
" 2>&1
echo ""

# ── 8. Genesis ───────────────────────────────────────────────────────────────
echo "--- GENESIS ---"
python -c "
import sys
try:
    import genesis as gs
    print(f'[GENESIS] version: {gs.__version__}')
except Exception as e:
    print(f'[GENESIS] import_error: {e}')
    sys.exit(0)

# Check gs_madrona (BatchRenderer backend)
try:
    import gs_madrona
    print('[GENESIS] gs_madrona: importable')
except ImportError as e:
    print(f'[GENESIS] gs_madrona: NOT importable ({e})')

# Check rasterizer (internal module path varies by version)
rast_found = False
for mod_path in ['genesis.vis.rasterizer', 'genesis.vis.Rasterizer', 'genesis.vis']:
    try:
        mod = __import__(mod_path, fromlist=['Rasterizer'])
        if hasattr(mod, 'Rasterizer'):
            print(f'[GENESIS] rasterizer: importable ({mod_path}.Rasterizer)')
            rast_found = True
            break
    except (ImportError, ModuleNotFoundError):
        pass
if not rast_found:
    # Check if genesis exposes it through scene internals
    try:
        from genesis.vis.rasterizer import Rasterizer
        print('[GENESIS] rasterizer: importable (genesis.vis.rasterizer)')
        rast_found = True
    except Exception:
        pass
if not rast_found:
    print('[GENESIS] rasterizer: not directly importable (accessed internally by scene — usually OK)')

# Check BatchRenderer (Madrona backend)
br_found = False
for mod_path in ['genesis.vis.batch_renderer', 'genesis.vis.BatchRenderer', 'genesis.vis']:
    try:
        mod = __import__(mod_path, fromlist=['BatchRenderer'])
        if hasattr(mod, 'BatchRenderer'):
            print(f'[GENESIS] batch_renderer: importable ({mod_path}.BatchRenderer)')
            br_found = True
            break
    except (ImportError, ModuleNotFoundError):
        pass
if not br_found:
    print('[GENESIS] batch_renderer: not directly importable (accessed internally by scene — usually OK)')
" 2>&1
echo ""

# ── 9. rsl-rl ────────────────────────────────────────────────────────────────
echo "--- RSL-RL ---"
python -c "
try:
    import rsl_rl
    print(f'[RSL_RL] version: {getattr(rsl_rl, \"__version__\", \"unknown\")}')
except Exception as e:
    print(f'[RSL_RL] error: {e}')
" 2>&1
echo ""

# ── 10. Quick functional test ────────────────────────────────────────────────
echo "--- FUNCTIONAL TEST ---"
python -c "
import torch
if not torch.cuda.is_available():
    print('[TEST] skip: no CUDA')
else:
    cap = torch.cuda.get_device_capability(0)
    cc = f'{cap[0]}.{cap[1]}'

    # Test 1: basic Genesis init
    try:
        import genesis as gs
        gs.init(backend=gs.gpu, logging_level='warning')
        print('[TEST] gs.init(gpu): OK')
    except Exception as e:
        print(f'[TEST] gs.init(gpu): FAIL ({e})')

    # Test 2: attempt BatchRenderer probe (non-destructive)
    if cap[0] > 7 or (cap[0] == 7 and cap[1] >= 5):
        print(f'[TEST] batch_renderer_eligible: YES (cc={cc})')
        try:
            import gs_madrona
            print('[TEST] gs_madrona_import: OK')
        except Exception as e:
            print(f'[TEST] gs_madrona_import: FAIL ({e})')
    else:
        print(f'[TEST] batch_renderer_eligible: NO (cc={cc}, need >= 7.5)')
        print('[TEST] fallback: Rasterizer with env_separate_rigid=True')
" 2>&1
echo ""

echo "========================================"
echo "  DIAGNOSTICS COMPLETE"
echo "========================================"
