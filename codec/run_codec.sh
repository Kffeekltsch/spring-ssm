#!/bin/bash
# ============================================================
# S5 Codec Experiment — Run Script
# ============================================================
# Run from the spring-ssm root directory:
#   bash codec/run_codec.sh
#
# Curriculum: start with phase 1, validate, then advance.
# ============================================================

set -e
cd "$(dirname "$0")/.."  # cd to spring-ssm root

echo "============================================="
echo "  S5 Dynamics Codec — Compression Experiment"
echo "============================================="

# ─── Phase 1: Single sinusoids (litmus test) ───
echo ""
echo ">>> Phase 1: Single sinusoids (d_state=8, analytical decoder)"
echo "    Target: SNR > 30 dB"
echo ""
python codec/train_codec.py --config codec/configs/phase1.yaml

# ─── Phase 2: Sum of sinusoids ───
echo ""
echo ">>> Phase 2: Sum of sinusoids (d_state=16, analytical decoder)"
echo "    Target: SNR > 20 dB"
echo ""
python codec/train_codec.py --config codec/configs/phase2.yaml

# ─── Phase 3: Decaying sinusoids ───
echo ""
echo ">>> Phase 3: Decaying sinusoids (d_state=32, analytical decoder)"
echo "    Target: SNR > 15 dB"
echo ""
python codec/train_codec.py --config codec/configs/phase3.yaml

# ─── Phase 4: Full mix, learned decoder ───
echo ""
echo ">>> Phase 4: Full complexity (d_state=64, learned decoder)"
echo "    Target: SNR > 10 dB"
echo ""
python codec/train_codec.py --config codec/configs/phase4.yaml

echo ""
echo "============================================="
echo "  All phases complete!"
echo "============================================="
