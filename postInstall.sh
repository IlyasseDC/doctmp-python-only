#!/bin/bash
echo "🔧 Running postInstall.sh ..."

# Forcer numpy 1.20.1 (compatible jpegio et torch==1.11.0+cu113)
pip install --force-reinstall --no-cache-dir "numpy==1.20.1"

# Vérification
python - <<EOF
import sys, numpy, torch
print("✅ Python:", sys.version)
print("✅ NumPy :", numpy.__version__)
print("✅ Torch :", torch.__version__)
EOF
