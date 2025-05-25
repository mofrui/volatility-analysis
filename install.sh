#!/bin/bash

# Check Python version (must be 3.11.x)
PY_VERSION=$(python3 --version | cut -d " " -f2)
if [[ "$PY_VERSION" != 3.11* ]]; then
  echo "❌ Python 3.11 is required. You're using $PY_VERSION"
  echo "Please install Python 3.11 (e.g., via pyenv or Homebrew) and try again."
  exit 1
else
  echo "✅ Python version is compatible: $PY_VERSION"
fi


# 3. Upgrade pip
echo "🚀 Upgrading pip..."
pip install --upgrade pip

# 4. Install all required packages
echo "📦 Installing dependencies..."
pip install pandas numpy==1.23.5 pyarrow scikit-learn statsmodels matplotlib \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 ipython tensorflow==2.16.2 \
    arch tqdm shiny faicons seaborn xgboost joblib

echo "✅ Environment setup complete."
