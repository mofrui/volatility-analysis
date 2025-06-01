#!/bin/bash

#!/bin/bash

# -------------------------------
# Python Version Check (3.11.x)
# -------------------------------
PY_VERSION=$(python3 --version | cut -d " " -f2)
if [[ "$PY_VERSION" != 3.11* ]]; then
  echo "[WARNING] Python 3.11 is required. Detected: $PY_VERSION"
  echo ""
  echo "▶ To proceed, please install Python 3.11 and make sure it is used by 'python3'."
  echo ""
  echo "Recommended (cross-platform): Use pyenv to set up a Python 3.11 environment:"
  echo "   1. Install pyenv: https://github.com/pyenv/pyenv#installation"
  echo "   2. Then run:"
  echo "      pyenv install 3.11.8"
  echo "      pyenv virtualenv 3.11.8 optiver_env"
  echo "      pyenv activate optiver_env"
  echo ""
  echo "Or download Python 3.11 manually: https://www.python.org/downloads/release/python-3110/"
  echo ""
  exit 1
else
  echo " Python version is compatible: $PY_VERSION"
fi


# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo "Installing required Python packages..."
python3 -m pip install --upgrade \
  pandas numpy==1.26.4 pyarrow scikit-learn statsmodels matplotlib \
  ipython tensorflow==2.16.2 arch tqdm shiny faicons seaborn \
  xgboost joblib absl-py jupyter jupyter-cache 


# Check for Quarto CLI
if ! command -v quarto &> /dev/null; then
  echo "[WARNING] Quarto CLI not found. You need to install it to render .qmd reports."
  echo ""
  echo "▶ To install Quarto CLI:"
  echo "  • macOS (Homebrew): brew install quarto"
  echo "  • Linux (Debian/Ubuntu):"
  echo "      sudo apt install gdebi-core"
  echo "      wget https://quarto.org/download/latest/quarto-linux-amd64.deb"
  echo "      sudo gdebi quarto-linux-amd64.deb"
  echo "  • Windows: https://quarto.org/docs/download/"
  echo ""
  echo "After installing, you can run: quarto render report.qmd"
else
  echo "Quarto CLI is installed: $(quarto --version)"
fi

echo "Environment setup complete."


