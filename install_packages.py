# this script automatically installs packages listed only if they’re not already installed
import subprocess
import sys

# Define packages and their versions (if needed)
versioned_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "scikit-learn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "torch": "torch==2.1.0",
    "torchvision": "torchvision==0.16.0",
    "torchaudio": "torchaudio==2.1.0",
    "ipython": "ipython"
}

def install_dependencies():
    installed = []

    def install_if_missing(package, import_name=None):
        try:
            __import__(import_name or package)
        except ImportError:
            print(f"Installing: {package}")
            installed.append(package)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    for pkg, ver in versioned_packages.items():
        install_if_missing(ver, pkg)

    print("Requirements satisfied. Installed:", installed if installed else "Nothing new.")

# Only run if this script is called directly (not when imported)
if __name__ == "__main__":
    install_dependencies()
