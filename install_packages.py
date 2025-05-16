import subprocess
import sys
import pkg_resources
import sys

def check_python_version(allowed_versions=("3.9", "3.10")):
    current_version = sys.version
    if not sys.version.startswith(allowed_versions):
        print("Warning: This project is tested with Python versions:", ", ".join(allowed_versions))
        print("You are using:", current_version)
        print("TensorFlow 2.14 may not work on this version.")
    else:
        print(f" Python version is compatible: {current_version.split()[0]}")

# Define packages and specific versions
versioned_packages = {
    "pandas": "pandas",
    "numpy": "numpy==1.23.5",
    "pyarrow": "pyarrow",
    "scikit-learn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "torch": "torch==2.1.0",
    "torchvision": "torchvision==0.16.0",
    "torchaudio": "torchaudio==2.1.0",
    "ipython": "ipython",
    "tensorflow": "tensorflow==2.14.0",
    "arch": "arch",
    "tqdm": "tqdm"   
}


def install_dependencies():
    installed = []

    for import_name, pip_spec in versioned_packages.items():
        package_name = pip_spec.split("==")[0]
        required_version = pip_spec.split("==")[1] if "==" in pip_spec else None

        try:
            dist = pkg_resources.get_distribution(import_name)
            current_version = dist.version

            if required_version and current_version != required_version:
                print(f"{import_name}=={current_version} installed, replacing with {pip_spec}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", pip_spec]
                )
                installed.append(pip_spec)
        except pkg_resources.DistributionNotFound:
            print(f"Installing missing package: {pip_spec}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_spec]
            )
            installed.append(pip_spec)

    print("All dependencies installed. Changes:", installed if installed else "Nothing new.")

if __name__ == "__main__":
    install_dependencies()
