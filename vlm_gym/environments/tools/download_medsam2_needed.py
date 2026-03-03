# install_missing_deps.py
import subprocess
import sys

print("Installing missing dependencies for MedSAM2...")

# 需要安装的包
packages = [
    "iopath",
    "fvcore"  # 可能也需要
]

for package in packages:
    print(f"\nInstalling {package}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"✗ Exception installing {package}: {e}")

# 验证安装
print("\nVerifying installations...")
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} imported successfully")
    except ImportError:
        print(f"✗ {package} import failed")

print("\nDone!")