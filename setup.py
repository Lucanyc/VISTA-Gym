
from setuptools import setup, find_packages

setup(
    name="vlm_gym",
    version="0.1.0",
    packages=find_packages(),
    py_modules=['data_adapters', 'tasks'],
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "jsonlines>=3.1.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "ray[default]>=2.5.0",
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "transformers>=4.35.0",
        "rouge>=1.0.1",
        "nltk>=3.8.1",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "jupyter"],
        "distributed": ["ray[default]"],
    },
    python_requires=">=3.8",
)
