"""
Setup script for XAIDroid package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="AIdroid",
    version="1.0.0",
    author="AIDroid Team",
    author_email="xaidroid@example.com",
    description="Explainable Android Malware Detection using Graph Attention Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rigzen45/aidroid",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "dash>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xaidroid-preprocess=scripts.preprocess_apks:main",
            "xaidroid-train-gat=scripts.train_gat:main",
            "xaidroid-train-gam=scripts.train_gam:main",
            "xaidroid-inference=scripts.inference:main",
            "xaidroid-evaluate=scripts.evaluate:main",
            "xaidroid-visualize=scripts.visualize_attention:main",
        ],
    },
    include_package_data=True,
    package_data={
        "xaidroid": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
)