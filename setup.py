"""Setup script for AIDroid-x package
This version:
 - Uses the distribution name "AIDroid-x"
 - Dynamically finds packages under src/
 - Builds package_data mapping programmatically (uses the primary package if present)
 - Builds safe console_scripts entry_points only if a package is found (assumes scripts live in <package>.scripts)
 - Keeps legacy behavior for README and requirements reading
"""
from setuptools import setup, find_packages
from pathlib import Path
from typing import List, Dict

HERE = Path(__file__).parent

# Read README
readme_path = HERE / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = HERE / "requirements.txt"
requirements: List[str] = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

# Discover packages under src/
packages = find_packages(where="src")
package_dir = {"": "src"}

# Determine a primary package to attach package_data and console scripts.
# Prefer 'xaidroid' if present, otherwise first discovered package.
primary_pkg = None
if "xaidroid" in packages:
    primary_pkg = "xaidroid"
elif packages:
    primary_pkg = packages[0]

# Configure package_data: include config files if package exists
package_data: Dict[str, List[str]] = {}
if primary_pkg:
    package_data = {
        primary_pkg: ["config/*.yaml", "config/*.yml", "config/*.json"]
    }

# Configure console_scripts entry points if we can resolve package-based scripts.
# Note: For these entry points to work, move the existing top-level scripts into
# src/<primary_pkg>/scripts/ and expose a main() in each script module.
console_scripts = []
if primary_pkg:
    console_scripts = [
        "xaidroid-preprocess={pkg}.scripts.preprocess_apks:main".format(pkg=primary_pkg),
        "xaidroid-train-gat={pkg}.scripts.train_gat:main".format(pkg=primary_pkg),
        "xaidroid-train-gam={pkg}.scripts.train_gam:main".format(pkg=primary_pkg),
        "xaidroid-inference={pkg}.scripts.inference:main".format(pkg=primary_pkg),
        "xaidroid-evaluate={pkg}.scripts.evaluate:main".format(pkg=primary_pkg),
        "xaidroid-visualize={pkg}.scripts.visualize_attention:main".format(pkg=primary_pkg),
    ]

setup(
    name="AIDroid-x",
    description="Explainable Android Malware Detection using Graph Attention Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rigzen-45/AIDroid-x",
    packages=packages,
    package_dir=package_dir,
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
    entry_points={"console_scripts": console_scripts} if console_scripts else {},
    include_package_data=True,
    package_data=package_data,
    zip_safe=False,
)
