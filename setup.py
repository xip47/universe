from setuptools import setup, find_packages

setup(
    name="universe",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
    include_package_data=True,
    install_requires=[
        "numpy>=2.2.0",
        "cupy-cuda12x>=13.4.0",
        "scipy>=1.13.0",
        "matplotlib>=3.10",
        "PyQt5>=5.15.0",
        "black>=25.0",
        "flake8>=7.0",
        "mypy>=1.16.0",
        "pytest>=8.0",
        "pytest-cov>=6.0"
    ],
    entry_points={
        "console_scripts": [
            "universe = universe.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
