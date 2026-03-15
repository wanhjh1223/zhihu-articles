from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-vlm-framework",
    version="0.1.0",
    author="AI Team",
    description="A complete training framework from data collection to LLM and VLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8", "mypy"],
        "flash": ["flash-attn>=2.3.0"],
    },
    entry_points={
        "console_scripts": [
            "llm-train=llm_training.cli:main",
            "vlm-train=vlm_training.cli:main",
            "data-collect=data_collection.cli:main",
        ],
    },
)
