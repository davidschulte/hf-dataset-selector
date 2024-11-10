from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="hf-dataset-selector",
    version="0.1.0",
    description="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidschulte/hf-dataset-selector",
    author="David Schulte",
    author_email="davidsiriusschulte@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bson >= 0.5.10",
        "auto_mix_prep == 0.2.0",
        "datasets == 3.0.2",
        "huggingface_hub == 0.26.1",
        "numba == 0.57.1",
        "numpy == 1.24.3",
        "torch == 2.1.2",
        "tqdm == 4.66.5",
        "transformers == 4.32.1"
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)