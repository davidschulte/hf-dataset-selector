from setuptools import find_packages, setup

with open("src/README.md", "r") as f:
    long_description = f.read()

setup(
    name="hfselect",
    version="0.0.1",
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
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)