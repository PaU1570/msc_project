"""Setup.py for `msc_project`."""

import os

from setuptools import setup, find_packages

def get_version() -> str:
    """Get the package version."""
    version_path = os.path.join(os.path.dirname(__file__), "src", "msc_project", "VERSION.txt")
    with open(version_path, encoding="utf-8") as version_file:
        return version_file.read().strip()
    
def get_long_description() -> str:
    """Get the package long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read().strip()
    
setup(
    name="msc_project",
    version=get_version(),
    description="A project to explore the use of ferroelectric devices for in-memory computing",
    long_description=get_long_description(),
    author="Paul Uriarte Vicandi",
    author_email="puriarte@ethz.ch",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"msc_project": ["VERSION.txt"]},
    license="MIT",
    url="https://github.com/PaU1570/msc_project",
    python_requires=">=3.7",
)

