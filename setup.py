from setuptools import find_packages, setup

# --- DEPENDENCY MANAGEMENT ---
# We read the 'requirements.txt' file to avoid listing dependencies manually twice.
# This ensures that 'pip install .' and 'pip install -r requirements.txt' remain consistent.
with open("requirements.txt") as f:
    content = f.readlines()

# List Comprehension for Cleaning:
# 1. x.strip(): Removes newlines (\n) and spaces from each line.
# 2. if "git+" not in x: Excludes dependencies installed directly from Git repositories.
#    (Standard 'install_requires' often struggles with git URLs; these should be handled separately).
requirements = [x.strip() for x in content if "git+" not in x]

# --- PACKAGE CONFIGURATION ---
setup(
    # The name of the package as it will appear in pip (e.g., pip install project_accidents_package)
    name='project_accidents_package',

    # Semantic versioning (Major.Minor.Patch)
    version="0.0.1",

    # Defines the dependencies required to run this package.
    # When someone installs this package, these libraries will be installed automatically.
    install_requires=requirements,

    # Automatically finds all directories containing an '__init__.py' file.
    # This includes your source code folder 'project_accidents_package' without needing to list files manually.
    packages=find_packages()
)
