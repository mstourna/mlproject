from setuptools import setup, find_packages

def load_requirements(filename='requirements.txt'):
    with open(filename) as f:
        lines = f.readlines()
    # Exclude -e . and comments/empty lines
    requirements = [
        line.strip() for line in lines
        if line.strip() and not line.startswith('#') and not line.startswith('-e')
    ]
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    description='A Python project that does awesome things',
    author='Maria',
    url='https://github.com/mstourna/mlproject',
    packages=find_packages(),
    install_requires=load_requirements())