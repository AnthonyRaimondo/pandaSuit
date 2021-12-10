import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).resolve().parent

# The text of the README file is used as a description
README = (HERE / "README.md").read_text()

setup(
    name='pandaSuit',
    version='0.0.7',
    packages=find_packages(where='src/main/python', exclude='test'),
    package_dir={'': 'src/main/python'},
    url='https://github.com/AnthonyRaimondo/pandaSuit',
    license='MIT',
    author='Anthony Raimondo',
    author_email='anthonyraimondo7@gmail.com',
    description='pandas wrapper and extension for DataFrame manipulation, statistics, and plotting',
    install_requires=["pandas", "scikit-learn"],
    long_description=README,
    long_description_content_type="text/markdown",
)
