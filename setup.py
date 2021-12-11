import pathlib

from setuptools import setup, find_packages


setup(
    author='Anthony Raimondo',
    author_email='anthonyraimondo7@gmail.com',
    description='pandas extension for DataFrame manipulation, statistics, and plotting',
    install_requires=["pandas", "scikit-learn"],
    license='MIT',
    long_description=(pathlib.Path(__file__).resolve().parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    name='pandaSuit',
    package_dir={'': 'src/main/python'},
    packages=find_packages(where='src/main/python', exclude='test'),
    url='https://github.com/AnthonyRaimondo/pandaSuit',
    version='1.1.5'
)
