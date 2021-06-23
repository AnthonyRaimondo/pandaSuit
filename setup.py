from setuptools import setup, find_packages

setup(
    name='pandaSuit',
    version='0.0.3',
    packages=find_packages(where='src/main/python'),
    package_dir={'': 'src/main/python'},
    url='https://github.com/AnthonyRaimondo/pandaSuit',
    license='GNU',
    author='Anthony Raimondo',
    author_email='anthonyraimondo7@gmail.com',
    description='Helper functions for pandas'
)
