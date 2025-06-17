from setuptools import setup, find_packages

setup(
    name='my_project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'sax',
        'scipy',
        'slycot',
        'control'  
    ],
    author='Srinivasan Ashwyn Srinivasan',
    author_email='ashwyn@lightmatter.co',
    description='Congo Laser module repository',
    url='https://github.com/lightmatter-ai/congo-clm',
)