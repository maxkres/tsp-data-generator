# File: setup.py

from setuptools import setup, find_packages

setup(
    name='adaptive_tsp',
    version='0.1.0',
    description='TSP solver with adaptive SOM-like grid and iterative refinement',
    author='Your Name',
    packages=find_packages(exclude=['tests', 'benchmarks', 'data_generator']),
    install_requires=[
        'numpy>=1.15',
        'matplotlib>=3.0',
        'scipy>=1.1',
        'networkx>=2.5'
    ],
    entry_points={
        'console_scripts': [
            'tsp-gen = tsp_generator.cli:main',
            'adapt-tsp = adaptive_tsp.adaptive_cli:main'
        ],
    },
)
