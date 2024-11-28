
from setuptools import setup, find_packages

setup(
    name='my_data_science_package',
    version='0.1.0',
    description='A collection of data science tools.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.20',
        'matplotlib>=3.0',
        'seaborn>=0.11',
        'scikit-learn>=0.24',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
