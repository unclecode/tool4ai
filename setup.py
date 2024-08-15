# File: setup.py

import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the project version from package __init__.py
def read_version():
    with open(os.path.join('tool4ai', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    raise RuntimeError('Unable to find version string.')

setup(
    name='tool4ai',
    version=read_version(),
    author='unclecode',
    author_email='unclecode@kidocode.com',
    description='Tool4AI: A model agnostic, LLM friendly router for tool/function call',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/unclecode/tool4ai',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=2.0.1',
        'litellm>=1.43.4',
        'pyyaml>=5.3',
        # Add any other core dependencies your project needs
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9',
            'black>=21.5b1',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx_rtd_theme>=0.5',
        ],
    },
    include_package_data=True,
    keywords='tool router llm ai function call',
    project_urls={
        'Bug Reports': 'https://github.com/unclecode/tool4ai/issues',
        'Source': 'https://github.com/unclecode/tool4ai/',
        'Documentation': 'https://tool4ai.readthedocs.io/',
    },
)