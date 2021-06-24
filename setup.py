from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'torch<1.7',
    'six',
    'matplotlib',
    'scikit-learn',
    'reformer-pytorch',
    'pytorch3d'
]

setup(
    name='Polygen-PyTorch',
    version='1.0',
    description='PyTorch Implementation of PolyGen: An Autoregressive Generative Model of 3D Meshes.',
    author='JS/PS/IK',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
