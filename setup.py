from setuptools import setup, find_packages

setup(
  name="ChemBert",
  version="0.1",
  packages=find_packages(),
  install_requires=["transformers", "datasets", "torch", "tqdm", "deepchem", "numpy==1.24"] #deepchem does not support numpy 2.0
)