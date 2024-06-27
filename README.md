# TinyChemBert

Training a Chemical Language Model on SMILES of Molecules containing only Carbon, Nitrogen, Oxygen, Sulfur, Phosphorus, Flourine, Chlorine, Bromine, and Iodine.
The model uses the [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) architecture. The training process is taken from [ChemBERTa](https://arxiv.org/abs/2010.09885). The dataset is the same as the 10M dataset used in the paper except with all molecules not containing the listed atoms removed.


## Installation Instruction

1. `git clone https://github.com/Ibrahimsuf/TinyChemBert.git`
2. `cd TinyChemBert`
3. create conda environment `conda create --name TinyChemBert`
4. activate the environment `conda activate TinyChemBert`
5. `conda install pip`
6. `pip install .`
7. Download the [dataset](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pubchem_10m.txt.zip) and unzip it.
8. `python scripts/prepare.py path/to/pubchem_10m.txt path/to/output_dir`
9. make model output directory
10. `python scripts/train.py data/train.txt data/val.txt path/to/model_output_dir --num_epochs 10`

## Future Work

I plan to train and evaluate the model on downstream tasks.
