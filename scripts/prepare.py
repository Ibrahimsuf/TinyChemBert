from ChemBert.utils import create_simple_smiles, split_train_test_val
import argparse
import os

def prepare(all_smiles_file: str):
  os.makedirs("data", exist_ok=True)
  create_simple_smiles(all_smiles_file, "data/simple_smiles.txt")
  split_train_test_val("data/simple_smiles.txt", "data/train.txt", "data/test.txt", "data/val.txt", 0.8, 0.1)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("all_smiles_file", type=str)
  args = parser.parse_args()
  prepare(args.all_smiles_file)