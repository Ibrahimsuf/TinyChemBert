import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from torch.utils.data import DataLoader
from typing import Optional
import random
from transformers import DataCollatorForLanguageModeling
from typing import Optional
allowed_atoms = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
def create_simple_smiles(all_smiles_file: str, simple_smiles_file: str) -> None:
  """Create a list of SMILES string with less than 20 characters and only allowed atoms"""
  with open(all_smiles_file, "r") as f, open(simple_smiles_file, "w") as out:
    for line in f:
      atoms = set([char for char in line if char.isalpha()])
      if len(line) < 20 and atoms.issubset(allowed_atoms):
        out.write(line)

def remove_unused_atoms_from_vocab_file(vocab_file: str, output_file: str) -> None:
  """Remove all words in the vocab file that contain atoms that are not allowed atoms"""
  with open(vocab_file, "r") as f, open(output_file, "w") as out:
    for _ in range(15):
      out.write(f.readline()) # The first 15 lines are special tokens
    for line in f:
      atoms = set([char for char in line if char.isalpha()])
      if atoms.issubset(allowed_atoms):
        out.write(line)
def split_train_test_val(examples_file: str, train_file: str, test_file: str, val_file: str, train_split: float, val_split: float, seed: Optional[int] = 0) -> None:
  """Split the train file into train, test, and val files"""
  if train_split + val_split > 1:
    raise ValueError("train_split + val_split must be less than 1")
  
  random.seed(0)
  num_lines = sum(1 for line in open(examples_file, "r"))
  num_train = int(num_lines * train_split)
  num_val = int(num_lines * val_split)
  permutation = random.sample(range(num_lines), num_lines)

  train_indexes = permutation[:num_train]
  val_indexes = permutation[num_train:num_train+num_val]
  test_indexes = permutation[num_train+num_val:]
  
  with open(examples_file, "r") as f, open(train_file, "w") as train, open(test_file, "w") as test, open(val_file, "w") as val:
    for i, line in enumerate(f):
      if i in train_indexes:
        train.write(line)
      elif i in val_indexes:
        val.write(line)
      elif i in test_indexes:
        test.write(line)

def get_data_loader(file:str, batch_size:Optional[int] = 32, mlm_probability:Optional[float] = 0.15) -> DataLoader:
  dataset = load_dataset('text', data_files=file, streaming=True)["train"]
  tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], return_special_tokens_mask=True), remove_columns=["text"])
  tokenizer = SmilesTokenizer("vocab/small_vocab.txt")
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability, return_tensors="pt")
  return DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=batch_size), tokenizer
