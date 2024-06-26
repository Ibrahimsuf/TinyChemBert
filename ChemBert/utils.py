from datasets import load_dataset
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from torch.utils.data import DataLoader
from typing import Optional
import random
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
def split_train_test_val(examples_file: str, train_file: str, test_file: str, val_file: str, train_split: float, val_split: float) -> None:
  """Split the train file into train, test, and val files"""
  if train_split + val_split > 1:
    raise ValueError("train_split + val_split must be less than 1")
  
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

class GetDataLoader:
  def __init__(self, file:str) -> None:
    self.file = file
    self.dataset = load_dataset('text', data_files=file, streaming=True)
    self.tokenizer = SmilesTokenizer("small_vocab.txt")
  def tokenize_function(self, example):
    return self.tokenizer(example['text'], truncation=True, padding='max_length', max_length=20)
  def collate_fn(self, examples):
    batch = self.tokenizer.pad(examples, return_tensors="pt")
    return batch
  def get_data_loader(self, batch_size:Optional[int] = 32) -> DataLoader:
    tokenized_dataset = self.dataset.map(self.tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    # load_dataset assumes all the data is training data so we have to index into train data to get all data
    return DataLoader(tokenized_dataset["train"], collate_fn=self.collate_fn, batch_size=batch_size)