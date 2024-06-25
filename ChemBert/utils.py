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