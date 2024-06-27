from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForCausalLM
from ChemBert.utils import get_data_loader
import argparse
from typing import Optional
import sys
from tqdm import tqdm
MODEL = "roneneldan/TinyStories-1M"
def train(train_directory: str, val_directory: str, save_directory: str, num_epochs: Optional[int] = 10, tensorboard_logdir: Optional[str] = None, checkpoint: Optional[str] = None) -> None:
  """Train the model"""
  if checkpoint is None:
    config = AutoConfig.from_pretrained(MODEL, use_cache=False)
    model = AutoModelForCausalLM.from_config(config=config)
  else:
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
  
  train_loader, tokenizer = get_data_loader(train_directory, batch_size=32)
  val_loader, _ = get_data_loader(val_directory)
  model.resize_token_embeddings(len(tokenizer))


  optimizer = optim.Adam(model.parameters(), lr=3e-4)
  # Training loop
  global_step = 0
  if tensorboard_logdir is not None:
    writer = SummaryWriter(tensorboard_logdir)
  for _ in range(num_epochs):
    for train_batch in tqdm(train_loader):
      optimizer.zero_grad()
      train_loss = model(**train_batch).loss
      train_loss.backward()
      optimizer.step()

      if tensorboard_logdir is not None:
        writer.add_scalar("Loss/train", train_loss, global_step)
      else:
        print(f"Global step {global_step}: train_loss = {train_loss} val_loss = {val_loss}")
      global_step += 1

    # Validation loop
    val_loss = 0
    num_val_batches = 0
    for val_batch in val_loader:
      with torch.no_grad():
        num_val_batches += 1
        val_loss += model(**val_batch).loss
    val_loss /= num_val_batches

    if tensorboard_logdir is not None:
      writer.add_scalar("Loss/val", val_loss, global_step)

  if tensorboard_logdir is not None:
    writer.flush()
    writer.close()

  model.save_pretrained(save_directory)
  tokenizer.save_pretrained(save_directory)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("train_directory", type=str)
  parser.add_argument("val_directory", type=str)
  parser.add_argument("save_directory", type=str)
  parser.add_argument("--num_epochs", type=int, default=10)
  parser.add_argument("--tensorboard_logdir", type=str, default=None)
  parser.add_argument("--checkpoint", type=str, default=None)
  args = parser.parse_args()

  train(args.train_directory, args.val_directory, args.save_directory, args.num_epochs, args.tensorboard_logdir, args.checkpoint)

