# %%
from operator import pos
from random import shuffle
import tiktoken

text_filepath = "../tokenizers/the-verdict.txt"

tokenizer = tiktoken.get_encoding("gpt2")

with open(text_filepath, "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
# %% Create input output pairs for llm pretraining with the encoding sample

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

# %% Get the x and y pairs for training

for i in range(1, context_size):
    print(f"x: {x[:i]}")
    print(f"y: {x[i]}")
    print("................")
# %%

for i in range(1, context_size):
    print(f"x: {tokenizer.decode(enc_sample[:i])}")
    print(f"y: {tokenizer.decode([enc_sample[i]])}")
    print("................")
# %% Pytorch Dataset

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDdatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenizes the entire text
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

# %% Create the dataloader to load this dataset

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDdatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

# %%

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, shuffle=False, stride=1)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# %% Initializing an embedding layer

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(f"Embedding Vector of 3: \n{embedding_layer(torch.tensor([3]))}")
# %% Positional Embeddings

# When I think about position embeddings I start to think about the relative positional embeddings.
# Things such as the cosine and sine based embeddings.
# However, if we use absolute embeddings for positions and use a learned position embeddigings that also works.

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                  stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"token ids: {inputs}")
print(f"inputs shape = {inputs.shape}")

token_embeddings = token_embedding_layer(inputs)
print(f"token embeddings shape: {token_embeddings.shape}")

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"pos embeddings shape: {pos_embeddings.shape}")

input_embeddings = token_embeddings + pos_embeddings

print(f"input_embeddings = token_embeddings + position embeddings \n{input_embeddings.shape} = {token_embeddings.shape} + {pos_embeddings.shape}")

# %%
