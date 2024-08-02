import os
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
local_dir = "financial_transcripts"
shard_size = int(1e3)  # 100M tokens per shard

# Create the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

# Get the EOT token ID
eot_token_id = tokenizer.eos_token_id


def tokenize(text):
    # Tokenizes a single text and returns a numpy array of uint16 tokens with EOT at the beginning
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens_with_eot = [eot_token_id] + tokens
    tokens_np = np.array(tokens_with_eot, dtype=np.uint16)
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# Read the input file
with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Split the text into chunks (you may need to adjust this based on your file structure)
chunks = [line.strip() for line in lines if line.strip()]

# Set up multiprocessing
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, chunks, chunksize=16):
        # Is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # Simply append tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"financial_transcripts_{split}_{shard_index:06d}")
            # Split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # Populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"financial_transcripts_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print("Processing complete!")
