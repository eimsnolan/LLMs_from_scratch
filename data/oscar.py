# example with OSCAR
# https://huggingface.co/datasets/oscar

from datasets import load_dataset

oscar_irish = load_dataset("oscar", "unshuffled_deduplicated_ga")


for d in oscar_irish:
    print(d[0])  # prints documents
