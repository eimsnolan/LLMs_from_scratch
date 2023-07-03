# Wiki ga irish dataset (also hosted by HuggingFace)
# website url: https://huggingface.co/datasets/wikipedia
from datasets import load_dataset

wiki_irish = load_dataset(
    "wikipedia", language="ga", date="20230620", beam_runner="DirectRunner"
)

for d in wiki_irish:
    print(d)  # prints documents
