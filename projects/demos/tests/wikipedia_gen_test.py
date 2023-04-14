from pathlib import Path
from hwgen.data.hw_generator import HWGenerator
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset
from textgen.basic_text_dataset import VOCABULARY
from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.fr")["train"]

basic_text_dataset = WikipediaEncodedTextDataset(
    dataset=dataset,
    vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
    exclude_chars="",
    use_unidecode=True,
    min_chars=8,
    max_chars=16,
    decode_vocabulary="default_expanded" ,
)

hw = HWGenerator(
    next_text_dataset=basic_text_dataset,
    batch_size=24,
    model="IAM",
    device="cuda",
    data_split="all",
)
a = []
while True:
    m = next(hw)
    a.append(m)
    if len(a) > 10:
        break

for i in hw:
    a.append(i)
    if len(a) > 20:
        break


input()