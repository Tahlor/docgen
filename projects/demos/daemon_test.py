import multiprocessing
import random
import string
from tqdm import tqdm
from torch.utils.data import DataLoader
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset, WikipediaWord
from projects.french_bmd.french_bmd_layoutgen import HWGenerator, parser, Daemon
#from projects.french_bmd.french_bmd_layoutgen import WordIterator
from textgen.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from time import sleep
import logging
logger = logging.getLogger()


args = """
  --config ./config/default.yaml 
  --count 20000
  --renderer novel
  --output /media/EVO970/data/synthetic/french_bmd/ 
  --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models 
  --wikipedia 20220301.fr
  --saved_hw_model IAM
  --hw_batch_size 64
  --workers 0
"""

opts = parser(args)

words_dataset = WikipediaEncodedTextDataset(
    use_unidecode=True,
    shuffle_articles=True,
    random_starting_word=True,
    dataset=load_dataset("wikipedia", opts.wikipedia)["train"],
    vocabulary=set(ALPHA_VOCABULARY),  # set(self.model.netconverter.dict.keys())
    exclude_chars="",
    symbol_replacement_dict={
        "}": ")",
        "{": "(",
        "]": ")",
        "[": "(",
        "–": "-",
        " ()": "",
        "\n": " "
    }
)
class WordIterator:
    def __init__(self, daemon):
        self.renderer_daemon = daemon

    def get_next_word_iterator(self):
        item = self.renderer_daemon.queue.get()
        while True:
            for i in range(len(item["text_list"])):
                if 0 in item["word_imgs"][i].shape:
                    continue  # kind of a bug, it's an empty image e.g. \n or something
                # yield item["word_imgs"][i], item["text_list"][i], item["author_id"][i]
                yield {"img": item["word_imgs"][i],
                       "text": item["text_list"][i],
                       "style": item["author_id"],
                       "text_decode_vocab": item["text_list_decode_vocab"][i]
                       }
            while True:
                try:
                    item = self.renderer_daemon.queue.get()
                    break
                except Exception as e:
                    sleep(.5)
                    logger.exception(e)
                    logger.exception("Timeout waiting for next item")
                    print("Timeoout waiting for next item")
                    continue

render_text_pair_gen = HWGenerator(next_text_dataset=words_dataset,
                                   batch_size=opts.hw_batch_size,
                                   model=opts.saved_hw_model,
                                   model_path=opts.saved_hw_model_folder,
                                   device=opts.device,
                                   style=opts.saved_hw_model,
                                   )

words_dataset = WikipediaWord(words_dataset,
                              process_fn=["lower"],
                              random_next_article=True, )

renderer_daemon = Daemon(
    render_text_pair_gen, buffer_size=5000,
)
renderer_daemon.start()
render_text_pair = WordIterator(renderer_daemon).get_next_word_iterator()

class DaemonDataset(Dataset):
    def __init__(self, daemon):
        self.daemon = daemon
        self.data = list()

    def __getitem__2(self, index):
        # Fetch the item from the buffer if it exists
        if index < len(self.data):
            return self.data[index]
        else:
            # If the requested index is out of range of the current buffer, fetch more data
            while index >= len(self.data):
                self.data.append(next(self.daemon))
            return self.data[index]

    def __getitem__(self, index):
        item = self.daemon.queue.get()
        while True:
            for i in range(len(item["text_list"])):
                if 0 in item["word_imgs"][i].shape:
                    continue  # kind of a bug, it's an empty image e.g. \n or something
                # yield item["word_imgs"][i], item["text_list"][i], item["author_id"][i]
                yield {"img": item["word_imgs"][i],
                       "text": item["text_list"][i],
                       "style": item["author_id"],
                       "text_decode_vocab": item["text_list_decode_vocab"][i]
                       }
            while True:
                try:
                    item = self.daemon.queue.get()
                    break
                except Exception as e:
                    sleep(.5)
                    logger.exception(e)
                    logger.exception("Timeout waiting for next item")
                    print("Timeoout waiting for next item")
                    continue

    def __len__(self):
        # The length is undefined in the case of a generator
        return len(self.data)  # or some predefined large number if you know the maximum possible size


# Initialize your dataset
print("Initializing dataset...")
daemon_dataset = DaemonDataset(render_text_pair)

# Initialize your DataLoader
print("Initializing dataloader...")
daemon_loader = DataLoader(daemon_dataset, batch_size=1, num_workers=opts.workers)  # Adjust batch_size as needed

# Iterate over data
print("Iterating over data...")
for data in tqdm(daemon_loader):
    # Process your data here
    print(data)
