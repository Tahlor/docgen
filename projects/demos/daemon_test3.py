import time
import multiprocessing
import random
import string
import sys
import threading
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
import queue

class Generator:
    def __init__(self, num_items=sys.maxsize):
        self.num_items = num_items

    def __iter__(self):
        while True:
            text_list = [''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) for _ in range(10)]
            word_imgs = [random.randint(10, 10) for _ in range(10)]  # random 10x10 images
            author_id = random.randint(0, 10)
            text_list_decode_vocab = [random.randint(0, 10) for _ in range(10)]
            item = {"text_list": text_list,
                    "word_imgs": word_imgs,
                    "author_id": author_id,
                    "text_list_decode_vocab": text_list_decode_vocab}
            #time.sleep(1)
            yield item

class Daemon(threading.Thread):
    def __init__(self, data_iterator=None, buffer_size=5000):
        print("NEW DAEMON")
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        if data_iterator is None:
            self.data_iterator = Generator()
        else:
            self.data_iterator = data_iterator

    def run(self):
        # This is the function that runs in the background thread.
        for item in self.data_iterator:
            # Add the item to the queue. This will block if the queue is full.
            self.queue.put(item)
            if self.stop_event.is_set():
                return

    def stop(self):
        self.stop_event.set()


# Initialize your daemon
if True:
    daemon = Daemon(buffer_size=1000)
    daemon.start()

class DaemonDataset(Dataset):
    def __init__(self, daemon):
        self.daemon = daemon
        self.data = list()
        self.timeout = 1

    def __getitem__(self, index):
        # Fetch the item from the buffer if it exists
        while True:
            try:
                return self.daemon.queue.get(block=False, timeout=self.timeout)
            except queue.Empty:
                pass

    def __len__(self):
        # The length is the current size of the buffer plus the size of the queue
        return sys.maxsize



# Initialize your dataset
daemon_dataset = DaemonDataset(daemon)

# Initialize your DataLoader
daemon_loader = DataLoader(daemon_dataset, batch_size=1, num_workers=2)  # Adjust batch_size as needed

# Iterate over data
for data in tqdm(daemon_loader):
    # Process your data here
    #print(data)
    pass

daemon.stop()
