from hwgen.daemon import Daemon
import logging
from typing import Optional, Dict, Any
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class WordImgIterator:
    def __init__(self, input_data_iterator, buffer_size=4000, style="handwriting"):
        self.input_data_iterator = input_data_iterator
        self.buffer_size = buffer_size
        self.renderer_daemon = Daemon(self.input_data_iterator, buffer_size=self.buffer_size)
        self.timeout = .2
        self.renderer_daemon.start()
        self.word_gen = self.get_next_word_iterator()
        self.style = style

    def update_buffer_size(self, new_buffer):
        self.buffer_size = new_buffer
        self.renderer_daemon.buffer_size = new_buffer
        self.renderer_daemon.queue.maxsize = new_buffer

    def restart_daemon_if_needed(self):
        if not self.renderer_daemon.is_alive():
            print("Daemon thread has died, restarting...")
            old_daemon = self.renderer_daemon
            self.renderer_daemon = Daemon(self.input_data_iterator, buffer_size=self.buffer_size)
            self.renderer_daemon.start()
            old_daemon.join()

    def get_next_word_iterator(self):
        failures = 0
        while True:
            item = None
            try:
                item = self.renderer_daemon.queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                #logger.exception("Timeout waiting for next item")
                failures += 1
                if failures and failures % 10 == 0:
                    logger.info(f"Timeout waiting for next item: {failures}")
                    self.restart_daemon_if_needed()
                continue

            #print(item)
            if item is not None:
                failures = 0
                for i in range(len(item["text_list"])):
                    if 0 in item["word_imgs"][i].shape:
                        continue  # kind of a bug, it's an empty image e.g. \n or something

                    try:
                        yield {"img": item["word_imgs"][i],
                               "text": item["text_list"][i],
                               "style": item["author_id"],
                               "text_decode_vocab": item["text_list_decode_vocab"][i]
                               }
                    except Exception as e:
                        logger.exception(e)
                        continue

    def get_next_word_queue(self):
        while True:
            item = None
            try:
                item = self.renderer_daemon.queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.exception("Timeout waiting for next item")
                continue
            #print(item)
            if item is not None:
                for i in range(len(item["text_list"])):
                    if 0 in item["word_imgs"][i].shape:
                        continue  # kind of a bug, it's an empty image e.g. \n or something

                    try:
                        yield {"img": item["word_imgs"][i],
                               "text": item["text_list"][i],
                               "style": item["author_id"],
                               "text_decode_vocab": item["text_list_decode_vocab"][i]
                               }
                    except Exception as e:
                        logger.exception(e)
                        continue
    def stop(self):
        self.renderer_daemon.stop()
        self.renderer_daemon.join()

    def get(self, size=None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Safely retrieves the next item from the iterator.

        Returns:
            Optional[Dict[str, Any]]: The next item, or None if no item could be retrieved.
        """
        try:
            next_item = next(self.word_gen)
            if size:
                next_item["img"] = resize_to_height_numpy(next_item["img"], size)
            return next_item
        except StopIteration:
            print("The generator has been exhausted.")
            return None
        except Exception as e:
            print(f"An error occurred while retrieving the next item: {e}")
            return None

def resize_to_height_numpy(img, height):
    width = int(img.shape[1] * height / img.shape[0])
    return cv2.resize(img, (width, height))
