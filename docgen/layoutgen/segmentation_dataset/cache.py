from collections import OrderedDict
import random

class Cache:
    def __init__(self, capacity=200, replace_after=10, min_count_before_random=10):
        """ It's important that, for small datasets, we only load the item once
            Otherwise, we want to gradually fill up the cache
            So when starting out, we randomly alternate between filling and randomly selecting
            To send a "fill" signal, we just return None
            Consider maybe passing in the filler function to fill it up?

        Args:
            capacity:
            replace_after:
        """
        self.capacity = capacity
        self.replace_after = replace_after
        self.cache = OrderedDict()  # Stores items in (key, (value, count)) format
        self.hits = {}  # Tracks how many times an item is accessed
        self.min_count_before_random = min_count_before_random

    def get(self, key, random_ok=True):
        if key not in self.cache and random_ok and ( len(self.cache) >= self.capacity or len(self) > self.min_count_before_random and random.random() < 0.5):
            # use the cache randomly if it is FULL OR if it is not empty and the random number is less than 0.5
            key = random.choice(list(self.cache.keys()))
        return self._get_from_cache(key)

    def _get_from_cache(self, key):
        if key in self.cache:
            value, count = self.cache[key]
            self.cache[key] = (value, count + 1)
            if count >= self.replace_after:
                del self.cache[key]
            return value
        return None

    def put(self, key, value):
        if key not in self.cache:
            self.cache[key] = (value, 1)  # Add new item with a usage count of 1

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)