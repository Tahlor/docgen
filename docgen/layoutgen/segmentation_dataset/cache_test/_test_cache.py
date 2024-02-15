from docgen.layoutgen.segmentation_dataset.cache import Cache
import random
from collections import OrderedDict

def generate_fake_data(num_items):
    """Generate fake data as key-value pairs."""
    return {f"key_{i}": f"value_{i}" for i in range(num_items)}

def evaluate_cache_behaviour():
    num_unique_items = 200  # Number of unique items to generate
    cache_capacity = 300  # Cache capacity, larger than the number of unique items

    # Generate fake data
    data = generate_fake_data(num_unique_items)

    # Initialize the cache
    cache = Cache(capacity=cache_capacity, replace_after=1000000, min_count_before_random=10)

    # Populate the cache
    for i in range(50000):
        for key, value in data.items():
            return_value = cache.get(key)
            if return_value is None:
                cache.put(key, value)

    # Verify cache behavior
    print(f"Cache Size: {len(cache)} (Expected: {num_unique_items})")
    for key in data.keys():
        value = cache.get(key)
        assert value is not None, f"Value for {key} should not be None"

    # Verify cache has not filled up beyond its capacity and no item is erased
    assert len(cache) <= cache_capacity, "Cache size should not exceed its capacity"
    assert len(cache) == num_unique_items, "Cache should contain exactly the number of unique items"
    print("Cache behavior verified: All items are correctly stored and accessible.")


# Run the test
evaluate_cache_behaviour()
