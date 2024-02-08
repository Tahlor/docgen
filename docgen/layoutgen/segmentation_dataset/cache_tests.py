from cache import Cache

# Create a cache instance with a small capacity for testing
cache = Cache(capacity=5, replace_after=3, min_count_before_random=2)

# Test case 1: Inserting elements and retrieving them before the cache is full
for i in range(5):
    cache.put(f"key{i}", f"value{i}")
    assert cache.get(f"key{i}") == f"value{i}", f"Test failed for key{i}"

# The cache should now be full. Let's verify that.
assert len(cache) == 5, "Cache should be full now."

# Test case 2: Accessing elements to trigger replacement policy
for _ in range(5):  # Accessing key0 more than replace_after times
    cache.get("key0")
# key0 should now be replaced, let's try retrieving it
value = cache.get("key0")
assert value is None or value != "value0", "key0 should have been deleted and returned a None or random value."

# Test case 3: Testing random selection logic
# As the cache is full, we'll attempt to access a non-existent key with random_ok=True
random_key_retrieval = cache.get("non_existent_key", random_ok=True)
assert random_key_retrieval is not None, "Expected a random value but got None."

# Inserting a new item should cause an older item (except recently accessed ones) to be evicted
cache.put("new_key", "new_value")
assert len(cache) <= 5, "Cache size should not exceed its capacity."
assert cache.get("new_key") == "new_value", "Failed to retrieve newly inserted item."

print("All test cases passed successfully!")
