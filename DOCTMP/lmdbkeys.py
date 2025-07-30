import lmdb

def list_last_lmdb_keys(lmdb_path, prefix_filter=None, last_n=50):
    """
    List the last N keys in an LMDB database, optionally filtering by prefix.

    Args:
        lmdb_path (str): Path to the LMDB directory.
        prefix_filter (str, optional): If set, only keys starting with this prefix will be included.
        last_n (int): Number of last keys to display.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    keys = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, _ in cursor:
            key = key_bytes.decode('utf-8')
            if prefix_filter is None or key.startswith(prefix_filter):
                keys.append(key)

    keys.sort()  # Sort lexicographically
    last_keys = keys[:last_n]

    print(f"Total matching keys: {len(keys)} | Showing last {last_n} keys:\n")
    for key in last_keys:
        print(key)

# === Replace this with your actual LMDB path ===
lmdb_test_path = 'DocTamperV1-TestingSet'

# List last 50 keys starting with "image-"
list_last_lmdb_keys(lmdb_test_path, prefix_filter='image-', last_n=50)
