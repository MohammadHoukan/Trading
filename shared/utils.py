import hashlib

def get_key_hash(api_key):
    """Generate a consistent hash for an API key to use as a lock ID."""
    return hashlib.md5(api_key.encode()).hexdigest()
