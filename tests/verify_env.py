import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import load_config

def test_config_loading():
    print("Testing Configuration Loading...")
    
    # Ensure we are in the right directory
    if not os.path.exists('config/settings.yaml'):
        print("Error: Run this script from the project root.")
        return

    try:
        config = load_config()
        success = True

        def mask(value):
            if value is None:
                return "<missing>"
            value = str(value)
            if len(value) <= 4:
                return "*" * len(value)
            return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"

        api_key = config['exchange']['api_key']
        base_currency = config['swarm']['base_currency']
        redis_host = config['redis']['host']
        
        print(f"Exchange Name: {config['exchange']['name']}")
        print(f"API Key (resolved): {mask(api_key)}")
        print(f"Base Currency: {base_currency}")
        print(f"Redis Host: {redis_host}")
        
        # Check against .env values
        if api_key and not str(api_key).startswith("${"):
            print("✅ API Key resolved correctly from .env (or defaults)")
        else:
            success = False
            print("❌ API Key unresolved; check .env or config placeholders")
        
        if base_currency:
            print("✅ Base Currency resolved correctly")
        else:
            success = False
            print("❌ Base Currency unresolved; check .env or config placeholders")
            
        if success:
            print("\nAll checks passed!")
        else:
            print("\nSome checks failed.")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")

if __name__ == "__main__":
    test_config_loading()
