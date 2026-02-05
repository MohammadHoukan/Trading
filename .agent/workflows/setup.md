---
description: Install dependencies and configure the environment for Spot-Grid-Swarm
---

1. Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies
// turbo
```bash
pip install -r requirements.txt
```

3. Ensure Redis is installed and running
// turbo
```bash
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Redis is not running. Attempting to start..."
    redis-server --daemonize yes
else
    echo "Redis is already running."
fi
```

4. Check Configuration
   - Review `config/settings.yaml` to ensure API keys are set (or use Testnet).
   - Review `config/strategies.json` for trading pairs.
