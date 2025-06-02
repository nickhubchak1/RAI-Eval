# === config.py ===
# Add your actual keys (or load from environment variables). Never commit real keys to Git.

import os

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "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")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "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")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "AIzaSyCxuAlu2_1J0GIPVDpF1H-h8biP4EQDUL4")

# Which Gemini model to call (e.g., "chat-bison-001" or whatever is current)
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "chat-bison-001")

# Toggle local HF models on or off
USE_LOCAL_MODELS  = False #os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

# Toggle Grok (if you ever add Grok support); defaults to False
USE_GROK          = os.getenv("USE_GROK", "false").lower() == "true"

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
