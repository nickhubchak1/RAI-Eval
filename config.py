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
