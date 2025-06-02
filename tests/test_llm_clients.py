# === tests/test_llm_clients.py ===

import os
import pytest

from llm_clients.openai_client   import query_openai
from llm_clients.anthropic_client import query_claude
from llm_clients.gemini_client  import query_gemini
from llm_clients.local_client    import query_local_model
from config import USE_LOCAL_MODELS

@pytest.mark.parametrize("msg", ["Hello, how are you?"])
def test_openai_client(msg):
    resp = query_openai(msg, model="gpt-3.5-turbo", max_retries=1)
    assert isinstance(resp, str) and len(resp) > 0

@pytest.mark.parametrize("msg", ["What is 2+2?"])
def test_anthropic_client(msg):
    resp = query_claude(msg, model="claude-3-opus-20240229", max_retries=1)
    assert isinstance(resp, str) and len(resp) > 0

@pytest.mark.parametrize("msg", ["Tell me a joke."])
def test_gemini_client(msg):
    resp = query_gemini(msg, model=None, max_retries=1)
    assert isinstance(resp, str) and len(resp) > 0

@pytest.mark.skipif(not USE_LOCAL_MODELS, reason="Local models disabled in config.py")
def test_local_client(msg="Write 'test' backward."):
    resp = query_local_model(msg, model_key="llama3")
    assert isinstance(resp, str) and len(resp) > 0
