# tests/conftest.py
import pytest
import streamlit as st
from typing import Dict, List

@pytest.fixture
def bing_key():
    return st.secrets.get("BING_SUBSCRIPTION_KEY")

TEST_CASES = [
    {
        "prompt": "What is quantum computing?",
        "seed": 42,
        "temperature": 0.7,
        "max_new_tokens": 256
    },
    {
        "prompt": "Explain the theory of relativity.",
        "seed": 42,
        "temperature": 0.7,
        "max_new_tokens": 256
    }
]

@pytest.fixture
def test_cases():
    return TEST_CASES