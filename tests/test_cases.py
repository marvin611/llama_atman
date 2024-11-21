# tests/test_cases.py
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class TestResult:
    model_name: str
    prompt: str
    response: str
    context: Optional[str] = None
    explanations: Optional[List[str]] = None
    sources: Optional[List[Dict]] = None
    execution_time: float = 0.0

class TestCase:
    def __init__(
        self,
        prompt: str,
        seed: int = 42,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ):
        self.prompt = prompt
        self.seed = seed
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens