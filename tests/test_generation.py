# tests/test_generation.py
import pytest
import torch
from llama import Llama
from tests.test_cases import TestCase, TestResult
import time


# tests/test_generation.py
def test_basic_generation(test_cases):
    """Test basic generation without RAG or explanations"""
    results = []
    
    for case in test_cases:
        test_case = TestCase(**case)
        
        # Initialize model
        model = Llama(model_name="TinyLlama-1.1B", device='cpu')
        model.setup()
        
        # Set seed for reproducibility
        torch.manual_seed(test_case.seed)
        
        # Generate response
        start_time = time.time()
        response = model.generate(
            test_case.prompt,
            max_new_tokens=test_case.max_new_tokens,
            temperature=test_case.temperature,
            seed=test_case.seed
        )
        execution_time = time.time() - start_time
        
        results.append(TestResult(
            model_name="TinyLlama-1.1B",
            prompt=test_case.prompt,
            response=response,
            execution_time=execution_time
        ))
    
    return results