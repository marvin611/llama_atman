# tests/test_rag_explanations.py
import pytest
import torch
from llama import Llama
from rag import RAG
from document_qa_explainer import DocumentQAExplainer
from tests.test_cases import TestCase, TestResult
import time
import streamlit as st

# tests/test_rag_explanations.py
def test_rag_with_explanations(test_cases):
    """Test generation with RAG and explanations"""
    results = []
    
    for case in test_cases:
        test_case = TestCase(**case)
        
        # Initialize model and RAG
        model = Llama(model_name="TinyLlama-1.1B", device='cpu')
        model.setup()
        
        rag = RAG(
            model_handler=model,
            device='cpu',
            web_search_enabled=True,
            bing_subscription_key=st.secrets.get("BING_SUBSCRIPTION_KEY")
        )
        
        # Set seed for reproducibility
        torch.manual_seed(test_case.seed)
        
        start_time = time.time()
        
        # Get context
        result = rag.retrieve_context(
            query=test_case.prompt,
            top_k=2,
            web_only=True
        )
        
        # Generate response
        response = rag.generate_response(
            query=test_case.prompt,
            context="\n\n".join(result.contexts) if result.contexts else None,
            seed=test_case.seed,
            temperature=test_case.temperature,
            max_new_tokens=test_case.max_new_tokens
        )
        
        # Generate explanations
        explanations = []
        if result.contexts:
            explainer = DocumentQAExplainer(
                model=model,
                document="\n\n".join(result.contexts),
                explanation_delimiters=['.', '?', '!'],
                device='cpu',
                suppression_factor=0.5
            )
            
            output = explainer.run(
                question=test_case.prompt,
                expected_answer=response
            )
            
            if output:
                postprocessed = explainer.postprocess(output)
                explanations = [chunk['chunk'] for chunk in sorted(
                    postprocessed.data,
                    key=lambda x: x['value'],
                    reverse=True
                )[:2]]
        
        execution_time = time.time() - start_time
        
        results.append(TestResult(
            model_name="TinyLlama-1.1B",
            prompt=test_case.prompt,
            response=response,
            context="\n\n".join(result.contexts) if result.contexts else None,
            explanations=explanations,
            sources=result.sources,
            execution_time=execution_time
        ))
    
    return results