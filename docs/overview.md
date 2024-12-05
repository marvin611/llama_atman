# Llama-Atman docs

## System Overview

### Core Components
1. **Attention Manipulation**
   - Direct modification of attention scores
   - Support for both pre and post-scaling manipulation
   - Compatible with various attention implementations

2. **Suppression Mechanisms**
   - Token-level suppression
   - Sentence-wise suppression
   - Conceptual suppression based on embedding similarity

3. **Explanation Systems**
   - Document QA explanation
   - Attention-based explanations
   - Cross-entropy analysis

4. **Interface and Integration**
   - User interface for model interaction
   - RAG integration
   - LLaMA model wrapper

## File Structure

### 1. Model Architecture
- [`modeling_llama.py`](atman_docs/modeling_llama.md)
  - Modified LLaMA architecture
  - Attention manipulation integration

- [`configuration_llama.py`](atman_docs/configuration_llama.md)
  - Model configuration
  - Custom parameters

### 2. Attention Manipulation
- [`manipulate_attention.py`](atman_docs/manipulate_attention.md)
  - Base attention manipulation functions
  - Score modification utilities

- [`conceptual_suppression.py`](atman_docs/conceptual_suppression.md)
  - Similarity-based suppression
  - Embedding analysis

- [`sentence_wise_suppression.py`](atman_docs/sentence_wise_suppression.md)
  - Text chunking
  - Suppression configuration

### 3. Explanation
- [`explainer.py`](atman_docs/explainer.md)
  - Base explanation framework
  - Attention analysis
  - Token importance calculation

- [`doc_qa_explainer.py`](atman_docs/doc_qa_explainer.md)
  - Document QA-specific explanations
  - Context relevance analysis
  - Answer attribution

- [`logit_parsing.py`](atman_docs/logit_parsing.md)
  - Delta logit calculations
  - Cross-entropy analysis

- [`outputs.py`](atman_docs/outputs.md)
  - Output data structures
  - Visualization methods

### 4. Interface and Integration
- [`interface.py`](atman_docs/interface.md)
  - User interface
  - Input/output handling

- [`rag.py`](atman_docs/rag.md)
  - RAG & document processing

- [`llama_wrapper.py`](atman_docs/llama_wrapper.md)
  - wrapper for custom LLaMA model

### 5. Utilities
- [`utils.py`](atman_docs/utils.md)
  - Helper functions
  - File operations


## Recommended Reading Order
1. System overview and architecture
2. Core manipulation components
3. Explanation systems:
   - Base explainer functionality
   - Document QA explanations
4. Interface and integration

