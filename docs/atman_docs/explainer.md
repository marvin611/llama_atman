# Explainer

A class for analyzing and explaining model outputs by manipulating attention scores based on a given suppression factor.

## Class Overview

### `Explainer`
Main class for generating explanations through attention manipulation.

#### Constructor Parameters
- `model` - Language model instance
- `device` - Computing device
- `tokenizer` - Model tokenizer
- `suppression_factor: float` - Factor for attention suppression (default: 0.1)
- `conceptual_suppression_threshold: float` - Threshold for conceptual grouping (default: 0.6)
- `modify_suppression_factor_based_on_cossim: bool` - Adjust suppression based on similarity (default: True)
- `multiplicative: bool` - Use multiplicative suppression (default: True)
- `do_log: bool` - Enable logging (default: False)
- `layers` - Specific layers to manipulate (default: None)
- `manipulate_attn_scores_after_scaling: bool` - When to apply manipulation (default: False)

## Key Methods

### `collect_logits_by_manipulating_attention(prompt: list, target: str, **kwargs)`
Main method for generating explanations through attention manipulation.

Example:
```python
results = explainer.collect_logits_by_manipulating_attention(
    prompt=["What is machine learning?"],
    target="Machine learning is...",
    max_batch_size=25
)
```

#### Parameters
- `prompt: list` - Input prompt(s)
- `target: str` - Target completion
- `prompt_explain_indices: list` - Tokens to suppress (optional)
- `max_batch_size: int` - Batch size for processing (default: 1)
- `configs: dict` - Custom suppression configuration (optional)
- `save_configs_as: str` - Path to save configurations (optional)
- `save_configs_only: bool` - Only save configs without running (default: False)

#### Custom Config Example
```python
custom_config = [
    {
        "suppression_token_index": [-1],
        "suppression_factor": [1.0]
    },
    {
        "suppression_token_index": [0, 1, 2, 3],
        "suppression_factor": [0.1, 0.1, 0.1, 0.1]
    }
]
```

## Internal Methods

### `collect_logits(batches: list, input_embeddings)`
Processes batches of inputs with attention manipulation.

### `compile_results_from_configs_and_logits(**kwargs)`
Compiles explanation results from model outputs.

### `convert_prompt_and_target_to_token_ids(prompt: str, target: str)`
Converts text inputs to token IDs.

### `get_default_configs_for_forward_passes(prompt_explain_indices, target_token_indices)`
Generates default suppression configurations.

## Output Structure

The explanation output includes:
- Original model logits
- Target token information
- Prompt length and indices
- Suppressed chunk logits
- Attention manipulation effects

## Dependencies

- `torch`
- `copy`
- `utils`: Utility functions for chunking and logit processing
- `conceptual_suppression`: Conceptual grouping functionality

## Usage Example

```python
# Initialize explainer
explainer = Explainer(
    model=llm_model,
    device="cuda",
    tokenizer=tokenizer,
    suppression_factor=0.1
)

# Generate explanation
results = explainer.collect_logits_by_manipulating_attention(
    prompt=["Explain the concept of gravity."],
    target="Gravity is a fundamental force...",
    max_batch_size=25
)

# Save configuration
results = explainer.collect_logits_by_manipulating_attention(
    prompt=["What is AI?"],
    target="AI is...",
    save_configs_as="ai_explanation_config.json"
)
```

