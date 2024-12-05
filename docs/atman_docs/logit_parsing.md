# Logit Parsing

This module provides functions to compute delta cross-entropies and delta logits for model outputs, helping to analyze the impact of attention manipulation.

## Functions

### `get_delta_cross_entropies(output: dict, square_outputs: bool = False, custom_chunks: bool = False, temperature: float = None) -> DeltaCrossEntropiesOutput`
Calculates the change in cross-entropy loss when attention is manipulated.

#### Parameters
- `output: dict` - Dictionary containing original and manipulated logits
- `square_outputs: bool` - Whether to square the output differences (default: False)
- `custom_chunks: bool` - Use custom chunking for explanations (default: False)
- `temperature: float` - Temperature for scaling logits (optional)

#### Returns
- `DeltaCrossEntropiesOutput` - Object containing explanation data

#### Example
```python
output = {
    "original_logits": torch.tensor(...),
    "suppressed_chunk_logits": [...],
    "target_token_ids": [...],
    "target_token_indices": [...],
    "prompt_explain_indices": [...],
    "prompt_length": ...
}
delta_ce = get_delta_cross_entropies(output)
```

### `get_delta_logits(output: dict, square_outputs: bool = False) -> DeltaLogitsOutput`
Calculates the change in logits when attention is manipulated.

#### Parameters
- `output: dict` - Dictionary containing original and manipulated logits
- `square_outputs: bool` - Whether to square the output differences (default: False)

#### Returns
- `DeltaLogitsOutput` - Object containing explanation data

#### Example
```python
output = {
    "original_logits": torch.tensor(...),
    "suppressed_chunk_logits": [...],
    "target_token_ids": [...],
    "target_token_indices": [...],
    "prompt_explain_indices": [...],
    "prompt_length": ...
}
delta_logits = get_delta_logits(output)
```

## Output Structures

### `DeltaCrossEntropiesOutput`
Contains data on how cross-entropy changes with attention manipulation.

### `DeltaLogitsOutput`
Contains data on how logits change with attention manipulation.

## Dependencies

- `torch`
- `outputs`: Custom output data structures

## Usage Example

```python
# Example output dictionary
output = {
    "original_logits": torch.tensor(...),
    "suppressed_chunk_logits": [...],
    "target_token_ids": [...],
    "target_token_indices": [...],
    "prompt_explain_indices": [...],
    "prompt_length": ...
}

# Calculate delta cross-entropies
delta_ce = get_delta_cross_entropies(output, square_outputs=True)

# Calculate delta logits
delta_logits = get_delta_logits(output, square_outputs=False)
```

## Notes

1. Ensure the output dictionary contains all required keys.
2. Supports both squared and non-squared output differences.
3. Can handle custom chunking for explanations.
4. Temperature scaling is optional for cross-entropy calculations.
