# Attention Manipulation

This module provides functionality for manipulating attention scores in transformer models.

## Functions

### `manipulate_attention_scores(attention_scores, attention_mask, modified_causal_attention_mask, multiplicative=True, apply_softmax=False)`
Modifies attention scores based on provided masks and parameters.

#### Parameters
- `attention_scores: torch.Tensor` - Original attention scores from the model
- `attention_mask: torch.Tensor` - Standard attention mask for padding
- `modified_causal_attention_mask: torch.Tensor` - Custom mask for attention manipulation
- `multiplicative: bool` - Whether to use multiplicative (True) or additive (False) manipulation
- `apply_softmax: bool` - Whether to apply softmax after manipulation

#### Returns
- `torch.Tensor` - Modified attention scores

#### Example
```python
modified_scores = manipulate_attention_scores(
    attention_scores=scores,
    attention_mask=mask,
    modified_causal_attention_mask=custom_mask,
    multiplicative=True
)
```

## Implementation Details

### Multiplicative Manipulation
When `multiplicative=True`:
1. Applies the modified causal attention mask multiplicatively
2. Sets masked positions to a very low value (-10000.0)

### Additive Manipulation
When `multiplicative=False`:
1. Adds the modified causal attention mask to the attention scores

## Usage Example

```python
# Prepare inputs
batch_size = 4
seq_len = 32
num_heads = 8
attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
attention_mask = torch.ones(batch_size, seq_len)
modified_mask = torch.ones(batch_size, num_heads, seq_len, seq_len)

# Manipulate attention
modified_scores = manipulate_attention_scores(
    attention_scores=attention_scores,
    attention_mask=attention_mask,
    modified_causal_attention_mask=modified_mask
)
```

## Notes

1. Batch size must match between attention scores and modified mask