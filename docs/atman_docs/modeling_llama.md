# Modified LLaMA Architecture

Modifications made to the original LLaMA architecture to support attention manipulation. Note that this is based on the implementation from Hugging Face, using Transformers 4.43.4.

## Key Modifications

### 1. Attention Manipulation Mixin

```python
class LlamaAttentionManipulationMixin:
    def manipulate_attention_weights(
        self,
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        modified_causal_attention_mask: Optional[torch.Tensor],
        manipulate_attn_scores_after_scaling: bool = False,
    ) -> torch.Tensor:
```

This mixin class provides attention manipulation capabilities:
- Integrates with existing LLaMA attention mechanisms
- Allows for custom attention weight modifications
- Supports both pre and post-scaling manipulation

### 2. LLaMA Attention Classes

The architecture supports multiple attention implementations:

```python
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}
```

### 3. Causal Mask Modifications

The `_update_causal_mask` method in `LlamaModel` has been modified to:
- Support custom 4D attention masks
- Handle attention manipulation masks
- Maintain compatibility with different attention implementations

```python
if attention_mask is not None and attention_mask.dim() == 4:
    # Custom 4D attention mask handling
    if attention_mask.max() != 0:
        raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0")
    causal_mask = attention_mask
```

### 4. LlamaForCausalLM Modifications

Added attention manipulation parameters to the model:

```python
class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Added attention manipulation parameters
        self.suppression_factors = None
        self.suppression_token_indices = None
        self.manipulate_attn_scores_after_scaling = False
        self.layers = None  # For specifying which layers to manipulate
```

## Usage

### Setting Up Attention Manipulation

```python
model = LlamaForCausalLM.from_pretrained("path/to/model")
model.suppression_factors = [0.1, 0.2]  # Suppression factors for tokens
model.suppression_token_indices = [1, 2]  # Token indices to manipulate
model.manipulate_attn_scores_after_scaling = False
model.layers = [0, 1, 2]  # Specific layers to manipulate
```

### Custom Attention Mask

```python
# Create a custom 4D attention mask
batch_size = 1
num_heads = model.config.num_attention_heads
seq_length = input_ids.size(1)
custom_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)
```

## Implementation Notes

1. **Compatibility**
   - Works with all LLaMA attention implementations
   - Maintains original model functionality when manipulation is disabled

2. **Attention Manipulation**
   - Can be applied pre or post attention scaling
   - Supports token-level granular control
   - Layer-specific manipulation possible

3. **Mask Handling**
   - Custom 4D masks must be in inverted form (max==0)
   - Automatically handles padding and causal masking
   - Compatible with Flash Attention 2 and SDPA

