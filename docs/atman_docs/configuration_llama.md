# LLaMA Model Configuration

This module defines the configuration class for the LLaMA model, which is used to instantiate and configure the model architecture.

## Class

### `LlamaConfig`
Configuration class for the LLaMA model, inheriting from `PretrainedConfig`.

#### Constructor Parameters

- `vocab_size` (`int`, *optional*, defaults to 32000): Vocabulary size of the LLaMA model.
- `hidden_size` (`int`, *optional*, defaults to 4096): Dimension of the hidden representations.
- `intermediate_size` (`int`, *optional*, defaults to 11008): Dimension of the MLP representations.
- `num_hidden_layers` (`int`, *optional*, defaults to 32): Number of hidden layers in the Transformer decoder.
- `num_attention_heads` (`int`, *optional*, defaults to 32): Number of attention heads for each attention layer.
- `num_key_value_heads` (`int`, *optional*): Number of key-value heads for Grouped Query Attention.
- `hidden_act` (`str` or `function`, *optional*, defaults to `"silu"`): Activation function in the decoder.
- `max_position_embeddings` (`int`, *optional*, defaults to 2048): Maximum sequence length for the model.
- `initializer_range` (`float`, *optional*, defaults to 0.02): Standard deviation for weight initialization.
- `rms_norm_eps` (`float`, *optional*, defaults to 1e-06): Epsilon for RMS normalization layers.
- `use_cache` (`bool`, *optional*, defaults to `True`): Whether to return the last key/values attentions.
- `pad_token_id` (`int`, *optional*): Padding token id.
- `bos_token_id` (`int`, *optional*, defaults to 1): Beginning of stream token id.
- `eos_token_id` (`int`, *optional*, defaults to 2): End of stream token id.
- `pretraining_tp` (`int`, *optional*, defaults to 1): Tensor parallelism rank used during pretraining.
- `tie_word_embeddings` (`bool`, *optional*, defaults to `False`): Whether to tie weight embeddings.
- `rope_theta` (`float`, *optional*, defaults to 10000.0): Base period of the RoPE embeddings.
- `rope_scaling` (`Dict`, *optional*): Scaling configuration for RoPE embeddings.
- `attention_bias` (`bool`, *optional*, defaults to `False`): Whether to use a bias in attention layers.
- `attention_dropout` (`float`, *optional*, defaults to 0.0): Dropout ratio for attention probabilities.
- `manipulate_attn_scores_after_scaling` (`bool`, *optional*, defaults to `False`): Whether to manipulate attention scores after scaling.
- `full_bf16` (`bool`, *optional*, defaults to `False`): Whether to use full bf16 precision.
- `rotary` (`bool`, *optional*, defaults to `True`): Whether to use rotary embeddings.
- `rotary_dim` (`int`, *optional*): Dimension of the rotary embeddings.

#### Example

```python
from transformers import LlamaModel, LlamaConfig

# Initializing a LLaMA configuration
configuration = LlamaConfig()

# Initializing a model from the configuration
model = LlamaModel(configuration)

# Accessing the model configuration
configuration = model.config
```

## Custom Parameters

- `manipulate_attn_scores_after_scaling`: Allows manipulation of attention scores after scaling.
- `full_bf16`: Enables full bf16 precision for the model.
- `rotary`: Toggles the use of rotary embeddings.
- `rotary_dim`: Specifies the dimension of rotary embeddings.
