# Llama Model Handler

A wrapper class for handling LLaMA-models with Atman.

## Class Overview

### `Llama`
Main class for managing LLaMA model operations.

#### Constructor Parameters
- `model_name: str` - Name of the model (default: "TinyLlama-1.1B")
- `models_dir: str` - Directory for model storage (default: "./models")
- `device: str` - Computing device (default: "cuda" if available, else "cpu")

## Supported Models

Currently supported models:
- `TinyLlama-1.1B`: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
- `Llama-3-DiscoLeo-8B-32k-Instruct`: "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1"

## Key Methods

### `setup() -> bool`
Sets up the complete model pipeline.

Example:
```python
model = Llama(model_name="TinyLlama-1.1B")
success = model.setup()
```
- **Returns:** `True` if setup successful, `False` otherwise

### `generate(input_text: str, max_new_tokens: int = 256, temperature: float = 0.7, do_sample: bool = True, seed: Optional[int] = None) -> str`
Generates text response for given input.

Example:

```python
response = model.generate(
    input_text="What is artificial intelligence?",
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
    seed=42
)
```

### `list_available_models() -> List[str]`
Class method to list all supported models.

Example:

```python
available_models = Llama.list_available_models()
# Returns: ["TinyLlama-1.1B", "Llama-3-DiscoLeo-8B-32k-Instruct"]
```

### `list_downloaded_models(models_dir: str = "./models") -> List[str]`
Class method to list locally downloaded models.

Example:

```python
downloaded = Llama.list_downloaded_models()
# Returns: ["TinyLlama-1.1B"]
```

### `preprocess_inputs(input_list: Union[str, List[str]], embed: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]`
Preprocesses text inputs for the model. Returns a tensor of tokens if `embed` is `True`, otherwise returns a list of tokens.

Example:

```python
tokens = model.preprocess_inputs(
    "Sample text",
    embed=False
)
embeddings = model.preprocess_inputs(
    "Sample text",
    embed=True
)
```

### `embed(inputs: List[torch.Tensor]) -> torch.Tensor`
Converts tokenized inputs to embeddings.

Example:

```python
tokens = model.preprocess_inputs("Sample text", embed=False)
embeddings = model.embed([tokens])
```

## Internal Methods

### `_download_model() -> None`
Downloads model files from Hugging Face Hub.

### `_load_config() -> None`
Loads model configuration.

### `_load_tokenizer() -> None`
Initializes the tokenizer.

### `_initialize_model() -> None`
Sets up the model architecture.

### `_load_weights() -> None`
Loads pre-trained model weights.

## Properties

### `word_embedding`
Access to the model's word embedding layer.

Example:

```python
embedding_layer = model.word_embedding
```

## Dependencies

- `torch`
- `transformers`
- `safetensors`
- `huggingface_hub`
- `modeling_llama`: Atman's LLaMA-model implementation

## Usage Example

```python
# Initialize model
model = Llama(model_name="TinyLlama-1.1B", device='cpu')

# Setup model
model.setup()

# Generate text
prompt = "What is the capital of France?"
response = model.generate(
    prompt,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True
)
print(f"Response: {response}")
```

## Notes

1. Models are automatically downloaded when not present locally
2. Supports both CPU and CUDA devices
3. Uses SDPA attention that was extended to be used with Atman
4. Includes special tokens for chat/instruction formatting
5. Implements temperature and sampling controls for generation
