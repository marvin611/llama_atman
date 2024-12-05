# Utilities

A collection of utility functions for model operations, text processing, and file handling.

## Model Operation Functions

### `get_output_logits_from_input_ids(model, input_ids) -> torch.Tensor`
Gets model output logits from input token IDs.

```python
logits = get_output_logits_from_input_ids(model, input_ids)
```

### `get_output_logits_from_embeddings(model, embeddings) -> torch.Tensor`
Gets model output logits from input embeddings.

```python
logits = get_output_logits_from_embeddings(model, embeddings)
```

## Text Processing Functions

### `chunks(xs, n) -> Generator`
Splits a sequence into chunks of size n.

```python
for chunk in chunks([1, 2, 3, 4, 5], n=2):
    print(chunk)  # [1, 2], [3, 4], [5]
```

### `split_str_into_tokens(x: str, tokenizer, numbered: bool = False) -> List[str]`
Splits a string into its constituent tokens.

```python
tokens = split_str_into_tokens(
    "Hello world",
    tokenizer,
    numbered=True
)  # ['0:Hello', '1:world']
```

### `parse_text_result(result, target_token_index: int = 0) -> Dict`
Parses explanation results for a specific target token.

```python
parsed = parse_text_result(result, target_token_index=0)
# Returns: {'values': [...], 'target_token': token_id}
```

## Tensor Operations

### `normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor`
Normalizes tensor values to range [0, 1].

```python
normalized = normalize_tensor(tensor)
```

## File Operations

### `load_json_as_dict(filename: str) -> Dict`
Loads a JSON file into a dictionary.

```python
data = load_json_as_dict("config.json")
```

### `dict_to_json(dictionary: Dict, filename: str)`
Saves a dictionary to a JSON file with verification.

```python
dict_to_json(data, "output.json")
```

### `create_folder_if_does_not_exist(folder: str)`
Creates a folder if it doesn't exist.

```python
create_folder_if_does_not_exist("output_dir")
```

## Dependencies

- `torch`
- `json`
- `os`

