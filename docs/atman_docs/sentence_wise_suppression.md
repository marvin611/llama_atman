# Sentence Wise Suppression

A utility class for splitting text into chunks based on specified delimiters and configuring suppression for each chunk.

## Functions

### `split_list_based_on_indices(list_to_split, split_indices)`
Splits a list into sublists based on specified indices.

#### Example
```python
list_to_split = [1, 2, 3, 4, 5]
split_indices = [2, 4]
sublists = split_list_based_on_indices(list_to_split, split_indices)
# Returns: [[1, 2], [3, 4], [5]]
```

### `search_multi_token_delimiter(token_ids, delimiter)`
Splits a list of token IDs based on a multi-token delimiter, including the delimiter at the end of each chunk.

#### Example
```python
token_ids = [0, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 3, 4, 9, 9, 9, 2, 1, 3, 56, 2]
delimiter = [3, 4]
chunks = search_multi_token_delimiter(token_ids, delimiter)
# Returns: [[0, 0, 1, 2, 3, 4], [5, 3, 4], [5, 6, 7, 8, 3, 4], [9, 9, 9, 2, 1, 3, 56, 2]]
```

## Class

### `SentenceWiseSuppression`
Handles text chunking and suppression configuration.

#### Constructor Parameters
- `prompt: str` - The input text to be chunked
- `delimiters: List[str]` - Delimiters for splitting text
- `tokenizer_encode_fn: Callable` - Function to encode text into tokens
- `tokenizer_decode_fn: Callable` - Function to decode tokens back into text

#### Methods

### `get_split_prompt() -> List[str]`
Splits the input prompt into chunks based on specified delimiters.

#### Returns
- `List[str]`: A list of text chunks, where each chunk is separated by the delimiters provided during initialization.

#### Example
```python
# Initialize with a prompt and delimiters
sentence_wise_suppression = SentenceWiseSuppression(
    prompt="Hello world! How are you? I am fine.",
    delimiters=["!", "?", "."],
    tokenizer_encode_fn=tokenizer.encode,
    tokenizer_decode_fn=tokenizer.decode
)

# Get the split prompt
chunks = sentence_wise_suppression.get_split_prompt()
print(chunks)
# Output: ['Hello world!', ' How are you?', ' I am fine.']
```

#### Notes
1. Delimiters are included at the end of each chunk
2. Leading whitespace is preserved in chunks
3. Empty chunks are filtered out
4. The original text can be reconstructed by joining the chunks
5. You could also pass `["word"]` as a delimiter to split the text into words	

### `get_config(suppression_factor: float) -> List[dict]`
Generates a configuration for suppression based on chunk indices.

#### Example
```python
# Initialize with a prompt and delimiters
sentence_wise_suppression = SentenceWiseSuppression(
    prompt="Hello world! How are you? I am fine.",
    delimiters=["!", "?", "."],
    tokenizer_encode_fn=tokenizer.encode,
    tokenizer_decode_fn=tokenizer.decode
)

# Get suppression configuration
config = sentence_wise_suppression.get_config(suppression_factor=0.5)
print(config)
# Output: [
#   {'suppression_token_index': [0], 'suppression_factor': [0.5]},
#   {'suppression_token_index': [1], 'suppression_factor': [0.5]},
#   {'suppression_token_index': [2], 'suppression_factor': [0.5]}
# ]
```

#### Notes
- The `suppression_factor` is applied uniformly across all chunks.
- Each dictionary in the list corresponds to a chunk, with indices and factors specified.


## Internal Methods

### `_split_by_words(text: str) -> List[str]`
Splits text into words.

### `_split_by_delimiters(text: str, delimiters: List[str]) -> List[str]`
Splits text using the provided delimiters.

## Usage Example

```python
sentence_wise_suppression = SentenceWiseSuppression(
    prompt="This is a sample document. It contains multiple sentences.",
    delimiters=["."],
    tokenizer_encode_fn=tokenizer.encode,
    tokenizer_decode_fn=tokenizer.decode
)

# Get split prompt
split_prompt = sentence_wise_suppression.get_split_prompt()

# Get suppression configuration
config = sentence_wise_suppression.get_config(suppression_factor=0.5)
```

