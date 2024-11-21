from typing import Callable, List
import numpy as np


def split_list_based_on_indices(list_to_split, split_indices):
    token_ids = np.array(list_to_split)
    split_token_ids = np.split(token_ids, indices_or_sections= split_indices)

    return [x.tolist() for x in split_token_ids]

def search_multi_token_delimiter(token_ids, delimiter):
    """Splits a list of token IDs based on a multi-token (or single token) delimiter
    Note that it includes the delimiter itself at the end of each chunk
    token_ids: list of token ids in prompt
    delimiter: list of toke ids which represent the delimiter

    For example:
    ```python
    token_ids = [0,0,1,2,3,4,5,3,4,5,6,7,8,3,4,9,9,9,2,1,3,56,2]
    delimiter = [3,4]  ## multi token delimiter
    search_multi_token_delimiter(token_ids, delimiter)
    ```
    >>> [[0, 0, 1, 2, 3, 4], [5, 3, 4], [5, 6, 7, 8, 3, 4], [9, 9, 9, 2, 1, 3, 56, 2]]
    """
    len_delimiter = len(delimiter)

    split_indices = []
    for i in range(0, len(token_ids)):
        if token_ids[i:i+len_delimiter] == delimiter:
            print(f"Found delimiter at position {i}")
            split_indices.append(i+len_delimiter)
    print(f"Found {len(split_indices)} split points")

    return split_indices


class SentenceWiseSuppression:
    def __init__(self, prompt: str, delimiters: List[str], tokenizer_encode_fn: Callable, tokenizer_decode_fn: Callable):
        """
        Provides config for suppressing a prompt one sentence at a time.
        The delimiters can be used to specify how a sentence ends, 
        like ["\n", ".", "?", "!"]
        """
        assert isinstance(prompt, str)
        assert isinstance(delimiters, list)

        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.delimiters = delimiters

        # Split text using multiple delimiters
        text_chunks = [prompt]
        for delimiter in delimiters:
            # Split all current chunks using this delimiter
            new_chunks = []
            for chunk in text_chunks:
                new_chunks.extend(chunk.split(delimiter))
            text_chunks = new_chunks

        # Remove empty chunks and whitespace
        text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
        
        self.chunk_token_ids = []
        current_position = 0
        self.chunk_indices = []
        
        for i, chunk in enumerate(text_chunks):
            # Add a period as default delimiter (could be configurable)
            chunk_text = chunk + "." if i < len(text_chunks)-1 else chunk
            chunk_tokens = self.tokenizer_encode_fn(chunk_text, add_special_tokens=False)
            
            # Store token indices for this chunk
            chunk_range = list(range(current_position, current_position + len(chunk_tokens)))
            self.chunk_indices.append(chunk_range)
            current_position += len(chunk_tokens)
            
            self.chunk_token_ids.append(chunk_tokens)

        # Flatten token IDs for full prompt
        self.prompt_token_ids = [t for chunk in self.chunk_token_ids for t in chunk]

        print(f"Split into {len(text_chunks)} chunks")
        print(f"First few chunks: {text_chunks[:3]}")
        
    def get_split_prompt(self):
        return [self.tokenizer_decode_fn(chunk) for chunk in self.chunk_token_ids]
            
    def get_config(self, suppression_factor: float):
        custom_config = [
            {
                "suppression_token_index": [-1], 
                "suppression_factor": [1.0]
            }
        ]
        
        for chunk_indices in self.chunk_indices:
            custom_config.append(
                {
                    "suppression_token_index": chunk_indices,
                    "suppression_factor": [suppression_factor for _ in chunk_indices]
                }
            )
            
        return custom_config
            
