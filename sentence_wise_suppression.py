from typing import Callable, List
import numpy as np
import re

def split_list_based_on_indices(list_to_split, split_indices):
    """
    Splits a list into sublists based on specified indices.

    Parameters:
    - list_to_split (list): The list to be split.
    - split_indices (list): Indices at which to split the list.

    Returns:
    - list: A list of sublists, where each sublist is a segment of the original list split at the specified indices.

    Example:
    >>> list_to_split = [1, 2, 3, 4, 5]
    >>> split_indices = [2, 4]
    >>> split_list_based_on_indices(list_to_split, split_indices)
    [[1, 2], [3, 4], [5]]
    """
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
    """
    Handles text chunking and suppression configuration for sentence-wise suppression.
    
    Parameters:
    - prompt (str): The input text to be chunked.
    - delimiters (List[str]): Delimiters for splitting text. If ["word"], splits the text into words.
    - tokenizer_encode_fn (Callable): Function to encode text into tokens.
    - tokenizer_decode_fn (Callable): Function to decode tokens back into text.
    """
    def __init__(self, prompt: str, delimiters: List[str], tokenizer_encode_fn: Callable, tokenizer_decode_fn: Callable):
        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.tokenizer_decode_fn = tokenizer_decode_fn
        
        # Choose chunking method based on delimiter
        if delimiters == ["word"]:
            text_chunks = self._split_by_words(prompt)
        else:
            text_chunks = self._split_by_delimiters(prompt, delimiters)
        
        # Process chunks into tokens
        self.chunk_token_ids = []
        current_position = 0
        self.chunk_indices = []
        
        for chunk in text_chunks:
            chunk_tokens = self.tokenizer_encode_fn(chunk, add_special_tokens=False)
            chunk_range = list(range(current_position, current_position + len(chunk_tokens)))
            self.chunk_indices.append(chunk_range)
            current_position += len(chunk_tokens)
            self.chunk_token_ids.append(chunk_tokens)

        # Flatten token IDs
        self.prompt_token_ids = [t for chunk in self.chunk_token_ids for t in chunk]

        print(f"Split into {len(text_chunks)} chunks")
        print(f"First few chunks: {text_chunks[:3]}")

    def _split_by_words(self, text: str) -> List[str]:
        """
        Splits the input text into a list of words.

        This method uses regular expressions to find all sequences of non-whitespace characters in the input text, effectively splitting the text into words.

        Parameters:
        - text (str): The input text to be split into words.

        Returns:
        - List[str]: A list of words extracted from the input text.
        """
        return re.findall(r'\S+', text)

    def _split_by_delimiters(self, text: str, delimiters: List[str]) -> List[str]:
        """
        Splits the input text into chunks based on the provided delimiters.

        This method iterates through each delimiter and splits the text into chunks. It accumulates the new chunks and updates the list of text chunks. Finally, it removes any empty chunks and leading/trailing whitespace from the chunks before returning them.

        Parameters:
        - text (str): The input text to be split into chunks.
        - delimiters (List[str]): A list of delimiters to split the text by.

        Returns:
        - List[str]: A list of text chunks split by the provided delimiters.
        """
        text_chunks = [text]
        for delimiter in delimiters:
            new_chunks = []
            for chunk in text_chunks:
                new_chunks.extend(chunk.split(delimiter))
            text_chunks = new_chunks

        # Remove empty chunks and whitespace
        return [chunk.strip() for chunk in text_chunks if chunk.strip()]

    def get_split_prompt(self) -> List[str]:
        """
        Splits the input prompt into chunks based on the token IDs stored in the instance.

        This method iterates over the list of chunk token IDs, decodes each chunk back into text using the tokenizer's decode function, and returns a list of these decoded text chunks.

        Returns:
        - List[str]: A list of text chunks, where each chunk is a decoded version of the corresponding token IDs.
        """
        return [self.tokenizer_decode_fn(chunk) for chunk in self.chunk_token_ids]
            
    def get_config(self, suppression_factor: float) -> List[dict]:
        """
        Generates a custom configuration for suppression based on the provided suppression factor and chunk indices.

        This method initializes a base configuration with a default suppression factor of 1.0 for the first chunk (index -1). Then, it iterates over the list of chunk indices stored in the instance and appends a new configuration for each chunk. The suppression factor for each chunk is set to the value provided as an argument. The method returns a list of these custom configurations.

        Parameters:
        - suppression_factor (float): The factor to apply for suppression.

        Returns:
        - List[dict]: A list of dictionaries, each representing a configuration for suppression. Each dictionary contains 'suppression_token_index' and 'suppression_factor' as keys.
        """
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
            
