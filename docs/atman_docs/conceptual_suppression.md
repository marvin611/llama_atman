# Conceptual Suppression

This module provides functionality for modifying attention suppression configurations based on conceptual similarity between token embeddings.

## Class

### `ConceptualSuppression`
Handles the modification of suppression configurations using cosine similarity between token embeddings.

#### Constructor Parameters
- `embeddings` - Tensor of token embeddings (shape: [1, seq, embedding_dim])
- `similarity_threshold: float` - Threshold for considering tokens as conceptually similar
- `modify_suppression_factor_based_on_cossim: bool` - Whether to adjust suppression factors based on cosine similarity

#### Methods

- `modify_configs(configs: list) -> list`
  - Modifies suppression configurations based on similarity scores.
  - **Example:**
    ```python
    new_configs = conceptual_suppression.modify_configs(configs)
    ```

- `get_similarity_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor`
  - Computes the cosine similarity matrix between two sets of vectors.
  - **Example:**
    ```python
    sim_matrix = conceptual_suppression.get_similarity_matrix(a, b)
    ```

- `get_embedding_similarity_matrix(embeddings_batch: torch.Tensor) -> torch.Tensor`
  - Returns a similarity matrix for the embeddings in the batch.
  - **Example:**
    ```python
    sim_matrix = conceptual_suppression.get_embedding_similarity_matrix(embeddings)
    ```

- `get_suppression_factor_from_cosine_similarity(suppression_factor: float, cosine_similarity: float) -> torch.Tensor`
  - Calculates a new suppression factor based on cosine similarity.
  - **Example:**
    ```python
    new_factor = conceptual_suppression.get_suppression_factor_from_cosine_similarity(0.1, 0.8)
    ```

## Usage Example

```python
# Initialize ConceptualSuppression
embeddings = torch.randn(1, 10, 768)  # Example embeddings
conceptual_suppression = ConceptualSuppression(
    embeddings=embeddings,
    similarity_threshold=0.6,
    modify_suppression_factor_based_on_cossim=True
)

# Modify configurations
configs = [
    {"suppression_token_index": [0], "suppression_factor": [0.1]},
    {"suppression_token_index": [1], "suppression_factor": [0.1]}
]
new_configs = conceptual_suppression.modify_configs(configs)
```

## Notes

1. The embeddings tensor should not contain the last element of the target.
2. The class assumes a batch size of 1 for the embeddings.
3. Cosine similarity is used to determine conceptual similarity between tokens.
4. Suppression factors can be adjusted based on similarity scores.
