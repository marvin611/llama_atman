# Outputs

This module defines classes for handling and visualizing the results of delta cross-entropy and delta logits calculations.

## Classes

### `DeltaCrossEntropiesOutput`
Handles output data for delta cross-entropy calculations.

#### Constructor Parameters
- `data: dict` - Explanation data

#### Methods

- `save(filename: str)`
  - Saves the explanation data to a JSON file.
  - **Example:**
    ```python
    output.save("cross_entropy_output.json")
    ```

- `get_text_heatmap(target_token_idx: int, square_outputs: bool = False) -> dict`
  - Generates a heatmap for a specific target token.
  - **Example:**
    ```python
    heatmap = output.get_text_heatmap(target_token_idx=0)
    ```

- `show_image(image_token_start_idx: int, target_token_idx: int) -> np.ndarray`
  - Returns a 12x12 heatmap for image tokens.
  - **Example:**
    ```python
    image_heatmap = output.show_image(image_token_start_idx=0, target_token_idx=0)
    ```

- `from_file(filename: str) -> DeltaCrossEntropiesOutput`
  - Loads explanation data from a JSON file.
  - **Example:**
    ```python
    output = DeltaCrossEntropiesOutput.from_file("cross_entropy_output.json")
    ```

### `DeltaLogitsOutput`
Handles output data for delta logits calculations.

#### Constructor Parameters
- `data: dict` - Explanation data

#### Methods

- `save(filename: str)`
  - Saves the explanation data to a JSON file.
  - **Example:**
    ```python
    output.save("logits_output.json")
    ```

- `get_text_heatmap(target_token_idx: int, square_outputs: bool = False) -> dict`
  - Generates a heatmap for a specific target token.
  - **Example:**
    ```python
    heatmap = output.get_text_heatmap(target_token_idx=0)
    ```

- `show_image(image_token_start_idx: int, target_token_idx: int) -> np.ndarray`
  - Returns a 12x12 heatmap for image tokens.
  - **Example:**
    ```python
    image_heatmap = output.show_image(image_token_start_idx=0, target_token_idx=0)
    ```

- `from_file(filename: str) -> DeltaLogitsOutput`
  - Loads explanation data from a JSON file.
  - **Example:**
    ```python
    output = DeltaLogitsOutput.from_file("logits_output.json")
    ```

## Dependencies

- `json`: For saving and loading JSON data
- `numpy`: For numerical operations and heatmap generation
- `utils`: Utility functions for loading JSON data

## Usage Example

```python
# Initialize output
data = {...}  # Some explanation data
output = DeltaCrossEntropiesOutput(data)

# Save to file
output.save("output.json")

# Load from file
loaded_output = DeltaCrossEntropiesOutput.from_file("output.json")

# Generate heatmap
heatmap = loaded_output.get_text_heatmap(target_token_idx=0)

# Show image heatmap
image_heatmap = loaded_output.show_image(image_token_start_idx=0, target_token_idx=0)
```

## Notes

1. Both classes provide similar functionality for different types of output data.
2. Heatmaps are generated as numpy arrays for visualization.
3. JSON files are used for saving and loading explanation data.
