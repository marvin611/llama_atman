# Document QA Explainer

A class for composing Atman modules and visualizing how different parts of a document contribute to question-answering tasks.

## Classes

### `DocumentQAExplanationOutput`
Container class for processed explanation outputs.

#### Methods
- `save_as(filename: str)` - Saves explanation data to JSON file
- `__repr__()` - Returns string representation of the data

### `DocumentQAExplainer`
Main class for generating and visualizing document-based QA explanations.

#### Constructor Parameters
- `model` - Language model instance with tokenizer
- `document: str` - Input document text
- `explanation_delimiters: str` - Delimiter for chunking text (default: '\n')
- `device: str` - Computing device (default: 'cuda:0')
- `suppression_factor: float` - Factor for attention suppression (default: 0.0)

## Key Methods

### `run(question: str, expected_answer: str, max_batch_size: int = 25)`
Generates explanation data for a question-answer pair.

Example:

```python
explainer = DocumentQAExplainer(model, document)
output = explainer.run(
question="What is the main topic?",
expected_answer="The main topic is AI.",
max_batch_size=25
)
```
- **Returns:** Raw explanation output data

### `postprocess(output: Union[DeltaLogitsOutput, DeltaCrossEntropiesOutput]) -> DocumentQAExplanationOutput`
Processes raw explanation data into a more usable format.

Example:

```python
raw_output = explainer.run(question, answer)
processed = explainer.postprocess(raw_output)
```
- **Returns:** Processed explanation data with chunks and their importance values

### `show_output(output, question, expected_answer, save_as: str = None, figsize = (15, 11), fontsize = 20)`
Visualizes explanation results as a horizontal bar chart.

Example:

```python
explainer.show_output(
output=raw_output,
question="What is the main topic?",
expected_answer="The main topic is AI.",
save_as="explanation.png"
)
```

### `get_chunks_and_values(postprocessed_output)`
Extracts chunks and their corresponding importance values.

Example:

```python
processed = explainer.postprocess(raw_output)
chunks, values = explainer.get_chunks_and_values(processed)
```
- **Returns:** Tuple of (chunks, values)


## Internal Methods

### `get_prompt(question: str) -> List[str]`
Formats the input document and question into a prompt.

### `get_chunks_and_values(postprocessed_output)`
Extracts text chunks and their importance values from processed output.

## Dependencies

- `explainer`: Base explanation generation
- `logit_parsing`: Cross-entropy calculations
- `outputs`: Output data structures
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `sentence_wise_suppression`: Text chunking
- `utils`: Utility functions

## Usage Example

# Initialize explainer

```python
document = "This is a sample document about AI..."
explainer = DocumentQAExplainer(
    model=llm_model,
    document=document,
    device='cuda:0'
)
```

# Generate explanation

```python	
output = explainer.run(
question="What is AI?",
expected_answer="AI is artificial intelligence"
)
```

# Process and visualize results

```python
processed = explainer.postprocess(output)
explainer.show_output(
output=output,
question="What is AI?",
    expected_answer="AI is artificial intelligence",
    save_as="ai_explanation.png"
)
```

