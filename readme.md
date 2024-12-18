## Installation

1. Clone the repository:
```bash
git clone https://github.com/marvin611/llama_atman.git
cd llama_atman
```

2. Create and activate a virtual environment, I used Python 3.12:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Set up Streamlit secrets:
```bash
mkdir .streamlit
touch .streamlit/secrets.toml
```

## Configuration

### Required API Keys
- **Bing Search API**: Required for web search functionality
  - Get your key from [Microsoft Azure](https://azure.microsoft.com/services/cognitive-services/bing-web-search-api/)
  - Add to `.streamlit/secrets.toml`:
    ```toml
    BING_SUBSCRIPTION_KEY = "your-key-here"
    ```

## Usage

### Running the Interface

Start the Streamlit interface:
```bash
streamlit run interface.py
```

The interface provides:
- Model selection and automatic downloading
- RAG configuration with web search
- Explanation settings
- Interactive chat with context viewing
- Automated tests

### Interface

1. **Model Selection**:
   - Choose from available models in the sidebar
   - Models are automatically downloaded when selected
   - Current model: TinyLlama-1.1B

2. **RAG Configuration**:
   - Toggle web search
   - Adjust number of retrieved results
   - View retrieved context in expandable sections

3. **Explanation Settings**:
   - Enable/disable explanations
   - Choose chunking mode (word/sentence/paragraph)
   - Click on response text to see explanations

4. **Testing**:
   - Run comparison tests from the sidebar
   - View side-by-side results of basic generation vs RAG+explanations
   - Reproducible results through fixed seeds

### Running Tests

#### Through the Interface
1. Start the interface
2. Open the test section in the sidebar
3. Click "Run Comparison Tests"
4. View results comparing:
   - Basic generation
   - RAG with explanations
   - Execution times
   - Retrieved context
   - Generated explanations

#### Command Line Testing
Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_generation.py
pytest tests/test_rag_explanations.py
```

## Development

### Adding New Tests
1. Add test cases to `tests/test_cases.py`
2. Create test functions in appropriate test files
3. Update `conftest.py` if needed

### Test Structure
- `test_cases.py`: Defines test cases and result structures
- `test_generation.py`: Tests basic generation
- `test_rag_explanations.py`: Tests RAG and explanations
- `conftest.py`: Test configuration and fixtures

## Troubleshooting

### Common Issues

1. **Web Search Not Working**
   - Check if Bing API key is properly set in secrets.toml
   - Verify web search is enabled in interface

2. **Model Download Issues**
   - Check internet connection
   - Verify enough disk space
   - Try removing partially downloaded files in models/

3. **CUDA/CPU Issues**
   - System automatically selects available device
   - Force CPU usage if needed through device selection

