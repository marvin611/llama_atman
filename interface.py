import streamlit as st
from annotated_text import annotated_text
from document_qa_explainer import DocumentQAExplainer
from typing import List
from st_click_detector import click_detector
from rag import RAG, RAGResult
import torch
import asyncio
from llama import Llama


# Initialize basic session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "TinyLlama-1.1B"  # Set default model
if "model_handler" not in st.session_state:
    st.session_state.model_handler = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if 'show_test_results' not in st.session_state:
    st.session_state.show_test_results = False

st.title("RAG Chatbot with Explanations")


# Run comparison tests
def run_comparison_tests():
    """Run comparison tests between basic generation and RAG+explanations"""
    st.markdown("## Automated Tests")
    
    with st.spinner("Running tests..."):
        from tests.test_generation import test_basic_generation
        from tests.test_rag_explanations import test_rag_with_explanations
        from tests.conftest import TEST_CASES
        
        # Run both test sets
        basic_results = test_basic_generation(TEST_CASES)
        rag_results = test_rag_with_explanations(TEST_CASES)
        
        # Display results
        for basic, rag in zip(basic_results, rag_results):
            st.markdown(f"### Test: {basic.prompt}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Generation")
                st.markdown(f"**Response:**\n{basic.response}")
                st.markdown(f"Time: {basic.execution_time:.2f}s")
            
            with col2:
                st.markdown("#### RAG + Explanations")
                st.markdown(f"**Response:**\n{rag.response}")
                if rag.explanations:
                    st.markdown("**Relevant Context:**")
                    for i, exp in enumerate(rag.explanations, 1):
                        st.markdown(f"{i}. {exp}")
                st.markdown(f"Time: {rag.execution_time:.2f}s")
            
            st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.markdown("### RAG Configuration")
    enable_web_search = st.toggle("Enable Web Search", value=True)
    top_k_web = st.slider("Web Results", 0, 5, 2)
    
    st.markdown("### Explanation Configuration")
    enable_explanations = st.toggle("Enable Explanations", value=True)
    chunk_mode = st.selectbox(
        "Chunk Mode",
        ["word", "sentence", "paragraph"],
        index=1
    )
    
    st.markdown("### Model Configuration")
    available_models = Llama.list_available_models()
    downloaded_models = Llama.list_downloaded_models()
    
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.current_model),
        key="model_selector"
    )
    
    # Show download status
    if selected_model in downloaded_models:
        st.success(f"Model {selected_model} is already downloaded")
    else:
        st.info(f"Model {selected_model} will be downloaded when selected")
    
    # Model parameters
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_new_tokens = st.slider("Max New Tokens", 64, 1024, 256)
    seed = st.number_input("Seed", min_value=0, max_value=1000000, value=42)

    # Add separator before test section
    st.markdown("---")
    st.markdown("### Test Section")
    
    if st.button("Run Comparison Tests", use_container_width=True):
        st.session_state.show_test_results = True

# Handle model initialization/switching
if (st.session_state.current_model != selected_model) or (st.session_state.model_handler is None):
    with st.spinner(f"Loading model {selected_model}..."):
        # Initialize model handler
        model_handler = Llama(
            model_name=selected_model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        if not model_handler.setup():
            st.error("Failed to set up model")
            st.stop()
        
        # Initialize RAG with the model handler
        bing_key = st.secrets.get("BING_SUBSCRIPTION_KEY", None)
        rag = RAG(
            model_handler=model_handler,  # Pass the model handler directly
            device='cuda' if torch.cuda.is_available() else 'cpu',
            web_search_enabled=bool(bing_key),
            bing_subscription_key=bing_key
        )
        
        # Update session state
        st.session_state.model_handler = model_handler
        st.session_state.rag = rag
        st.session_state.current_model = selected_model
        st.success(f"Model {selected_model} loaded successfully!")

# Add a warning if Bing key is not available
if enable_web_search and not st.session_state.rag.bing_subscription_key:
    st.warning("Web search is enabled but Bing API key is not configured. Please add BING_SUBSCRIPTION_KEY to your secrets.")



# Helper functions
def create_annotated_text(text: str, highlight_chunks: List[str]) -> List[tuple]:
    """
    Create annotated text with custom styling for the most relevant chunks.
    Args:
        text: The full context text
        highlight_chunks: List of chunks to highlight
    Returns:
        List of tuples formatted for annotated_text component
    """
    annotated = []
    remaining_text = text
    
    # Sort chunks by length (longest first) to handle overlapping chunks
    sorted_chunks = sorted(highlight_chunks, key=len, reverse=True)
    
    while remaining_text and sorted_chunks:
        chunk = sorted_chunks[0]
        if chunk in remaining_text:
            parts = remaining_text.split(chunk, 1)
            if parts[0]:  # Add text before chunk
                annotated.append(parts[0])
            # Add highlighted chunk with label
            annotated.append((chunk, "#faa"))
            remaining_text = parts[1]
        else:
            sorted_chunks.pop(0)
    
    if remaining_text:  # Add any remaining text
        annotated.append(remaining_text)
    
    return annotated


def create_clickable_text(text: str, chunk_mode: str = "word") -> str:
    """
    Create clickable text format based on chunk mode
    Args:
        text: The text to make clickable
        chunk_mode: "word" or "sentence" or "paragraph"
    Returns:
        HTML formatted string with clickable elements
    """
    if chunk_mode == "sentence":
        # More robust sentence splitting
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        if current.strip():  # Add any remaining text
            sentences.append(current.strip())
            
        clickable_elements = []
        for i, sentence in enumerate(sentences):
            if sentence:
                clickable_elements.append(
                    f'<a href="#" id="sentence_{i}" style="color: inherit; cursor: pointer; transition: all 0.2s ease;">{sentence}</a>'
                )
        
        return f'<div style="line-height: 1.5">{" ".join(clickable_elements)}</div>'
    
    else:  # word mode
        words = text.split()
        clickable_words = []
        for i, word in enumerate(words):
            clickable_words.append(
                f'<a href="#" id="word_{i}" style="color: inherit; cursor: pointer; transition: all 0.2s ease;">{word}</a>'
            )
        
        return f'<div style="line-height: 1.5">{" ".join(clickable_words)}</div>'


# When initializing the explainer, set the delimiter based on chunk_mode
def get_delimiters_for_mode(mode: str) -> List[str]:
    """
    Returns appropriate delimiters based on the chunking mode
    Args:
        mode: One of "word", "sentence", "paragraph"
    Returns:
        List of delimiter strings
    """
    if mode == "word":
        return [" "]
    elif mode == "sentence":
        return [".", "!", "?"]
    else:  # paragraph
        return ["\n\n"]


def handle_click_and_explanations(clicked_text: str, context: str, model_handler):
    """Handle click events and generate explanations for selected text"""
    if clicked_text:
        with st.spinner("Analyzing selected text..."):
            # Get delimiters based on chunk mode
            delimiters = get_delimiters_for_mode(chunk_mode)
            # Initialize explainer with context
            explainer = DocumentQAExplainer(
                model=model_handler,
                document=context,
                explanation_delimiters=delimiters,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                suppression_factor=0.5
            )
            
            # Run explainer on selected text
            output = explainer.run(
                question="Why is this part of the response relevant?",
                expected_answer=clicked_text
            )
            
            if output:
                # Get the two most relevant chunks
                postprocessed = explainer.postprocess(output)
                relevant_chunks = [chunk['chunk'] for chunk in sorted(
                    postprocessed.data, 
                    key=lambda x: x['value'], 
                    reverse=True
                )[:2]]
                
                # Display annotated context
                st.markdown("#### Context Analysis:")
                annotated_chunks = create_annotated_text(context, relevant_chunks)
                annotated_text(*annotated_chunks)
            else:
                st.warning("Could not generate explanations for the selected text.")



# Chat input and response generation
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get context from RAG (but don't display it)
    result = st.session_state.rag.retrieve_context(
        query=prompt,
        top_k=top_k_web,
        web_only=enable_web_search
    )
    
    # Generate response
    response = st.session_state.rag.generate_response(
        query=prompt,
        context="\n\n".join(result.contexts) if result.contexts else None,
        seed=seed,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    
    # Add assistant message with context stored but not displayed
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": "\n\n".join(result.contexts) if result.contexts else None
    })

    # Optional: Display context viewer
    if result.contexts:  # Only show expander if there's context to display
        with st.expander("View Retrieved Context"):
            for ctx, src, score in zip(result.contexts, result.sources, result.relevance_scores):
                with st.container():
                    # Source header with score
                    st.markdown(f"**Source ({score:.3f}):**")
                    
                    # Source metadata
                    if src['type'] == 'web':
                        st.markdown(f"Type: Web\nURL: {src['metadata'].get('url', 'N/A')}")
                    else:
                        st.markdown(f"Type: {src['type']}")
                    
                    # Source content in a distinct box
                    st.markdown("**Content:**")
                    st.markdown(f"```\n{ctx}\n```")
                    st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Add clickable version for explanations if enabled
            if enable_explanations:
                clicked = click_detector(
                    create_clickable_text(message["content"], chunk_mode)
                )
                
                # Show explanations if text is clicked and context exists
                if clicked and message.get("context"):
                    handle_click_and_explanations(
                        clicked_text=clicked,
                        context=message["context"],
                        model_handler=st.session_state.rag.llm
                    )
        else:
            # Display user messages normally
            st.markdown(message["content"])

if st.session_state.show_test_results:
    st.markdown("## Test Results")
    with st.spinner("Running tests..."):
        from tests.test_generation import test_basic_generation
        from tests.test_rag_explanations import test_rag_with_explanations
        from tests.conftest import TEST_CASES
        
        progress_text = st.empty()
        
        # Run tests with progress updates
        progress_text.text("Running basic generation tests...")
        basic_results = test_basic_generation(TEST_CASES)
        
        progress_text.text("Running RAG + explanations tests...")
        rag_results = test_rag_with_explanations(TEST_CASES)
        
        progress_text.empty()
        
        # Display results
        for basic, rag in zip(basic_results, rag_results):
            st.markdown(f"### Test: {basic.prompt}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Basic Generation")
                st.markdown(f"**Response:**\n{basic.response}")
                st.markdown(f"Time: {basic.execution_time:.2f}s")
            
            with col2:
                st.markdown("#### RAG + Explanations")
                st.markdown(f"**Response:**\n{rag.response}")
                if rag.explanations:
                    st.markdown("**Relevant Context:**")
                    for i, exp in enumerate(rag.explanations, 1):
                        st.markdown(f"{i}. {exp}")
                st.markdown(f"Time: {rag.execution_time:.2f}s")
                
                # Show context in expander if available
                if rag.context:
                    with st.expander("View Context"):
                        st.markdown(rag.context)
            
            st.markdown("---")
    
    # Reset flag after displaying
    st.session_state.show_test_results = False