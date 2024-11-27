import streamlit as st
from annotated_text import annotated_text
from document_qa_explainer import DocumentQAExplainer
from typing import List
from st_click_detector import click_detector
from rag import RAG, RAGResult
import torch
import asyncio
from llama import Llama
import time
import re


# Initialize example prompts
example_prompts = [
    {
        "text": "Who is the current leader of Germany?",
        "seed": 44,
        "temperature": 0.7,
        "max_tokens": 256
    },
    {
        "text": "What is the latest release of Meta's Llama 3?",
        "seed": 789,
        "temperature": 0.7,
        "max_tokens": 256
    },
    {
        "text": "Who won the latest presidential election in the US?",
        "seed": 123,
        "temperature": 0.7,
        "max_tokens": 256
    },
    {
        "text": "What will the weather be like in Tokyo tomorrow?",
        "seed": 456,
        "temperature": 0.7,
        "max_tokens": 256
    }
]

# Initialize basic session state
DEFAULT_SESSION_STATE = {
    "messages": [],
    "current_model": "TinyLlama-1.1B",
    "model_handler": None,
    "rag": None,
    "selected_sources": []
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("RAG Chatbot with Explanations")
st.divider()


# Sidebar controls
with st.sidebar:
    st.markdown("### RAG Configuration")
    enable_web_search = st.toggle("Enable Web Search", value=True)
    top_k_web = st.slider("Web Results", 0, 5, 2)
    
    st.divider()

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
    status_placeholder = st.empty()
    with status_placeholder.container():
        if selected_model in downloaded_models:
            st.success(f"Model {selected_model} is already downloaded")
        else:
            st.info(f"Model {selected_model} will be downloaded when selected")
        # Make status message disappear after 3 seconds
        time.sleep(3)
    status_placeholder.empty()
    
    # Model parameters
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_new_tokens = st.slider("Max New Tokens", 64, 1024, 256)
    seed = st.number_input("Seed", min_value=0, max_value=1000000, value=42)
    
    st.divider()

    st.markdown("### Explanation Configuration")
    enable_explanations = st.toggle("Enable Explanations", value=True)
    
    input_chunk_mode = st.selectbox(
        "Input Chunk Mode (Selection)",
        ["word", "sentence", "paragraph"],
        index=1,
        help="How to split the assistant's response for selection"
    )
    
    output_chunk_mode = st.selectbox(
        "Output Chunk Mode (Explanation)",
        ["word", "sentence", "paragraph"],
        index=1,
        help="How to analyze the context for explanations"
    )
    
    num_chunks = st.slider(
        "Number of Explanation Chunks",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of relevant chunks to highlight in the context"
    )


# Handle model initialization/switching
if (st.session_state.current_model != selected_model) or (st.session_state.model_handler is None):
    status_placeholder = st.empty()
    with status_placeholder.container():
        if selected_model in downloaded_models:
            st.success(f"Model {selected_model} is already downloaded")
        else:
            st.info(f"Model {selected_model} will be downloaded when selected")
    
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
            model_handler=model_handler,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            web_search_enabled=bool(bing_key),
            bing_subscription_key=bing_key
        )
        
        # Update session state
        st.session_state.model_handler = model_handler
        st.session_state.rag = rag
        st.session_state.current_model = selected_model
        
        success_placeholder = st.empty()
        success_placeholder.success(f"Model {selected_model} loaded successfully!")
        # Clear both status messages after a moment
        status_placeholder.empty()
        success_placeholder.empty()

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
            # None for label, rgba for color
            annotated.append((chunk, None, "rgba(90, 35, 170, 1)"))
            remaining_text = parts[1]
        else:
            sorted_chunks.pop(0)
    
    if remaining_text:  # Add any remaining text
        annotated.append(remaining_text)
    
    return annotated


def create_clickable_text(text: str, chunk_mode: str = "word") -> str:
    if chunk_mode == "sentence":
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
            
        clickable_elements = []
        for i, sentence in enumerate(sentences):
            if sentence:
                clickable_elements.append(
                    f'<a href="#" id="sentence_{i}" style="color: white;">{sentence}</a>'
                )
        
        return " ".join(clickable_elements)
    
    else:  # word mode
        words = text.split()
        clickable_words = []
        for i, word in enumerate(words):
            clickable_words.append(
                f'<a href="#" id="word_{i}" style="color: white;">{word}</a>'
            )
        
        return " ".join(clickable_words)


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
        return ["word"]
    elif mode == "sentence":
        return [".", "!", "?"]
    else:  # paragraph
        return ["\n\n"]


def handle_click_and_explanations(clicked_text: str, context: str, model_handler):
    """Handle click events and generate explanations for selected text"""
    if not clicked_text:
        return None
        
    # Get delimiters based on output chunk mode
    delimiters = get_delimiters_for_mode(output_chunk_mode)
    # Initialize explainer with context
    explainer = DocumentQAExplainer(
        model=model_handler,
        document=context,
        explanation_delimiters=delimiters,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        suppression_factor=0.0
    )
    
    # Run explainer on selected text
    output = explainer.run(
        question="Why is this part of the response relevant?",
        expected_answer=clicked_text
    )
    
    if output:
        # Get the specified number of most relevant chunks
        postprocessed = explainer.postprocess(output)
        relevant_chunks = [chunk['chunk'] for chunk in sorted(
            postprocessed.data, 
            key=lambda x: x['value'], 
            reverse=True
        )[:num_chunks]]
        
        return relevant_chunks
    return None


# Helper function to clean text
def clean_text(text: str) -> str:
    """Remove HTML tags and clean up text"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Replace &amp; with &
    text = text.replace('&amp;', '&')
    return text.strip()


def perform_rag_operation(query: str, top_k: int, web_only: bool, seed: int, temperature: float, max_tokens: int) -> tuple:
    """Perform RAG retrieval and response generation"""
    print("\n=== Starting Web Search ===")
    result = st.session_state.rag.retrieve_context(
        query=query,
        top_k=top_k,
        web_only=web_only
    )
    print(f"Found {len(result.contexts)} sources")
    
    # Clean the contexts
    cleaned_contexts = [clean_text(ctx) for ctx in result.contexts]
    context_text = "\n\n".join(cleaned_contexts) if cleaned_contexts else None
    
    print("\n=== Generating Response ===")
    print("Query:", query)
    print("Context length:", len(context_text) if context_text else 0)
    response = st.session_state.rag.generate_response(
        query=query,
        context=context_text,
        seed=seed,
        temperature=temperature,
        max_new_tokens=max_tokens
    )
    print("Response generated successfully")
    
    return response, context_text, result.sources


# Chat input and response generation
prompt = st.chat_input("What would you like to know?")

# Check for example prompt buttons
col1, col2 = st.columns(2)
with col1:
    for example in [example_prompts[0], example_prompts[2]]:
        if st.button(
            example["text"],
            use_container_width=True,
            type="secondary",
            key=f"example_button_{example['text'][:20]}"
        ):
            prompt = example["text"]
            seed = example["seed"]
            temperature = example["temperature"]
            max_new_tokens = example["max_tokens"]

with col2:
    for example in [example_prompts[1], example_prompts[3]]:
        if st.button(
            example["text"],
            use_container_width=True,
            type="secondary",
            key=f"example_button_{example['text'][:20]}"
        ):
            prompt = example["text"]
            seed = example["seed"]
            temperature = example["temperature"]
            max_new_tokens = example["max_tokens"]


# Add a note about RAG and explanations status
if any(msg["role"] == "user" and msg["content"] in [p["text"] for p in example_prompts] for msg in st.session_state.messages):
    st.info(
        "💡 Try toggling RAG (web search) and explanations in the sidebar to see how "
        "the responses change with different settings!"
    )

# Handle any prompt (typed or from example)
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message and spinner in chat flow
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
 
    response, context, sources = perform_rag_operation(
        query=prompt,
        top_k=top_k_web,
        web_only=enable_web_search,
        seed=seed,
        temperature=temperature,
        max_tokens=max_new_tokens
    )
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": context,
        "sources": sources
    })

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if enable_explanations:
                clicked = click_detector(
                    create_clickable_text(message["content"], input_chunk_mode),
                    key=f"clickable_{idx}"
                )
            else:
                st.write(message["content"])
            
            # Display context right after the assistant's message if it exists
            if message.get("context"):
                contexts = message["context"].split("\n\n")
                sources = message.get("sources", [])
                
                # Initialize selected_sources if needed
                if len(st.session_state.selected_sources) != len(contexts):
                    st.session_state.selected_sources = [True] * len(contexts)
                
                cols = st.columns(min(len(contexts), top_k_web))
                
                for src_idx, (ctx, source, col) in enumerate(zip(contexts, sources, cols)):
                    with col:
                        # Get metadata from source
                        title = source.get("metadata", {}).get("title", f"Source {src_idx + 1}")
                        url = source.get("metadata", {}).get("url", None)
                        domain = source.get("metadata", {}).get("domain", None)
                        
                        # Collapsible container for each source
                        with st.expander(title, expanded=True):
                            # Add checkbox inside expander
                            st.session_state.selected_sources[src_idx] = st.checkbox(
                                f"Explain",
                                value=st.session_state.selected_sources[src_idx],
                                key=f"source_checkbox_{idx}_{src_idx}"  # Include message idx in key
                            )
                            
                            if domain:
                                st.caption(f"From: {domain}")
                            if url:
                                st.markdown(f"[🔗 Source Link]({url})")
                            st.divider()
                            
                            # Display the content with explanations if enabled
                            if enable_explanations and clicked and st.session_state.selected_sources[src_idx]:
                                with st.spinner("🔍 Analyzing context..."):
                                    print(f"\n=== Running Explanation for Source {src_idx + 1} ===")
                                    print("Clicked Text:", clicked)
                                    print("Context length:", len(ctx))
                                    
                                    relevant_chunks = handle_click_and_explanations(
                                        clicked_text=clicked,
                                        context=ctx,
                                        model_handler=st.session_state.rag.llm
                                    )
                                    
                                    print("Found relevant chunks:", len(relevant_chunks) if relevant_chunks else 0)
                                    print("Relevant Chunks:", relevant_chunks)
                                    print("=" * 50)
                                
                                if relevant_chunks:
                                    annotated_chunks = create_annotated_text(ctx, relevant_chunks)
                                    annotated_text(*annotated_chunks)
                                else:
                                    st.markdown(ctx)
                            else:
                                st.markdown(ctx)
        else:
            st.write(message["content"])

