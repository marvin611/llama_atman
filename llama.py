import torch
from transformers import AutoTokenizer, LlamaConfig, AutoConfig
from modeling_llama import LlamaForCausalLM
from huggingface_hub import snapshot_download, list_models
from safetensors.torch import load_file
import traceback
from typing import Optional, List, Union, Tuple
from pathlib import Path
import copy
import os

class Llama:
    # List of supported models - could be expanded
    SUPPORTED_MODELS = {
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Llama-3-DiscoLeo-8B-32k-Instruct": "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1",
        
    }

    def __init__(
        self,
        model_name: str = "TinyLlama-1.1B",
        models_dir: str = "./models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Llama model handler
        Args:
            model_name: Name of the model from SUPPORTED_MODELS
            models_dir: Base directory for all models
            device: Device to run model on
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.repo_id = self.SUPPORTED_MODELS[model_name]
        self.models_dir = Path(models_dir)
        self.model_dir = self.models_dir / self.model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return list of supported model names"""
        return list(cls.SUPPORTED_MODELS.keys())

    @classmethod
    def list_downloaded_models(cls, models_dir: str = "./models") -> List[str]:
        """Return list of already downloaded models"""
        models_path = Path(models_dir)
        if not models_path.exists():
            return []
        return [d.name for d in models_path.iterdir() if d.is_dir() and d.name in cls.SUPPORTED_MODELS]

    def is_model_downloaded(self) -> bool:
        """Check if model files exist locally"""
        required_files = ["config.json", "tokenizer.json", "model.safetensors"]
        return all((self.model_dir / file).exists() for file in required_files)

    def _download_model(self) -> None:
        """Download model files if not present."""
        try:
            if not self.is_model_downloaded():
                print(f"Downloading {self.model_name} from Hugging Face Hub...")
                self.model_dir.mkdir(parents=True, exist_ok=True)
                
                snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Model files downloaded to {self.model_dir}")
            else:
                print(f"Model files already present at {self.model_dir}")
        except Exception as e:
            raise Exception(f"Error downloading model: {e}")

    def setup(self) -> bool:
        """
        Set up the complete model pipeline.
        Returns True if successful, False otherwise.
        """
        try:
            self._download_model()
            self._load_config()
            self._load_tokenizer()
            self._initialize_model()
            self._load_weights()
            return True
        except Exception as e:
            print(f"Error during setup: {e}")
            traceback.print_exc()
            return False

    def _load_config(self) -> None:
        """Load model configuration."""
        try:
            self.config = LlamaConfig.from_pretrained(self.model_dir)
            self.config._attn_implementation = "sdpa"
            print("Config loaded successfully:")
            print(self.config)
        except Exception as e:
            raise Exception(f"Error loading config: {e}")

    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print("Tokenizer loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading tokenizer: {e}")

    def _initialize_model(self) -> None:
        """Initialize the model with config."""
        try:
            self.model = LlamaForCausalLM(self.config).to(self.device)
            print("Model initialized successfully")
        except Exception as e:
            raise Exception(f"Error initializing model: {e}")

    def _load_weights(self) -> None:
        """Load pre-trained weights."""
        try:
            state_dict = load_file(self.model_dir / "model.safetensors")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            print("Weights loaded successfully")
            if missing_keys:
                print("Missing keys:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
        except Exception as e:
            raise Exception(f"Error loading weights: {e}")

    def preprocess_inputs(self, input_list: Union[str, List[str]], embed: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Preprocesses text inputs for the Llama model. This method can handle both single string inputs and lists of strings.
        It tokenizes the inputs, applies padding and truncation, and optionally embeds the tokenized inputs.

        Args:
            input_list (Union[str, List[str]]): The input text(s) to be preprocessed. Can be a single string or a list of strings.
            embed (bool, optional): If True, the method will embed the tokenized inputs. Defaults to True.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The preprocessed inputs. If `embed` is True, returns the embedded tensors. Otherwise, returns the tensor of token IDs.
        """
        # Convert single string input to a list for consistency
        if isinstance(input_list, str):
            input_list = [input_list]
            
        # Deep copy the input list to ensure original data is not modified
        input_list = copy.deepcopy(input_list)
        processed_inputs = []
        
        # Iterate over each input in the list
        for i, inp in enumerate(input_list):
            # Check if the input is a string
            if isinstance(inp, str):
                # Tokenize the input string
                tokens = self.tokenizer(
                    inp,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_position_embeddings
                ).input_ids
                processed_inputs.append(tokens)
            else:
                # Raise an error if the input is not a string
                raise ValueError(f'Invalid input type: {type(inp)}. Only string inputs are supported.')

        # If there are multiple inputs, stack them into a single tensor
        if len(processed_inputs) > 1:
            processed_inputs = torch.cat(processed_inputs, dim=0)
        else:
            processed_inputs = processed_inputs[0]  # This will maintain the 2D shape

        # If embed is True, embed the processed inputs
        if embed:
            return self.embed(processed_inputs)  # Wrap in list for embed method
        
        return processed_inputs  # Return tensor of token IDs when embed=False

    def embed(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts a list of tokenized inputs to their corresponding embeddings.

        This method iterates over each input tensor in the list, ensuring it has a 2D shape (batch_size, sequence_length) by adding a batch dimension if necessary. It then moves each tensor to the device specified by `self.device` and uses the model's embedding layer to convert the tokens to embeddings. If there are multiple inputs, the method concatenates the embeddings along the sequence dimension before returning the result.

        Args:
            inputs (List[torch.Tensor]): A list of tokenized input tensors to be embedded.

        Returns:
            torch.Tensor: The embedded tensors, either a single tensor if there's one input or a concatenated tensor if there are multiple inputs.
        """
        embedded_outputs = []
        
        for x in inputs:
            # Validate input type
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(x)}")
            
            # Ensure input tensor has a 2D shape [batch_size, sequence_length]
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            elif x.dim() != 2:
                raise ValueError(f"Expected 2D tensor (batch_size, sequence_length), got {x.dim()}D")
                
            # Move input tensor to the device specified by self.device
            x = x.to(self.device)
            
            # Convert tokens to embeddings using the model's embedding layer
            with torch.no_grad():
                embedded = self.model.model.embed_tokens(x)
                embedded_outputs.append(embedded)
        
        # Concatenate embeddings along the sequence dimension if there are multiple inputs
        if len(embedded_outputs) > 1:
            return torch.cat(embedded_outputs, dim=1)
        return embedded_outputs[0]

    @property
    def word_embedding(self):
        """
        Property to access the word embedding layer directly.
        Useful for compatibility with existing code.
        """
        return self.model.model.embed_tokens

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generates text based on the input provided. This method formats the input text for chat or instruction, loads a pre-trained language model, and uses it to generate text. The generated text is then cleaned up by removing the formatting and the original input text if it appears at the start of the response.

        Args:
            input_text (str): The text to use as input for generation.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): The temperature for sampling. Defaults to 0.7.
            do_sample (bool, optional): Whether to use sampling or greedy decoding. Defaults to True.
            seed (Optional[int], optional): The seed for reproducibility. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's generate method.

        Returns:
            str: The generated text response.
        """
        try:
            if seed is not None:
                torch.manual_seed(seed)
            
            from transformers import AutoModelForCausalLM
            
            # Format input for chat/instruction
            formatted_input = f"<|system|>You are a helpful assistant.<|user|>{input_text}<|assistant|>"
            
            temp_model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map=self.device,
                trust_remote_code=True
            ).to(self.device)
            
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            outputs = temp_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response by removing the instruction format
            response = response.replace("<|system|>You are a helpful assistant.", "")
            response = response.replace("<|user|>", "")
            response = response.replace("<|assistant|>", "")
            
            # Remove the original input text
            if response.startswith(input_text):
                response = response[len(input_text):].strip()
            
            return response.strip()
            
        except Exception as e:
            raise Exception(f"Error during generation: {e}")
