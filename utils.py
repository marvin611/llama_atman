import torch
import json 
import os

@torch.no_grad()
def get_output_logits_from_input_ids(
    model,
    input_ids,  ## input_ids
):
    """
    Computes the output logits from input IDs using the given model.

    Args:
        model (torch.nn.Module): The model to use for computing the output logits.
        input_ids (torch.Tensor): The input IDs tensor.

    Returns:
        torch.Tensor: The output logits tensor.
    """
    output = model(input_ids.to(model.device))

    return output.logits

@torch.no_grad()
def get_output_logits_from_embeddings(
    model,
    embeddings,  ## input_ids
):
    """
    Computes the output logits from embeddings using the given model.

    Args:
        model (torch.nn.Module): The model to use for computing the output logits.
        embeddings (torch.Tensor): The embeddings tensor.

    Returns:
        torch.Tensor: The output logits tensor.
    """
    assert embeddings.ndim == 3, f'Expected a tensor with 3 dims for embeddings, but got shape: {embeddings.shape}'
    output = model.model.forward(inputs_embeds = embeddings.to(model.device))

    return output.logits


def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

def split_str_into_tokens(x: str, tokenizer, numbered = False):
    """Splits a string to it's respective tokens

    Args:
        `x` (`str`): input text
        `tokenizer`: tokenizer for the model
        `numbered` (`bool`, `optional`): If set to true, adds an index number in the beginning of each item in the output. Defaults to False.

    Returns:
        list of strings
    """
    prompt_split_into_tokens = [
        tokenizer.decode([token_id]) for token_id in tokenizer.encode(x)
    ]

    if numbered:
        prompt_split_into_tokens = [
            f'{i}:{prompt_split_into_tokens[i]}' for i in range(len(prompt_split_into_tokens))
        ]
    return prompt_split_into_tokens

def parse_text_result(
    result, 
    target_token_index = 0, 
):
    """
    Parses explanation results for a specific target token.

    Args:
        result (list): The list of results to parse.
        target_token_index (int, optional): The index of the target token for which to parse the result. Defaults to 0.

    Returns:
        dict: A dictionary containing the values and the target token.
    """
    selected_result = result[target_token_index]
    explanations = selected_result["explanation"]
    target_token = selected_result['target_token_id']

    assert len(explanations) != 0, "length of explanations is 0, you probably cropped it too much"


    values = [
        explanations[i]["value"] for i in range(len(explanations))
    ]

    return {
        'values':values,
        'target_token':target_token
    }

def normalize_tensor(self, x, eps=1e-8):
    """
    values are shifted and rescaled so that they end up ranging between 0 and 1
    """
    if len(x) > 1:
        normalized_tensor = (x - x.min()) / (x.max() - x.min()) + eps
    else:
        ## no normalization when there's only one item in x
        # with only one item we know that this item will have 100% importance
        normalized_tensor = torch.tensor([1.0], dtype=x.dtype, device=x.device)
    return normalized_tensor


def load_json_as_dict(filename: str):
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
    except json.decoder.JSONDecodeError:
        raise AssertionError(f'Error reading filename: {filename}')

    return data


def dict_to_json(dictionary, filename):
    import json
    print(f"\nSaving to JSON:")
    print(f"Number of entries to save: {len(dictionary)}")
    for i, entry in enumerate(dictionary):
        print(f"Entry {i}: {entry}")
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)
    
    # Verify what was saved
    with open(filename, 'r') as f:
        saved_data = json.load(f)
    print(f"\nVerified saved data:")
    print(f"Number of entries in file: {len(saved_data)}")

def create_folder_if_does_not_exist(folder: str):
    if os.path.exists(folder) is False:
        print(f"making folder: {folder}")
        os.mkdir(folder)
    else:
        pass