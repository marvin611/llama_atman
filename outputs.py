import json
import numpy as np
from utils import load_json_as_dict

class DeltaCrossEntropiesOutput:
    """
    Represents the output of delta cross-entropies calculations, providing methods to save, load, and process the data.
    """

    def __init__(self, data: dict):
        """
        Initializes the DeltaCrossEntropiesOutput object with the given data.

        Args:
            data (dict): The data to be stored in the object.
        """
        self.data = data
        self.num_tokens_in_image = 144  # Assuming a fixed number of tokens in an image for processing

    def save(self, filename: str):
        """
        Saves the data to a JSON file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, indent=4)
        print(f'saved: {filename}')

    def get_text_heatmap(self, target_token_idx: int, square_outputs=False):
        """
        Generates a heatmap for the specified target token index.

        Args:
            target_token_idx (int): The index of the target token for which to generate the heatmap.
            square_outputs (bool, optional): If True, squares the output values before returning. Defaults to False.

        Returns:
            dict: A dictionary containing the target token ID and the heatmap.
        """
        selected_item = self.data[target_token_idx]
        target_token_id = selected_item['target_token_id']
        heatmap = np.array([x['value'] for x in selected_item['explanation']])

        if square_outputs:
            heatmap = heatmap**2

        return {
            'token_id': target_token_id,
            'heatmap': heatmap
        }

    @classmethod
    def from_file(cls, filename: str):
        """
        Loads data from a JSON file and returns a DeltaCrossEntropiesOutput object.

        Args:
            filename (str): The name of the file to load the data from.

        Returns:
            DeltaCrossEntropiesOutput: An instance of DeltaCrossEntropiesOutput with the loaded data.
        """
        data = load_json_as_dict(filename=filename)
        return cls(data=data)

        

    def show_image(self, image_token_start_idx: int, target_token_idx: int) -> np.ndarray:
        """
        Generates a 12x12 heatmap for the specified image tokens based on their explanation values.

        Args:
            image_token_start_idx (int): The starting index of the image tokens in the explanation.
            target_token_idx (int): The index of the target token for which to generate the heatmap.

        Returns:
            np.ndarray: A 12x12 numpy array representing the heatmap for the image tokens.
        """
        x = self.data[target_token_idx]
        image_explanation_values = [
            x['value'] for x in x['explanation'][image_token_start_idx:image_token_start_idx+self.num_tokens_in_image]
        ]
        heatmap = np.array(image_explanation_values).reshape(12,12)
        
        return heatmap
    
class DeltaLogitsOutput:
    """
    Class to represent the output of delta logits.
    """
    def __init__(
        self,
        data: dict
    ):
        self.data = data
        self.num_tokens_in_image = 144

    def save(self, filename: str):
        """
        Saves the data to a JSON file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, indent = 4)
        print(f'saved: {filename}')

    def get_text_heatmap(self, target_token_idx: int, square_outputs = False):
        """
        Returns the heatmap for the specified target token.

        Args:
            target_token_idx (int): The index of the target token for which to generate the heatmap.
            square_outputs (bool, optional): Whether to square the output differences. Defaults to False.
        """
        selected_item = self.data[target_token_idx]
        target_token_id = selected_item['target_token_id']
        heatmap = np.array(
                [
                x['value'] for x in selected_item['explanation']
            ]
        )

        if square_outputs is True:
            heatmap = heatmap**2

        return {
            'token_id':target_token_id,
            'heatmap': heatmap
        }

    @classmethod
    def from_file(cls, filename: str):
        """
        Loads the data from a JSON file and returns a DeltaLogitsOutput object.

        Args:
            filename (str): The name of the file to load the data from.

        Returns:
            DeltaLogitsOutput: An instance of DeltaLogitsOutput with the loaded data.
        """
        data = load_json_as_dict(
            filename = filename
        )

        return cls(data = data)

        

    def show_image(self, image_token_start_idx, target_token_idx: int):
        """
        Generates a 12x12 heatmap for the specified image tokens.

        Args:
            image_token_start_idx (int): The starting index of the image tokens in the explanation.
            target_token_idx (int): The index of the target token for which to generate the heatmap.
        """
        x = self.data[target_token_idx]
        image_explanation_values = [
            x['value'] for x in x['explanation'][image_token_start_idx:image_token_start_idx+self.num_tokens_in_image]
        ]
        heatmap = np.array(image_explanation_values).reshape(12,12)
        
        return heatmap