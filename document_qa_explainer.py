from explainer import Explainer
from logit_parsing import get_delta_cross_entropies
from outputs import DeltaCrossEntropiesOutput, DeltaLogitsOutput
import numpy as np

from typing import  Union
import matplotlib.pyplot as plt
from sentence_wise_suppression import SentenceWiseSuppression
from utils import dict_to_json

class DocumentQAExplanationOutput:
    """
    Container class for processed explanation outputs.
    """
    def __init__(self, postprocessed_output):
        """
        Initialize the DocumentQAExplanationOutput with the postprocessed output.
        """
        self.data = postprocessed_output
    
    def save_as(self, filename: str):
        """
        Save the explanation data to a JSON file.
        """
        dict_to_json(dictionary=self.data, filename=filename)

    def __repr__(self):
        """
        Return the string representation of the data.
        """
        return str(self.data)

class DocumentQAExplainer:
    """
    Initializes a DocumentQAExplainer instance with the necessary components for generating explanations.
    
    Parameters:
    - model: The language model instance with a tokenizer.
    - document (str): The input document text.
    - explanation_delimiters (str, optional): The delimiter for chunking text. Defaults to '\n'.
    - device (str, optional): The computing device to use. Defaults to 'cuda:0'.
    - suppression_factor (float, optional): The factor for attention suppression. Defaults to 0.0.
    """
    def __init__(
        self,
        model,
        document: str, 
        explanation_delimiters: str = '\n',
        device = 'cuda:0',
        suppression_factor = 0.0,
    ):
        
        # Initializes the Explainer instance with the provided model, device, and suppression factor.
        self.explainer = Explainer(
            model = model, 
            device = device, 
            tokenizer = model.tokenizer, 
            conceptual_suppression_threshold = None,
            suppression_factor = suppression_factor
        )
        
        # Stores the input document text.
        self.document = document

        # Instruction to be displayed before the document.
        self.instruction_before_document = 'Read the contents of the document carefully and answer the questions below.\n'
        
        # Prepares the prompt prefix by combining the instruction and the document.
        self.prompt_prefix = f'''{self.instruction_before_document}{self.document}'''
        
        # Initializes SentenceWiseSuppression for sentence-level chunking.
        self.sentence_wise_suppression = SentenceWiseSuppression(
            prompt = self.prompt_prefix,
            tokenizer_encode_fn=model.tokenizer.encode,
            tokenizer_decode_fn= model.tokenizer.decode,
            delimiters= explanation_delimiters ## for now one token only
        )

        # Generates custom chunk configuration based on the suppression factor.
        self.custom_chunk_config = self.sentence_wise_suppression.get_config(self.explainer.suppression_factor)
        # Instruction to be displayed after the document.
        self.instruction_after_document = 'Now answer the question below based on the document given above.'

    def get_prompt(self, question: str) -> List[str]:
        """
        Formats the input document and question into a prompt for generating explanations.

        Parameters:
        - question (str): The question to be answered based on the document.

        Returns:
        - List[str]: A list containing the formatted prompt string.
        """
        prompt = [
            f'''{self.prompt_prefix}
Question: {question}
Answer:'''
        ]
        return prompt

    def run(
        self,
        question: str, 
        expected_answer: str,
        max_batch_size: int = 25
    ):
        """
        Generates explanation data for a question-answer pair by manipulating attention and calculating delta cross-entropies.

        Parameters:
        - question (str): The question to be answered based on the document.
        - expected_answer (str): The expected answer to the question.
        - max_batch_size (int, optional): The maximum batch size for processing. Defaults to 25.

        Returns:
        - output: The raw explanation output data, including delta cross-entropies for each chunk.
        """
        prompt = self.get_prompt(question=question)
                
        logit_outputs = self.explainer.collect_logits_by_manipulating_attention(
            prompt = prompt.copy(),
            target = expected_answer,
            max_batch_size=max_batch_size,
            configs = self.custom_chunk_config
        )
        
        output = get_delta_cross_entropies(
            output = logit_outputs,
            square_outputs = True,
            custom_chunks = True
        )
        
        return output

    def postprocess(self, output: Union[DeltaLogitsOutput, DeltaCrossEntropiesOutput]) -> DocumentQAExplanationOutput:
        """
        Postprocesses the raw explanation output to a more readable format.

        This method takes the raw output from the explanation generation process and transforms it into a more interpretable format. It extracts the split prompt from the sentence-wise suppression module, calculates the total importance values for each chunk, and pairs each chunk with its corresponding total value.

        Parameters:
        - output (Union[DeltaLogitsOutput, DeltaCrossEntropiesOutput]): The raw output from the explanation generation process.

        Returns:
        - DocumentQAExplanationOutput: The postprocessed output, containing a list of dictionaries where each dictionary represents a chunk with its total importance value.
        """
        split_prompt =  self.sentence_wise_suppression.get_split_prompt()
        chunks = None
        
        all_values = []
        for target_token_idx in range(len(output.data)):
            single_explanation = output.data[target_token_idx]

            values = np.array([x['value'] for x in single_explanation['explanation']])
            all_values.append(values)
            if chunks is None:
                # chunks = [
                #     f'Chunk {i}: {split_prompt[i]}'.replace('.', '.\n') for i in range(len(split_prompt))
                # ]
                chunks = [
                    f'{split_prompt[i]}'.replace('.', '.\n') for i in range(len(split_prompt))
                ]

        data = []

        for chunk, value in zip(chunks, sum(all_values)):
            x = {
                "chunk":chunk,
                "value":value
            }
            data.append(x)

        return DocumentQAExplanationOutput(postprocessed_output=data)

    def get_chunks_and_values(self, postprocessed_output: DocumentQAExplanationOutput) -> Tuple[List[str], List[float]]:
        """
        Extracts chunks and their corresponding importance values from the postprocessed output.

        This method takes the postprocessed output from the explanation generation process and extracts the chunks and their corresponding importance values.

        Parameters:
        - postprocessed_output (DocumentQAExplanationOutput): The postprocessed output from the explanation generation process.

        Returns:
        - Tuple[List[str], List[float]]: A tuple containing a list of chunks and a list of their corresponding importance values.
        """
        chunks = [x['chunk'] for x in postprocessed_output.data]
        values = [x['value'] for x in postprocessed_output.data]

        return chunks, values

    def show_output(self, output, question, expected_answer, save_as: str = None, figsize = (15, 11), fontsize= 20):
        """
        Visualizes the explanation output as a horizontal bar chart.

        This method takes the raw explanation output, processes it, and visualizes the importance values of each chunk as a horizontal bar chart. The chart displays the chunks on the y-axis and their corresponding importance values on the x-axis. The title of the chart includes the question and the expected answer.

        Parameters:
        - output: The raw output from the explanation generation process.
        - question (str): The question for which the explanation is generated.
        - expected_answer (str): The expected answer to the question.
        - save_as (str, optional): The file path to save the chart. If None, the chart is not saved. Defaults to None.
        - figsize (tuple, optional): The size of the figure. Defaults to (15, 11).
        - fontsize (int, optional): The font size of the text in the chart. Defaults to 20.

        Returns:
        - None
        """
        postprocessed_output = self.postprocess(output)
        chunks, values = self.get_chunks_and_values(postprocessed_output=postprocessed_output)

        fig = plt.figure(figsize=figsize)
        plt.barh(chunks[::-1], values[::-1])  # Reversed to read from top to bottom
        plt.yticks(rotation=0)
        plt.xticks(fontsize=fontsize)
        plt.grid()
        plt.title(f'Question: {question}\nAnswer: {expected_answer}', fontsize=fontsize)
        plt.show()

        if save_as is not None:
            plt.tight_layout()
            fig.savefig(save_as)
            print('saved:', save_as)