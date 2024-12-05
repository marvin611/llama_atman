import torch

def manipulate_attention_scores(
        attention_scores,
        attention_mask,
        modified_causal_attention_mask,
        multiplicative: bool = True,
        apply_softmax: bool = False
    ) -> torch.Tensor:
        """
        Manipulates attention scores on each transformer layer based on the provided masks and parameters.

        This function applies a custom mask to the attention scores, either multiplicatively or additively, depending on the `multiplicative` parameter. Additionally, it sets masked positions to a very low value (-10000.0) to effectively zero them out after softmax application, if `attention_mask` is provided.

        Args:
            attention_scores (torch.Tensor): The original attention scores tensor.
            attention_mask (torch.Tensor, optional): The standard attention mask for padding. Defaults to None.
            modified_causal_attention_mask (torch.Tensor): The custom mask for attention manipulation.
            multiplicative (bool, optional): Whether to use multiplicative (True) or additive (False) manipulation. Defaults to True.
            apply_softmax (bool, optional): Whether to apply softmax after manipulation. Defaults to False.

        Returns:
            torch.Tensor: The modified attention scores tensor.
        """
        batch_size = attention_scores.shape[0]
        
        # Verify that the batch size of modified_causal_attention_mask matches the batch size of attention_scores
        assert (
            modified_causal_attention_mask.shape[0] == batch_size
        ), f"Expected modified_causal_attention_mask to have a batch size of {batch_size} but got shape: {modified_causal_attention_mask.shape}"

        if multiplicative:
            # Apply the modified causal attention mask multiplicatively
            attention_scores = attention_scores * modified_causal_attention_mask
            
            # If attention_mask is provided, set all values where mask == True to a very low value (-10000.0)
            # This effectively zeros them out after softmax application
            if attention_mask is not None:
                attention_scores.masked_fill_(
                    ~attention_mask.to(attention_scores.device), -10000.0
                )
        else:
            # Apply the modified causal attention mask additively
            attention_scores = attention_scores + modified_causal_attention_mask

        # Apply softmax if specified
        if apply_softmax:
            attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores