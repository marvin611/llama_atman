import torch

def manipulate_attention_scores(
        attention_scores,
        attention_mask,
        modified_causal_attention_mask,
        multiplicative = True,
        apply_softmax = False
    ):
        """
        Manipulates attention scores on each transformer layer as per the values of suppression_factors and suppression_token_indices.
        """
        batch_size = attention_scores.shape[0]
        
        assert (
            modified_causal_attention_mask.shape[0] == batch_size
        ), f"Expected modified_causal_attention_mask to have a batch size of {batch_size} but got shape: {modified_causal_attention_mask.shape}"

        if multiplicative:
            
            #attention_scores = attention_scores - attention_scores.min(-1).values.unsqueeze(3)
            
            ## apply modified mask
            attention_scores = attention_scores * modified_causal_attention_mask

            ## set all values where mask == True to a very low value so that they become 0 after softmax
            if attention_mask is not None:
                attention_scores.masked_fill_(
                    ~attention_mask.to(attention_scores.device), -10000.0
                )
        else:
            attention_scores = attention_scores + modified_causal_attention_mask

        return attention_scores