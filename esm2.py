from transformers import EsmModel, EsmTokenizer
import numpy as np
import torch


tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.eval()

def get_embeddings(sequence: str) -> np.ndarray:
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sequence_embeddings = outputs.last_hidden_state

    return sequence_embeddings