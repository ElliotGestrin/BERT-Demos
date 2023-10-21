from transformers import AutoModel,AutoTokenizer
from typing import Literal
import torch

@torch.no_grad()
def new_bert(name: str):
    """Switch the global BERT objects to new ones."""
    global BERT_MODEL, BERT_TOKEN
    BERT_MODEL = AutoModel.from_pretrained(name)
    BERT_TOKEN = AutoTokenizer.from_pretrained(name)
    BERT_MODEL.name = name
    BERT_TOKEN.name = name

@torch.no_grad()
def similarity(A: torch.Tensor, B:torch.Tensor)-> torch.Tensor:
    """
    Return a similarity matrix between each row in 2d tensors A & B. 
    1 = Most similar, -1 = Most dissimilar. Shape is A[0]xB[0].
    """
    return dist_similarity(A,B)

@torch.no_grad()
def dist_similarity(A: torch.Tensor,  B:torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Return a distance based similarity matrix between each row in 2d tensors A & B. 
    Large is similar, small is dissimilar. Shape is A[0]xB[0].
    """
    differences = A.unsqueeze(1) - B  # Create a matrix of differences
    sim = -torch.norm(differences, dim=2)  # Compute the norm along dimension 2
    if normalize:
        sim = sim*2/(sim.max() - sim.min()) # Normalize to span of length 2
        sim = sim - sim.max() + 1 # Set max to 1, which also gives min of -1
    return sim

@torch.no_grad()
def cosine_similarity(A: torch.Tensor,  B:torch.Tensor):
    """
    Return a cosine similarity matrix between each row in 2d tensors A & B. 
    1 = Similar, -1 = Dissimilar. Shape is A[0]xB[0].
    """    
    A_norm = A.square().sum(-1).sqrt()
    B_norm = B.square().sum(-1).sqrt()
    corr = A @ B.T
    corr_norm = (corr.T / A_norm).T / B_norm # Correctly broadcast the norms
    return corr_norm

import torch
from typing import Literal

# Define a function to calculate the similarity between a given term and a set of documents
@torch.no_grad()
def term_similarity(
    pos_term: str, 
    embedded_documents: dict[str,torch.Tensor], 
    neg_term: str = None, 
    neg_weight: float = 0.5,
    strategy: Literal["max","mean","top3"] = "max"
) -> dict[str, float]:
    """
    Calculate the similarity between a given term and a set of documents.

    Args:
        pos_term (str): The positive term to compare against the documents.
        embedded_documents (dict[str,torch.Tensor]): A dictionary of document names and their corresponding embeddings.
        neg_term (str, optional): The negative term to compare against the documents. Defaults to None.
        neg_weight (float, optional): The weight to give to the negative term. Defaults to 0.5.
        strategy (Literal["max","mean","top3"], optional): The strategy to use for calculating document similarity. 
            Can be one of "max", "mean", "top3". Defaults to "max".

    Returns:
        dict[str, float]: A dictionary of document names and their corresponding similarity scores.
    """
    # Embed the positive and negative terms
    to_emb = [pos_term, neg_term] if neg_term else [pos_term]
    term_embs = bert_embed(to_emb)

    # Calculate the similarity between the terms and each document
    full_sim = {}
    for doc_name, doc_emb in embedded_documents.items():
        term_doc_sim = similarity(term_embs,doc_emb)
        # If a negative term is given, remove it and renormalise
        if neg_term: 
            term_doc_sim = (term_doc_sim[0,:] - neg_weight*term_doc_sim[1,:])/(1+neg_weight)
        full_sim[doc_name] = term_doc_sim

    # Calculate the document similarity based on the chosen strategy
    match strategy.lower():
        case "max":  
            sum_sim = {key: sims.max().item()  for key, sims in full_sim.items()}
        case "mean": 
            sum_sim = {key: sims.mean().item() for key, sims in full_sim.items()}
        case "top3": 
            sum_sim = {
                key: 
                    sims.topk(3).values.mean().item() if len(sims) >= 3 else 
                    sims.mean().item() 
                for key, sims in full_sim.items()
            }
        case _: 
            raise ValueError(f"{strategy} is not a valid document similarity strategy")

    # Normalise the similarity scores to be between -1 and 1
    max_val, min_val = max(sum_sim.values()), min(sum_sim.values())
    offset = max_val/(max_val - min_val)-1
    norm_sim = {key: val/(max_val - min_val) - offset for key, val in sum_sim.items()}
    return norm_sim

@torch.no_grad()
def mean_pooling(output, attention):
    """
    Performs mean pooling on the output of a BERT model.

    Args:
        output (torch.Tensor): The output tensor from the BERT model.
        attention (torch.Tensor): The attention tensor from the BERT model.

    Returns:
        torch.Tensor: The mean pooled tensor.
    """
    token_emb = output[0]
    input_mask = attention.unsqueeze(-1).expand(token_emb.size()).float()
    return torch.sum(token_emb * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

@torch.no_grad()
def bert_embed(to_emb: list[str]):
    """Return the SBERT-embedding for each element in to_emb."""
    tokens = BERT_TOKEN(to_emb,padding=True,return_tensors="pt",truncation=True,max_length=380)
    output = BERT_MODEL(**tokens)
    embeddings = mean_pooling(output, tokens["attention_mask"])
    return BERT_MODEL(**BERT_TOKEN(to_emb,padding=True,return_tensors="pt"))["pooler_output"]

new_bert("KBLab/sentence-bert-swedish-cased")