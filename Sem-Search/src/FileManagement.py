from .BERT import BERT_TOKEN, BERT_MODEL, bert_embed
from typing import Generator,Iterable
import os, torch, pickle
from tqdm import tqdm

def dict_chunks(dict: dict[any,Iterable], size: int = 512) -> Generator[None,dict[any,Iterable],None]:
    """Takes a dictionary and yields it in chunks of length size. Each value should be the same length."""
    length = next(iter(dict.values())).shape[-1] # The length of each value
    for start in range(0,length,int(size/2)):
        end = min(length, start + size)
        yield {key: val[:,start:end] for key, val in dict.items()}

def split_text(text: str) -> list[str]:
    for delim in [".",".|.|.|","?","!"]:
        text = f"{delim.replace('|','')}|".join(text.split(delim))
    return text.split("|")

def dir_emb_cache_path(dir_path: str) -> str:
    return os.path.join(dir_path,f"cache_{BERT_MODEL.name.replace('/','_')}.emb")

def embed_or_load_dir(dir_path: str) -> dict[str,torch.Tensor]:
    if os.path.isfile(dir_emb_cache_path(dir_path)):
        return load_dir_emb(dir_path)
    else:
        return embed_dir2(dir_path)

@torch.no_grad()
def embed_dir(dir_path: str, cache: bool = True) -> dict[str,torch.Tensor]:
    """Create the embedding dictionary for a directory."""
    embedded_documents = {}
    for file in tqdm(os.listdir(dir_path),colour="magenta",desc="Embedding Documents"):
        if not file.endswith("txt"): continue
        embeddings = []
        with open(os.path.join(dir_path,file)) as f:
            doc = f.read()
            doc_embs = BERT_TOKEN(doc,return_tensors="pt")
            for chunk in dict_chunks(doc_embs):
                emb = BERT_MODEL(**chunk)["pooler_output"]
                embeddings.append(emb)
        embedded_documents[file] = torch.concat(embeddings)
    if cache:
        cache_path = dir_emb_cache_path(dir_path)
        with open(cache_path,"wb") as cache_file:
            pickle.dump(embedded_documents,cache_file)
    return embedded_documents

@torch.no_grad()
def embed_dir2(dir_path: str, cache: bool = True) -> dict[str,torch.Tensor]:
    """Create the embedding dictionary for a directory."""
    embedded_documents = {}
    for file in tqdm(os.listdir(dir_path),colour="magenta",desc="Embedding Documents"):
        if not file.endswith("txt"): continue
        with open(os.path.join(dir_path,file)) as f:
            doc = f.read()
        sentences = split_text(doc)
        embedded_documents[file] = bert_embed(sentences)
    if cache:
        cache_path = dir_emb_cache_path(dir_path)
        with open(cache_path,"wb") as cache_file:
            pickle.dump(embedded_documents,cache_file)
    return embedded_documents

def load_dir_emb(dir_path) -> dict[str,torch.Tensor]:
    cache_path = dir_emb_cache_path(dir_path)
    with open(cache_path,"rb") as cache_file:
        embedded_documents = pickle.load(cache_file)
    return embedded_documents