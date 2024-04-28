#! retrieval system
from typing import Any, List, Optional, Sequence
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import warnings
from pprint import pprint
warnings.filterwarnings(action="ignore", message=".*a chunk of size.*")
import os
import sys
from ragatouille import RAGPretrainedModel
from lib.preprocess import DatasetBase
import langchain_community
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import requests
import string
import json

class RetrieverBase:
    def __init__(self) -> None:
        
        self.docs: list[Document]
        pass
    
    def query(
        self,
        query: str | list[str],
        k: int = 5,
        lang: str = None,
        model = None
    ) -> list[Document] | list[str]:
        raise NotImplementedError

class BM25Retriever(RetrieverBase):
    def __init__(
        self, 
        dataset: DatasetBase
    ):
        self.dataset = dataset
        self.docs = self.dataset.gt_split_docs()
        self.bm25_retriever = langchain_community.retrievers.BM25Retriever.from_documents(self.docs, k=5)

    def query(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False,
        lang: str = None
    ) -> list[Document]:

        res = self.bm25_retriever.get_relevant_documents(query)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res

class WikidataRetriever(RetrieverBase):
    def __init__(self):
        pass

    def query(self, query, lang, k = None,):
        results = []
        for q in query:
            clean = q.strip()
            clean = "".join([char for char in clean if char not in string.punctuation])
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}&language={lang}&format=json"
            result = json.loads(requests.get(url).text)["search"]
            if len(result) > 0:
                results.extend([f"{q}: {item['description']}" for item in result])
        print("wiki results:", results)
        return results

class VectorRetriever(RetrieverBase):
    def __init__(
        self, 
        dataset: DatasetBase,
        model_name: str = "intfloat/e5-large-v2",
    ):
        self.dataset = dataset
        self.docs = self.dataset.gt_split_docs()
        self.model_name = model_name
        
        self.hf_emb = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': 'cpu' if not torch.cuda.is_available() else 'cuda'},
        )
        print("making chroma db")
        self.db = Chroma.from_documents(self.docs, self.hf_emb)

    def query(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False,
        lang: str = None
    ) -> list[Document]:
        print("in retrieval pass")

        res = self.db.similarity_search(query, k=k)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res
