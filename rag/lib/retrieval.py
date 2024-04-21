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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

class RetrieverBase:
    def __init__(self) -> None:
        
        self.docs: list[Document]
        pass
    
    def query(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
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
    ) -> list[Document]:

        res = self.bm25_retriever.get_relevant_documents(query)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res

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
    ) -> list[Document]:

        res = self.db.similarity_search(query, k=k)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res
