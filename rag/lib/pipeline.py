#! rag pipeline
from time import sleep

from lib.generate import TogetherGeneratorBase
from lib.retrieval import RetrieverBase
from langchain_core.documents import Document
def log(txt):
    with open("log.txt", "a") as o:
        o.write(f"{txt}\n")

class RAGPipeline:
    def __init__(self, retriever, generator, k, lang = None):
        self.retriever: RetrieverBase = retriever
        self.generator: TogetherGeneratorBase = generator
        self.k = k
        self.lang = lang

    def retrieval_pass(self, question):
        log("in retrieval pass")
        retrieved = self.retriever.query(self.generator.extract_objects(question), k=self.k, lang = self.lang)
        return {
            "retrieved": retrieved,
        }

    def generation_pass(self, question: str, documents: list[Document] | list[str]):
        cleaned_answer, model_output, generation_prompt= self.generator.answer_with_context(question, documents)
        return {
            "model_output": model_output,
            "cleaned_answer": cleaned_answer,
            "generation_prompt": generation_prompt, 
        }


    def detailed_pass(self, query):# -> tuple[Any, Any, Any]:
        assert False 
        if self.hypothetical_generation:
            stops = stopwords.words("english")
            query_tok = [word.lower() for word in word_tokenize(query) if word.lower() not in stops]
            if len(query_tok) < 3:
                docs = self.generator.augment_query(query)
                new_query = query + "\n" + docs
        retrieved = self.retriever.query(query if not self.hypothetical_generation else new_query, k=self.k)
        answer = self.generator.answer(query, retrieved)
        return answer, retrieved

    def __call__(self, query):
        answer, retrieved = self.detailed_pass(query)
        return answer
