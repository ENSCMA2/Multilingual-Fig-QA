from langchain_core.documents import Document
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import Together
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompt_values import ChatPromptValue
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from openai import OpenAI
from collections import Counter
from itertools import chain
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
# prompt engineering references
# https://www.promptingguide.ai/
# https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/


def log(txt):
    with open("log.txt", "a") as o:
        o.write(f"{txt}\n")

def format_documents(docs: list[Document] | list[str]):
    res = ['']
    for doc in docs:
        if type(doc) != str:
            res.append(doc.page_content)
        else:
            res.append(doc)
    res.append('')
    
    return "\n-----\n".join(res)

def extract_answer(output):
    if "1" in output:
        return 0
    return 1


class TogetherGeneratorBase():
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.client = OpenAI(
          api_key=api_key,
          base_url='https://api.together.xyz/v1',
        )
        self.system_prompt = f"You are completing a multiple-choice task. Answer with a single integer, 1 or 2."

    def __call__(self, system_prompt, user_prompt, max_tokens=20) -> str:
        predictions = self.client.chat.completions.create(messages = 
                                                        [{"role": "system",
                                                          "content": system_prompt,},
                                                         {"role": "user",
                                                          "content": user_prompt,}],
                                                     model=self.model_name,
                                                     max_tokens = 20).choices[0].message.content
        log(f"model output: {predictions}")
        ext = extract_answer(predictions)
        return ext

    def extract_objects(self, question, tok = False):
        if not tok:
            system_prompt = "You are a syntax parsing assistant. Generate a comma-separated list of answers."
            user_prompt = f'''Extract the objects of the three sentences below.
    {question}
    Express your answer in the following format: 'object1, object2, object3'.'''
            predictions = self.client.chat.completions.create(messages = 
                                                            [{"role": "system",
                                                              "content": system_prompt,},
                                                             {"role": "user",
                                                              "content": user_prompt,}],
                                                         model=self.model_name,
                                                         max_tokens = 40).choices[0].message.content
            log(f"objects: {predictions}")
            predictions = predictions.replace("Sure, I'd be happy to help! Here are the objects of the three sentences you provided:", "")
            predictions = predictions.strip("'").strip("\n").strip("\t")
            if "," in predictions:
                return predictions.split(",")
            return predictions.split()
        qs = question.split("\n")
        toks = []
        tok_list = chain(*[list(set(word_tokenize(q))) for q in qs])
        ctr = Counter(tok_list)
        return [t for t in ctr if ctr[t] == 1]

    def answer_with_context(self, question: str, documents: list[Document] | list[str]) -> tuple[str, str, str]:
        formatted_docs = format_documents(documents)
        prompt = f'''{question}
Here are some relevant documents that may help inform your answer.
{formatted_docs}'''
        output: str = self(self.system_prompt, prompt)
        return output, prompt
    