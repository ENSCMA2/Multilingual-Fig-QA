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
# prompt engineering references
# https://www.promptingguide.ai/
# https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/


def format_documents(docs: list[Document]):
    res = ['']
    for doc in docs:
        res.append(doc.page_content)
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
        self.system_prompt = f"You are completing a multiple-choice task. Answer with a single integer."

    def __call__(self, system_prompt, user_prompt, max_tokens=20) -> str:
        predictions = self.client.chat.completions.create(messages = 
                                                        [{"role": "system",
                                                          "content": sys,},
                                                         {"role": "user",
                                                          "content": u,}],
                                                     model=self.model_name,
                                                     max_tokens = 20,
                                                    ).choices[0].message.content
        return extract_answer(predictions)

    def answer_with_context(self, question: str, documents: list[Document]) -> tuple[str, str, str]:
        formatted_docs = format_documents(documents)
        prompt = f'''{question}
Here are some relevant documents that may help inform your answer.
{formatted_docs}'''
        output: str = self(self.system_prompt, prompt)
        return output, prompt
    