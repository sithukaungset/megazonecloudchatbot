# from gpt_index import SimpleDirectoryReader, GPTListIndex,readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from gpt_index import SimpleDirectoryReader, GPTListIndex,readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from types import FunctionType
from llama_index import ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader, load_index_from_storage
import sys
import os
import time
import openai
from dotenv import load_dotenv
from llama_index.node_parser import SimpleNodeParser
from langchain.llms import AzureOpenAI


from llama_index import StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
parser = SimpleNodeParser()
# Load env variables (create .env with OPENAI_API_KEY and OPENAI_API_BASE)
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 500
    max_chunk_overlap = 256
    chunk_size_limit = 1024

    print("*"*5, "Documents parsing initiated", "*"*5)
    def file_metadata(x): return {"filename": x}
    reader = SimpleDirectoryReader(directory_path, file_metadata=file_metadata)
    documents = reader.load_data()

    # nodes = parser.get_nodes_from_documents(documents)
    # index = GPTVectorStoreIndex(nodes)
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=AzureOpenAI(
        temperature=0, deployment_name="embeddingada002", model_name="text-embedding-ada-002", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # print("*"*5, "Index creation initiated", "*"*5)
    index = GPTVectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )
    # print("*"*5, "Index created", "*"*5)
    index.storage_context.persist("./entire_docs")
    return index


construct_index("./test.txt")
storage_context = StorageContext.from_defaults(persist_dir="./entire_docs")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
while True:
    text_input = input("YOU : ")
    response = query_engine.query(text_input)
    print("ChatBot : ", response)
    print('\n')
