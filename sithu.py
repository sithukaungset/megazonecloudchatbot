from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain.base_language import BaseLanguageModel

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()

# Set Azure environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://mtcaichat01.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "824fe43e851f4862af326fa83c3d3cfe"


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=AzureOpenAI(
        deployment_name="embeddingada002",
        model_name="text-embedding-ada-002",
        temperature=0.7,
        max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def main():
    st.set_page_config(page_title="Megazone Cloud ChatBot")
    st.markdown("<h1 style='text-align: center; color: lightgreen;'>Megazone Cloud ChatBot 💬</h1>",
                unsafe_allow_html=True)

    index = construct_index("docs")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        with st.spinner('Reading the PDF...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            with st.spinner('Creating knowledge base...'):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_input("Ask a question 🤖:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                with st.spinner('Generating answer...'):
                    response = index.query(
                        user_question, response_mode="compact")

                # Display the result in a more noticeable way
                st.markdown(
                    f'### Answer: \n {response.response}', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

# from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain.chat_models import ChatOpenAI
# import gradio as gr
# import sys
# import os
# from dotenv import load_dotenv

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import AzureOpenAI
# from langchain.callbacks import get_openai_callback

# os.environ["OPENAI_API_KEY"] = 'Your API Key'


# def construct_index(directory_path):
#     max_input_size = 4096
#     num_outputs = 512
#     max_chunk_overlap = 20
#     chunk_size_limit = 600

#     prompt_helper = PromptHelper(
#         max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm_predictor = LLMPredictor(llm=ChatOpenAI(
#         temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

#     documents = SimpleDirectoryReader(directory_path).load_data()

#     index = GPTSimpleVectorIndex(
#         documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#     index.save_to_disk('index.json')

#     return index


# def chatbot(input_text):
#     index = GPTSimpleVectorIndex.load_from_disk('index.json')
#     response = index.query(input_text, response_mode="compact")
#     return response.response


# iface = gr.Interface(fn=chatbot,
#                      inputs=gr.components.Textbox(
#                          lines=7, label="Enter your text"),
#                      outputs="text",
#                      title="Custom-trained AI Chatbot")

# index = construct_index("docs")
# iface.launch(share=True)
