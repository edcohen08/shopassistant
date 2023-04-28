import os

import streamlit as st

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone


@st.cache_resource
def load_pinecone():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"]
    )
    return Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=OpenAIEmbeddings())

def load_chain():
    template = """You are a shopping assistant helping a person find clothing.
    Given the following extracted products from the catalog and a question, create a final answer
    including your sources.
    =============
    {chat_history}
    Human: {question}
    =============
    {summaries}
    Shopping assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "summaries"], 
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    llm = OpenAI(temperature=0)
    return load_qa_with_sources_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

product_search = load_pinecone()
chain = load_chain()