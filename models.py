import os

import streamlit as st

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
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

@st.cache_resource
def load_chain():
    template = """You are a shopping assistant helping a person find clothing.
    Given the following question and content summaries, if applicable, recommend up 
    to two clothing items including your sources.

    {chat_history}
    Shopper: {question}
    {summaries}
    Shopping assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "summaries"], 
        template=template
    )
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", input_key="question")
    llm = OpenAI(temperature=0)
    return load_qa_with_sources_chain(llm, verbose=True, chain_type="stuff", memory=memory, prompt=prompt)

product_search = load_pinecone()
chain = load_chain()