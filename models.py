import os

import streamlit as st

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
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

def load_chain():
    template = """You are a shopping assistant that helps people find the clothes they’re looking for.
    Include urls in your responses. Only include items from the provided summaries.
    {chat_history}
    Shopper: {question}
    {summaries}
    Shopping assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "summaries"], 
        template=template
    )
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", input_key="question")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return load_qa_with_sources_chain(llm, verbose=True, chain_type="stuff", memory=memory, prompt=prompt)

product_search = load_pinecone()
chain = load_chain()