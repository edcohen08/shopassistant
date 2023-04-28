import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="Shop Assistant", page_icon="🛍")

from models import chain, product_search

st.header("Your personal ai shopping assistant")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Shop till you drop", key="input")
    return input_text


user_input = get_text()

if user_input:
    products = product_search.similarity_search(user_input)
    output = chain.run(input_documents=products, question=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for ind, output in enumerate(st.session_state["generated"]):
        message(st.session_state["past"][ind], is_user=True, avatar_style="fun-emoji", key=str(ind) + "_user")
        message(output, avatar_style="fun-emoji", seed=42, key=str(ind))

