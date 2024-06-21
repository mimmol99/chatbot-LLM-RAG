# Import necessary libraries
import getpass 
import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

from loading_ds import Loader
from Embedding import EmbeddingModel
from langchain_community.document_loaders import PyPDFLoader
from answer_generation import AnswerGenerator


# Set the title for the Streamlit app
st.title("RAG enhanced Chatbot")

# Set up the OpenAI API key from Streamlit input
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Cached function to create a vectordb for the provided PDF files

@st.cache_data
def pdf_to_docs(files):

    with st.spinner(f"Processing {files} pdf.."):
        
        all_docs = []
        for f in files:

            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(f.getvalue())


            pdfloader = PyPDFLoader(temp_file)
            docs = pdfloader.load()

            all_docs.extend(docs)
        return all_docs

@st.cache_data
def answer_gen():
    with st.spinner(f"Updating chatbot.."):
         st.session_state['answer_generator'] = AnswerGenerator(EmbeddingModel(docs).get_parent_retriever())
   
# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state

if pdf_files:

    ###
    docs = pdf_to_docs(pdf_files)
    st.session_state["vectordb"] = EmbeddingModel(docs).create_vector_store()
    answer_gen()
    ###



# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages


# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:

    answer_generator = st.session_state.get("answer_generator",None)

    if not answer_generator:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)


    # Call ChatGPT with streaming and display the response as it comes
    with st.spinner(f"Retrieving answer.."):
        result = answer_generator.answer_prompt(question)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

    for message in prompt:
        
        if message["role"] not in ["system","user"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

