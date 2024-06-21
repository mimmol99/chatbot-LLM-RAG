import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import getpass
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

class Loader:
    def __init__(self, pdf_directory, model):
        self.pdf_directory = pdf_directory
        self.all_docs = []
        self.model = None
        self.model_api_key = None

    def load_documents(self):
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdfloader = PyPDFLoader(os.path.join(self.pdf_directory, filename))
                docs = pdfloader.load()
                s_docs = self.summarization(docs)
                self.all_docs.extend(docs)
        return self.all_docs

    def summarization(self,docs):
        if not self.model_api_key:
            # Prompt for the API key if it is empty
            self.model_api_key = getpass.getpass(prompt="Enter your model API key: ")
        if not self.model:
            self.model = ChatOpenAI(model="gpt-4o", temperature=0,openai_api_key = self.model_api_key)
         # Define prompt
        prompt_template = """Write a detailed summary,considering every section, of the following document:
        "{text}"
        SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        llm_chain = LLMChain(llm=self.model, prompt=prompt)


        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
       

        s_dc = stuff_chain.invoke(docs)["output_text"]


        return s_dc




