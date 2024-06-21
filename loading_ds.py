import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import getpass
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
#from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers.json import SimpleJsonOutputParser

class Loader:


    def __init__(self, file_directory):
    
        self.file_directory = file_directory
        self.all_docs = []
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OPENAI_API_KEY")
        self.model_api_key = os.environ["OPENAI_API_KEY"]
        self.model = ChatOpenAI(model="gpt-4o", temperature=0,openai_api_key = self.model_api_key)
        

    def load_documents(self):
    
        for filename in os.listdir(self.file_directory):
            if filename.endswith(".pdf"):
                pdfloader = PyPDFLoader(os.path.join(self.file_directory, filename))
                docs = pdfloader.load()
                s_docs = self.summarization(docs)
                self.all_docs.extend(docs)
        return self.all_docs


    def summarization(self,docs):

        # Define prompt
        prompt_template = """Write a detailed summary,considering every section, of the following document:
        "{text}"
        SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        llm_chain = LLMChain(llm=self.model, prompt=prompt)

        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        s_dc = stuff_chain.invoke(docs)["output_text"]

        return s_dc


  
       

    
      
