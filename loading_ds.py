import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

import validators 





class Loader:


    def __init__(self, file_directory):
    
        self.file_directory = file_directory
        self.all_docs = []
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OPENAI_API_KEY")
        self.model_api_key = os.environ["OPENAI_API_KEY"]
        self.model = ChatOpenAI(model="gpt-4o", temperature=0,openai_api_key = self.model_api_key)
        

    def load_documents(self):
        loader = DirectoryLoader(self.file_directory, glob="**/*.*", use_multithreading = True,show_progress=True)
        docs = loader.load()
        return docs
    

    def is_string_an_url(url_string: str) -> bool:
            result = validators.url(url_string)
            if result is not True:
                return False
            return result


    def load_urls(self,urls):
        urls = [url for url in urls if self.is_string_an_url(url) is True]
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return docs
    

    def summarize_docs(self,docs):
        summarized_docs = []
        for doc in docs:
            summarized_docs.append(self.summarize_doc(doc))
            #summarized_docs.append(doc)
        return summarized_docs


    def summarize_doc(self,doc):

        # Define prompt
        prompt_template = """Write a detailed summary,considering every section, of the following document:
        "{text}"
        SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        llm_chain = LLMChain(llm=self.model, prompt=prompt)

        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        s_dc = stuff_chain.invoke([doc])["output_text"]

        doc.page_content = s_dc

        return doc



  
       

    
      
