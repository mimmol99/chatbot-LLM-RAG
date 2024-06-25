import os
from langchain_community.document_loaders import PyPDFLoader


import getpass
from langchain_openai import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        

    def load_documents(self,accepted_files):
        # Use a list of patterns to match specific file types
        patterns = ["**/*."+f for f in accepted_files]
        docs = []
        for pattern in patterns:
            loader = DirectoryLoader(self.file_directory, glob=pattern, use_multithreading=True, show_progress=False)
            docs.extend(loader.load())
        return docs
    

    def is_string_an_url(self,url_string: str) -> bool:
            result = validators.url(url_string)
            if result is not True:
                return False
            return result


    def load_urls(self,urls):
        urls = [url for url in urls if self.is_string_an_url(url) is True]
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return docs
    

    def load_htmls(self,htmls):
        laoder = UnstructuredHTMLLoader(htmls,mode = 'single')
    





  
       

    
      
