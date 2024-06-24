import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
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
        

    def load_documents(self):
        # Use a list of patterns to match specific file types
        patterns = ["**/*.html","**/*.pdf", "**/*.txt", "**/*.doc", "**/*.docx", "**/*.word", "**/*.pptx"]
        docs = []
        for pattern in patterns:
            loader = DirectoryLoader(self.file_directory, glob=pattern, use_multithreading=True, show_progress=True)
            docs.extend(loader.load())
        return docs
    
    def load_documents_parallel(self):
        # Use a list of patterns to match specific file types
        patterns = ["**/*.html", "**/*.pdf", "**/*.txt", "**/*.doc", "**/*.docx", "**/*.word", "**/*.pptx"]
        docs = []
        
        # Function to load documents for a given pattern
        def load_pattern(pattern):
            loader = DirectoryLoader(self.file_directory, glob=pattern, use_multithreading=True, show_progress=True)
            return loader.load()

        # Use ThreadPoolExecutor to load documents in parallel
        with ThreadPoolExecutor() as executor:
            future_to_pattern = {executor.submit(load_pattern, pattern): pattern for pattern in patterns}
            for future in as_completed(future_to_pattern):
                pattern = future_to_pattern[future]
                try:
                    docs.extend(future.result())
                except Exception as e:
                    print(f"Error loading pattern {pattern}: {e}")

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



  
       

    
      
