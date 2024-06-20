import os
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings


class Loader:


    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.all_docs = []


    def load_documents(self):
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdfloader = PyPDFLoader(os.path.join(self.pdf_directory, filename))
                docs = pdfloader.load()
                self.all_docs.extend(docs)
        return self.all_docs


class Database:


    def __init__(self,collection_name):
        #self.chroma_client = chromadb.Client()
        self.client = chromadb.PersistentClient(path= os.path.join(os.path.dirname(os.path.abspath(__file__)),"chroma_db") ) 
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


    def get_user_docs(self,username):
        results_query = self.collection.query(
            where={"username": username}
        )
        return results_query


    def add_user_docs(self,username,docs):
        #print(docs,docs[0])
        page_contents = [doc.page_content for doc in docs]
        titles = [doc.metadata['source'].split(os.path.sep)[-1] for doc in docs]
        metadatas = [{"username": username, "title": titles[i]} for i in range(len(docs))]
        ids = [str(i) for i in range(len(docs))]

        print(metadatas)
        self.collection.add(
            documents=page_contents,
            metadatas=metadatas,
            ids = ids
        )
        



