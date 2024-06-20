import chromadb
from chromadb.config import Settings
import os

class ChromaDatabase:
    def __init__(self, collection_name):
        self.client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


    def get_collection(self):
        return self.collection
    

    def get_user_docs(self, username):
        # Providing a dummy query text to satisfy the requirement of query parameters
        query_text = "everything"
        results_query = self.collection.query(
            query_texts=[query_text],
            where={"username": username}
        )
        
        return results_query['documents']


    def add_user_docs(self, username, docs):
        page_contents = [doc.page_content for doc in docs]
        titles = [doc.metadata['source'].split(os.path.sep)[-1] for doc in docs]
        metadatas = [{"username": username, "title": titles[i]} for i in range(len(docs))]
        ids = [str(i) for i in range(len(docs))]

        
        self.collection.add(
            documents=page_contents,
            metadatas=metadatas,
            ids=ids
        )
