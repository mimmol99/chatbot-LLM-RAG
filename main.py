from loader import Loader
from database import ChromaDatabase
from retrieving import Retriever
from answer_generation import AnswerGenerator
from gui import GUI
import os

def main():

    username = "Domenico"
    database_name = "chrome_db_documents"
    database = ChromaDatabase(database_name)
    user_docs = database.get_user_docs(username)[0]
    print("Found in database")
    print(len(user_docs))
    print("\n\n\n")
    if len(user_docs)<1:
        print("Adding docs in database")
        documents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"PDF_FILES")
        user_docs = Loader(documents_path).load_documents()
        database.add_user_docs(username=username,docs=user_docs)
        #print(f"Added {len(user_docs)} documents"")
        retriever = Retriever(user_docs)
    else:
        print(f"Found {len(user_docs)} documents")
        print(user_docs)
        retriever = Retriever(user_docs,vectorstore=database.get_collection())
    
    #retriever = Retriever(user_docs,vectorstore=database.get_collection())
    answer_generator = AnswerGenerator(retriever)
    GUI(retriever,answer_generator)
    
if __name__ == "__main__":
    main()


