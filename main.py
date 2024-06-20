from loading_ds import Loader
from retrieving import Retriever
from answer_generation import AnswerGenerator
from gui import GUI

def main():
    documents_path = "./PDF_FILES"
    docs = Loader(documents_path).load_documents()
    retriever = Retriever(docs).get_retriever()
    GUI(retriever)
    
if __name__ == "__main__":
    main()


