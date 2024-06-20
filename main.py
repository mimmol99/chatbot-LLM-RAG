from loading_ds import Loader
from retrieving import Retriever
from answer_generation import AnswerGenerator
from gui import GUI

def main():
    documents_path = "./PDF_FILES"
    docs = Loader(documents_path).load_documents()
    retriever = Retriever(docs).get_retriever()
    answer_generator = AnswerGenertor(retriever)
    GUI(retriever,answer_generator)
    
if __name__ == "__main__":
    main()


