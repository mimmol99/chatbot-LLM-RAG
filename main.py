from loading_ds import Loader
from retrieving import Retriever
from answer_generation import AnswerGenerator
from gui import GUI

def main():
    documents_path = "./PDF_FILES"
    model = None
    docs = Loader(documents_path, model).load_documents()
    retriever = Retriever(docs).get_retriever()
    answer_generator = AnswerGenerator(retriever)
    GUI(retriever,answer_generator)
    
if __name__ == "__main__":
    main()


