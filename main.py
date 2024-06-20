from loading_ds import Loader
from Embedding import EmbeddingModel
from answer_generation import AnswerGenerator
from gui import GUI
import os

def main():
    documents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"PDF_FILES")
    docs = Loader(documents_path).load_documents()
    retriever = EmbeddingModel(docs).get_parent_retriever()
    answer_generator = AnswerGenerator(retriever)
    GUI(retriever,answer_generator)
    
if __name__ == "__main__":
    main()


