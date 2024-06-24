from loading_ds import Loader
from Embedding import EmbeddingModel
from answer_generation import AnswerGenerator
from gui import GUI
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
load_dotenv(Path("./api_key.env"))


def main():
    
    files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"FILES")
    urls = ["https://ainews.it/synthesia-creazione-di-avatar-ai-anche-da-mobile/"]
    loader = Loader(files_path)
    docs_urls = loader.load_urls(urls)
    
    docs = loader.load_documents_parallel()
    docs.extend(docs_urls)
    #summarized_docs = loader.summarize_docs(docs)
    parent_retriever = EmbeddingModel(docs).get_parent_retriever()
    answer_generator = AnswerGenerator(parent_retriever)
    GUI(answer_generator)
    
if __name__ == "__main__":
    main()


