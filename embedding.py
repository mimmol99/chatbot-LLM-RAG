from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

class EmbeddingModel:
    
	def __init__(self, docs,embedding_model_name = "BAAI/bge-small-en"):
		self.docs = docs

	def hugging_face_bge_embeddings(self,model_name = "BAAI/bge-small-en"):
		self.model_name = model_name
		model_kwargs = {'device': 'cpu'}
		encode_kwargs = {'normalize_embeddings': True}
		sentence_transformer_ef  = HuggingFaceBgeEmbeddings(
			model_name=self.model_name,
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
		)
		return sentence_transformer_ef
	
	def fast_embed_embeddings(self,model_name = "BAAI/bge-small-en-v1.5"):
		embeddings = FastEmbedEmbeddings()
		return embeddings
	
	def open_ai_embeddings(self,model_name = "text-embedding-ada-002"):
		openai = OpenAIEmbeddings(model = model_name)
		return openai

	def create_vector_store(self,docs,sentence_transformer_ef):
		self.vectorstore = Chroma.from_documents(documents=docs,embedding =sentence_transformer_ef)

	def get_vectore_store(self):
		return self.vectorstore
	
	def get_sentence_transformer_ef(self):
		return self.sentence_transformer_ef



