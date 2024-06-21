from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class EmbeddingModel:
	def __init__(self, docs):
		self.docs = docs
		self.vectorstore = self.create_vector_store()
		self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, add_start_index=True)
		self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
	
	def create_vector_store(self):

		#sentence_transformer_ef = SentenceTransformerEmbeddings(model_name = "all_MiniLM-L6-v2")	
		model_name = "BAAI/bge-small-en"
		model_kwargs = {'device': 'cpu'}
		encode_kwargs = {'normalize_embeddings': True}
		self.sentence_transformer_ef  = HuggingFaceBgeEmbeddings(
			model_name=model_name,
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
		)

		self.vectorstore = Chroma(collection_name = "full_documents", embedding_function=self.sentence_transformer_ef)
		return self.vectorstore

		
	def get_parent_retriever(self):
		store = InMemoryStore()
		retriever = ParentDocumentRetriever(
			vectorstore=self.vectorstore,
			docstore=store,
			child_splitter=self.child_splitter,
			parent_splitter=self.parent_splitter,
		)

		retriever.add_documents(self.docs)

		return retriever
	

	def get_vectore_store(self):
		return self.vectorstore
	
	
	def get_sentence_transformer_ef(self):
		return self.sentence_transformer_ef


		
