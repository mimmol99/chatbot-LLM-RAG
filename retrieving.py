from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore

class Retriever:
	def __init__(self, docs):
		self.docs = docs
	
	def embedding(self):
		sentence_transformer_ef = SentenceTransformerEmbeddings(model_name = "all_MiniLM-L6-v2")	
		vectorstore = Chroma(collection_name = "full_documents", embedding_function=sentence_transformer_ef)
		return vectorstore
	
	def splitter(self):
		child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, add_start_index=True)
		parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)

		return child_splitter,parent_splitter
		
	def get_retriever(self):
		vectorstore = self.embedding()
		child_splitter,parent_splitter = self.splitter()
		store = InMemoryStore()
		retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

		retriever.add_documents(self.docs)

		return retriever


		
