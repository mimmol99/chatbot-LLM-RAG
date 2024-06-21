# chatbot-LLM-RAG
chatbot-LLM-RAG

An example of LLM-RAG architecture:
![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/rag-chatbot-architecture-1.png?raw=true)

In particular in this repository has been used:

- PyPDFLoader to load PDF files and transform in Document objects
   enable files upload for each user? store files in local or cloud? saving the loaded files? adding/removing files?
- RecursiveCharacterTextSplitter for documents splitting
- HuggingFaceBgeEMbeddings for documents embedding
- Chroma for vector storing
  local or cloud database? store every chat history?
- ParentDocumentRetriever to retrieve child and parent chunks using similarity
- chatgpt3.5 turbo as LLM
  free/opensource models?fine tuning?efficiency?speed?cost?
- RunnableWithMessageHistory to make the LLM chat history aware

