# chatbot-LLM-RAG
chatbot-LLM-RAG

An example of LLM-RAG architecture:
![alt text](https://github.com/DLfrontiere/chatbot-LLM-RAG/blob/main/images/rag-chatbot-architecture-1.png?raw=true)

In particular in this repository has been used:
- PyPDFLoader to load PDF files and transform in Document objects
- RecursiveCharacterTextSplitter for documents splitting
- HuggingFaceBgeEMbeddings for documents embedding
- Chroma for vector storing
- ParentDocumentRetriever to retrieve child and parent chunks using similarity
- chatgpt3.5 turbo as LLM
- RunnableWithMessageHistory to make the LLM chat history aware
