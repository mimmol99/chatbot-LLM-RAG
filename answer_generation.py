import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

class AnswerGenerator():


    def __init__(self,retriever,model_name = "gpt-3.5-turbo",temperature = 0):

        self.retriever = retriever
        self.model_name = model_name
        self.model_api_key = None

        #if not os.getenv("OPENAI_API_KEY"):
            # Prompt for the API key if it is empty
            # os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")

        self.model_temperature = temperature
        self.initialize_model()
        self.store = {}
        self.rag_chain = self.create_rag_chain()

    
    def initialize_model(self):
        self.check_api_key()
        self.model = ChatOpenAI(model=self.model_name, temperature=self.model_temperature,openai_api_key = self.model_api_key)


    def check_api_key(self):
        if not self.model_api_key:
            # Prompt for the API key if it is empty
            self.model_api_key = getpass.getpass(prompt="Enter your model API key: ")


    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    

    def create_rag_chain(self):

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "Context:"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        print(type(self.model),type(qa_prompt))
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(self.model, self.retriever, contextualize_q_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain
    

    def answer_prompt(self,user_prompt,session_id):
        dict_answer = self.rag_chain.invoke({"input":user_prompt},config={"configurable": {"session_id": session_id}})
        return dict_answer['answer']
    

    def get_rag_chain(self):
        return self.rag_chain



    def rrf_retriever(query: str) -> list[Document]:

    # RRF chain
        chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
        result = chain.invoke({"query": query})
        return result
    
    def reciprocal_rank_fusion(results: list[list], k=60):
    
        fused_scores = {}
        for docs in results:
            # Assumes the docs are returned in sorted order of relevance
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # for TEST (print reranked documentsand scores)
        print("Reranked documents: ", len(reranked_results))
        for doc in reranked_results:
            print('---')
            print('Docs: ', ' '.join(doc[0].page_content[:100].split()))
            print('RRF score: ', doc[1])

        # return only documents
        return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]
    
    

    def query_generator(original_query: dict) -> list[str]:
   
    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information in Japanese, without significantly altering the original query's meaning"),
        ("user", "OUTPUT (3 queries):")
    ])

    # LLM model
    model = ChatOpenAI(
                temperature=0,
                model_name=LLM_MODEL_OPENAI
            )

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    # for TEST
    print('Generated queries:\n', '\n'.join(queries))

    return queries



        


    