import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

class AnswerGenerator():

    """ def __init__(self,retriever,model_name = "llama3-70b-8192",temperature = 0):"""
    def __init__(self,retriever,model):

        self.retriever = retriever
        self.model = model
        self.store = {}
        self.rag_chain = self.create_rag_chain()



    def get_store(self):
        return self.store
    

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
    

    def answer_prompt(self,user_prompt,session_id = "123"):
        dict_answer = self.rag_chain.invoke({"input":user_prompt},config={"configurable": {"session_id": session_id}})
        return dict_answer['answer']
    

    def get_rag_chain(self):
        return self.rag_chain
    
    
    def get_retriever(self):
        return self.retriever



        


    
