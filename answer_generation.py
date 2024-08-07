import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import pandas as pd

class AnswerGenerator():

    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model
        self.store = {}
        self.context_store = {}  # Store the context used for each session
        self.rag_chain = self.create_rag_chain()
        self.csv_file = './chat_history.csv'

    def get_store(self):
        return self.store

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_rag_chain(self):
        system_prompt = (
            "Sei un assistente chatbot per una compagnia di assicurazioni (Yolo-insurance). "
            "Usa i seguenti pezzi di contesto recuperato per rispondere "
            "alla domanda. Quando citi la nostra compagnia di assicurazione parla in "
            "prima personale plurale (e.g contatta la nostra assistenza clienti)."
            "Non citare mai che hai usato il contesto"
            " o documenti nella risposta "
            " e non citare mai riferimenti ad altre compagnie assicurative."
            "Se non conosci la risposta o non trovi "
            "la risposta nel contesto,scusati e rispondi semplicemnte di contattare l'assistenza"
            "\n"
            "Contesto:"
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
        "Data una cronologia della chat e l'ultima domanda dell'utente,"
        "che potrebbe fare riferimento al contesto nella cronologia della chat, "
        "formulare una domanda autonoma che possa essere compresa "
        "senza la cronologia della chat. NON rispondere alla domanda, "
        "riformularla se necessario e altrimenti restituirla così com'è."
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

    def answer_prompt(self, user_prompt, session_id="123"):
        lower_user_prompt = user_prompt.lower()
        context = self.get_current_context(lower_user_prompt)
        self.context_store[session_id] = context  # Store the context used
        dict_answer = self.rag_chain.invoke({"input": lower_user_prompt}, config={"configurable": {"session_id": session_id}})
        answer = dict_answer['answer']
        return answer
    

    def get_current_context(self, prompt):
        context = self.retriever.invoke(prompt)
        return context

    def get_rag_chain(self):
        return self.rag_chain

    def get_retriever(self):
        return self.retriever

    def get_context(self, session_id="123"):
        if session_id in self.context_store:
            return self.context_store[session_id]
        else:
            return "No context found for this session."



        


    
