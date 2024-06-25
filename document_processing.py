    
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

class DocumentProcessor():

    def __init__(self,docs,model):
        self.docs = docs
        self.model = model


    def summarize_docs(self,docs):
        summarized_docs = []
        for doc in docs:
            summarized_docs.append(self.summarize_doc(doc))
        return summarized_docs


    def summarize_doc(self,doc):

        # Define prompt
        prompt_template = """Write a detailed summary,considering every section, of the following document:
        "{text}"
        SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        llm_chain = LLMChain(llm=self.model, prompt=prompt)

        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        s_dc = stuff_chain.invoke([doc])["output_text"]

        doc.page_content = s_dc

        return doc
