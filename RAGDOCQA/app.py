import streamlit as st
import os
from  langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
##Load the Groq API_Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key= os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_Project']="ChatBot With Groq"

#Use any model EX- Gemma
llm = ChatGroq(groq_api_key=groq_api_key,model="Gemma-7b-It")

prompt=ChatPromptTemplate.from_template(
      """
      Answer the questions based on the provided context only.
      Please provide the most accurate response based on question 
      <context>
      {context}
      Question:{input}


      """
)

def create_vector_embedding():
    if "vectors"  not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research paper") 
        ##Data Ingestion step
        st.session_state.docs=st.session_state.loader.load()
        ## Document Loading
        st.session_state.text_splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.doc[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter query from Research Paper")

if st.button("Document Embedding"):
     create_vector_embedding()
     st.write("Vector DataBase is Ready")

import time 
 
if user_prompt:
   document_chain= create_stuff_documents_chain(llm,prompt)
   retriever=st.session_state.vectors.as_retriever()
   retrieval_chain= create_retrieval_chain(retriever,document_chain)

   ##Get time required to process this Request
   start=time.process_time()
   response = retrieval_chain.invoke({'input':user_prompt})
   print(f"Response Time :{time.process_time()-start}")

   st.write(response['answer'])

   ## With a streamlit expander
   with st.expander("Document similarity Search"):
       for i, doc in enumerate(response['context']):
           st.write(doc.page_content)
           st.write('---------------')
