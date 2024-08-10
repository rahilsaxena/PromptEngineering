from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

#LangSmith Tracking 
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_Project']="ChatBot With Ollama"


##  Prompt Template
prompt=ChatPromptTemplate.from_messages(

    [
        ("system","Hey, Assistant.Please help to user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm,temprature,max_token):
    llm=Ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt | llm |output_parser
    answer=chain.invoke({'question':question})
    return answer


## DropDown to select various Open AI Model
llm=st.sidebar.selectbox("Select Ollama Model",["mistral","gemma2","phi3","llama3.1"])
temprature = st.sidebar.slider("Temprature",min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Max_Token",min_value=50,max_value=300,value=150) 

##Main Interface for User Input
st.write("Question Please")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input, llm,temprature,max_token)
    st.write(response)
else:
    st.write("Please provide some input")
   
