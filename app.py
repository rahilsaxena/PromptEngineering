import streamlit as st
import openai
from  langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

#LangSmith Tracking 
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_Project']="ChatBot With Open AI"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(

    [
        ("system","Hey, Assistant.Please help to user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temprature,max_token):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt | llm |output_parser
    answer=chain.invoke({'question':question})
    return answer

    

#Title of APP
st.title("Enchanced Q&A Chatbot With OpenAI")
##side bar for settings
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

## DropDown to select various Open AI Model
llm=st.sidebar.selectbox("Select an Open AI Model",["gpt-4o","gpt-4-turbo","gpt-4"])
temprature = st.sidebar.slider("Temprature",min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Max_Token",min_value=50,max_value=300,value=150) 

##Main Interface for User Input
st.write("Ask Questions")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key, llm,temprature,max_token)
    st.write(response)
else:
    st.write("Please provide some input")
   