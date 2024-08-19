import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


## Arvix & wikipedia tools
arvix_wrapper=ArxivAPIWrapper(top_k_resuilts=1,doc_content_chars_max=200)
arvix=ArxivQueryRun(api_wrapper=arvix_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_resuilts=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

## Sidebar for settings

st.sidebar.title("Langchain - chatwith Search")

""""
We are using streamlet call back handler to display the though 
and action of an agent in an interactive seesion

"""
api_key=st.sidebar.text_input("Enter your Groq Open API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi, I am a chatbot who can search the web.How can i help yoy"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arvix,wiki]

    search_agent=initialize_agent(tools,llm,agents=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errros=True)
    
    with st.chat_message("Assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'Assistant',"content":response})
        st.write(response)



