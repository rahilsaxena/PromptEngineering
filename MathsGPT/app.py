import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains.llm import LLMChain 


from langchain.chains.llm_math.base import LLMMathChain

from langchain.agents import initialize_agent,Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os

#LangSmith Tracking 
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_Project']="Math4GTP Using Groq"

##Set Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant",page_icon="iii")
st.title("Text to math Problem solver using google gemma 2")
groq_api_key=st.sidebar.text_input(label="Enter your Groq Open API Key:",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

prompt=""""
Agent for solving user math question and display the answer 
Question:{question}
Answer :
"""
prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

## Initializing the tools
wikipedia_wrapper= WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikepedia",
    func= wikipedia_wrapper.run,
    description="Tool to Solve Math Problem"
)
## reasoning_tool
reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning"
)
## intialize the math tool
math_chain=LLMMathChain(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math question.Only input math expression"
)


## initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False, 
    handle_parsing_errors=True
)


if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi, I am math chatbot"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## functions to generate the response
def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

#Interaction
question=st.text_area("Enter your questions","Demo Question")
if st.button("Get Ans"):
    if question:
        with st.spinner("Generate Response.."):
            st.session_state.messages.append({"role","user","content",question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb],chat_history=False)
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write(response)
            st.success(response)

else:
    st.warning("Please provide input")






