import validators , streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

#LangSmith Tracking 

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_Project']="Text Summerization Using HuggingFace-Langchain Models"

### StreamLit App
st.set_page_config(page_title="Langchain Integration with Hugging Face:Summarize Text from YT or website")
st.title("LangcChain: Summarize Text from YT or Website")
st.subheader('Summarize URL')

## Get the Groq API and url (YT or website) to be summarized
with st.sidebar:
    hf_api_key=st.text_input("Hugging Face API Token", value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## intialize with Gemma Model and Groq API
#llm= ChatGroq(model="Gemma-7b-It",groq_api_key=groq_api_key)
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temprature=0.7, token=hf_api_key)

prompt_template= """
provide summary of following content in 300 words:Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the content from YT or WebSite"):
    ##Validation of input
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can be YT video or Website Url")
    
    else:
        try:
            with st.spinner("waiting .."):

                ## Loading youtube data
                if "youtube.com" in generic_url:
                   loader= YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False)

            doc=loader.load()
            ##Chain for Summarization
            chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
            output_summary=chain.run(doc)
            st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")