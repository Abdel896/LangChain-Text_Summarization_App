import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Function to generate response using OpenAI API
def generate_response(txt, openai_api_key):
    if openai_api_key:
        # Instantiate the LLM model with the provided API key
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
        # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        # Text summarization using OpenAI API
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        return chain.run(docs)
    else:
        return "Please provide a valid OpenAI API key."

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    # Input field for the OpenAI API key
    openai_api_key = st.text_input('OpenAI API Key', type='password', value='', max_chars=100)
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key:
        if openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                # Generate response using the provided text and API key
                response = generate_response(txt_input, openai_api_key)
                result.append(response)
        else:
            st.error("Invalid OpenAI API key.")
            
# Display the result
if result:
    st.info(result[0])  # Display the first (and only) element of the result list

