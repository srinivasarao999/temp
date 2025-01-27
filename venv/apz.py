import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ollama_project"

# Initialize the Ollama LLM
llm = Ollama(model="gemma2:2b")  # Use the correct model name
output_parser = StrOutputParser()

# Function to create the prompt template
def create_prompt(question: str) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", f"Question: {question}")
        ]
    )
    return prompt

# Streamlit framework
st.title("Chat with LLAMA2 🤖")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user
if prompt := st.chat_input("What is the question on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from the model
    with st.chat_message("assistant"):
        try:
            # Create the prompt chain
            prompt_template = create_prompt(prompt)
            chain = prompt_template | llm | output_parser
            response = chain.invoke({"question": prompt})

            # Display the response
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            response = "Sorry, I couldn't generate a response. Please try again."

    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})