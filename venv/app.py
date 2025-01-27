import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ollama_project"

# Streamlit app title
st.title("LangChain with Gemma")

# Add model selection dropdown
model_options = ["gemma:7b", "llama2", "mistral"]  # Add more models as needed
selected_model = st.selectbox("Select a model:", model_options)

# User input
input_text = st.text_input("What is the question in your mind?")

try:
    # Initialize OllamaLLM
    llm = OllamaLLM(model=selected_model)
    output_parser = StrOutputParser()

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ])

    # Create the chain
    chain = prompt | llm | output_parser

    # Process the input
    if input_text:
        try:
            # Invoke the chain with the input
            response = chain.invoke({"question": input_text})
            
            # Display the result in Streamlit
            st.write(response)
        except Exception as e:
            # Handle errors gracefully
            st.error(f"An error occurred while processing: {e}")

except Exception as e:
    st.error(f"""Model initialization failed: {e}""")