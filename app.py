import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load the Llama model and tokenizer
model_name = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Sidebar for description
with st.sidebar:
    st.markdown("## About the Chatbot")
    st.markdown("This chatbot uses the Llama model to answer questions based on the content of an uploaded article. You can upload a text or markdown file and ask questions about its content.")
    st.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

# Main interface
st.title("📝 File Q&A with Llama")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()
    prompt = f"Here's an article:\n\n<article>\n{article}\n</article>\n\n{question}"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=150)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write("### Answer")
    st.write(response)
