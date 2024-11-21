import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Sidebar 
with st.sidebar:
    st.markdown("## About Fela")
    st.markdown("This chatbot uses the Llm model to answer questions based on the content of an uploaded article. You can upload a text or markdown file and ask questions about its content.")
    st.markdown("[View the source code](https://github.com/Al0ni/LLM-Files-QA/blob/main/app.py)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Al04ni/LLM-Files-QA?quickstart=1)")

# Main interface
st.title(" File Q&A 📝 with Fela")

# Initialize variables for model and tokenizer
model_name = "mistralai/Mistral-7B"
tokenizer, model = None, None

try:
    # Attempt to load the tokenizer and model
    with st.spinner("Loading model and tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error("Failed to load the model or tokenizer. Please check your configuration.")
    st.warning(f"Error Details: {e}")
    st.stop()  # Stop execution if model loading fails

# Add tabs for suggested questions, easing kabisa
tabs = st.tabs(["Summary", "Key Points", "Actionable Insights", "Critical Analysis", "General"])

#questions suggested for each tab
suggested_questions = {
    "Summary": ["Can you summarize the article?", "What are the main points discussed?"],
    "Key Points": ["What are the key takeaways?", "What facts are highlighted in the article?"],
    "Actionable Insights": ["What actions can be taken based on this article?", "How can the information be applied?"],
    "Critical Analysis": ["Are there any biases present?"],
    "General": ["Can you rewrite the article?", "Who is the intended audience for this article?"]
}

# Display  questions in tabs
selected_question = ""
for tab_name, questions in suggested_questions.items():
    with tabs[list(suggested_questions.keys()).index(tab_name)]:
        st.markdown(f"### Suggested Questions: {tab_name}")
        for question in questions:
            if st.button(question, key=f"{tab_name}_{question}"):
                selected_question = question

# File uploader and question input
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    value=selected_question,  # Populate the selected question
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()
    prompt = f"Here's an article:\n\n<article>\n{article}\n</article>\n\n{question}"

    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate the response
        with torch.no_grad():
            with st.spinner("Generating response..."):
                outputs = model.generate(inputs.input_ids, max_length=150)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write("### Answer")
        st.write(response)

    except Exception as e:
        st.error("An error occurred while processing your query.")
        st.warning(f"Error Details: {e}")
else:
    if not uploaded_file:
        st.info("Please upload a file to get started.")
    elif not question:
        st.info("Please enter a question to ask about the uploaded file.")
