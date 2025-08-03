import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

st.header('Reasearch Tool')

paper_input = st.selectbox("Select Research Paper Name",
                           ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers",
                            "GPT-3: Language Models are Few-Shot Learners",
                            "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",
                           ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length",
                            ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

base_dir = os.path.dirname(os.path.abspath(__file__))
json_template_path = os.path.join(base_dir, 'template.json')
template = load_prompt(json_template_path)

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
