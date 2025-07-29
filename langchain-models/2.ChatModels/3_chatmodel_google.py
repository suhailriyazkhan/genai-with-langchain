from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Check all available models
for model in genai.list_models():
    print(model.name)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

result = model.invoke('What is the capital of India')

print(result.content)