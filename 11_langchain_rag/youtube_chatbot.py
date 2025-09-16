from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from utils.gemini_client import GoogleGenAIChatClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
# Load .env file (must contain GOOGLE_API_KEY)
load_dotenv()
# Initialize Gemini embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
llm_model = GoogleGenAIChatClient().model

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
transcript = ""

try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en"]).snippets
    # print(transcript_list.snippets)
    for fetched_trans_snippet in transcript_list:
        transcript += " " + fetched_trans_snippet.text
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))

vector_store = Chroma.from_documents(chunks, embedding_model)

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke('What is deepmind')

# Step 3 - Augmentation

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_doc = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_doc)
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})

## Step 4 - Generation
answer = llm_model.invoke(final_prompt)
print(answer.content)
