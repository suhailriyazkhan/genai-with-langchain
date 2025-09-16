from uuid import uuid4
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load .env file (must contain GOOGLE_API_KEY)
load_dotenv()

# Initialize Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

# Create sample documents
docs = [
    Document(page_content="Virat Kohli ... Royal Challengers Bangalore", metadata={"team": "Royal Challengers Bangalore"}),
    Document(page_content="Rohit Sharma ... Mumbai Indians", metadata={"team": "Mumbai Indians"}),
    Document(page_content="MS Dhoni ... Chennai Super Kings", metadata={"team": "Chennai Super Kings"}),
    Document(page_content="Jasprit Bumrah is ... Mumbai Indians", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Ravindra Jadeja ... Chennai Super Kings", metadata={"team": "Chennai Super Kings"})
]

# Create Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",
)

# Add documents
print("Adding documents to the vector store")
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)
print("Documents added successfully")

# Similarity search
print("Performing similarity search")
result = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
print("Similarity search completed")

# Show results
for doc in result:
    print(f"\nContent: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
