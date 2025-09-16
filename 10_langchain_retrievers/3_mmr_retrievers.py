
"""Example of using Maximal Marginal Relevance (MMR) with LangChain and Gemini embeddings.
Maximal Marginal Relevance (MMR) is an information retrieval technique
that selects results to maximize relevance to a query while minimizing redundancy among the results.
It helps return a diverse set of documents by balancing similarity to the query
and dissimilarity to already selected documents. This is useful in applications like
document retrieval, summarization, and search, where you want both relevance and diversity in the output.
"""


from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load .env file (must contain GOOGLE_API_KEY)
load_dotenv()

# Initialize Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Step 3: Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="my_collection"
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)