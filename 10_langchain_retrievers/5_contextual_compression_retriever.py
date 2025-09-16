"""
The ContextualCompressionRetriever in LangChain is a retriever that combines a
base retriever (such as a vector store retriever) with a document compressor (often an LLM-based chain).
It first retrieves a set of documents relevant to a query, then uses the compressor to condense or
filter these documents based on the query context, returning only the most relevant and concise information.
This approach helps reduce noise and improve the quality of retrieved results,
especially when working with large document sets or when concise answers are needed.
"""

from langchain_chroma import Chroma
from utils.gemini_client import GoogleGenAIChatClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from dotenv import load_dotenv
# Load .env file (must contain GOOGLE_API_KEY)
load_dotenv()
# Initialize Gemini embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
llm_model = GoogleGenAIChatClient().model

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]
vectorstore = Chroma.from_documents(docs, embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = LLMChainExtractor.from_llm(llm_model)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)
# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)