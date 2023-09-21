# Import necessary libraries and modules
import os
from dotenv import load_dotenv
from typing import List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check if the API key is present or not
if not OPENAI_API_KEY:
    print("OpenAI API Key not found in .env file!")
else:
    print("OpenAI API Key loaded successfully!")

# Load documents from a CSV file using the TextLoader class
loader = TextLoader("/Users/david.leconte/Downloads/finance_blog_samples.csv")
documents = loader.load()

# Split the loaded documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings for generating document vectors
embeddings = OpenAIEmbeddings()

# Define the connection string for the PostgreSQL database
CONNECTION_STRING = "postgresql+psycopg2://david.leconte@localhost:5432/rag_app"

# [Alternative method for generating connection string from environment variables]
# import os
# CONNECTION_STRING = PGVector.connection_string_from_db_params(
#     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
#     host=os.environ.get("PGVECTOR_HOST", "localhost"),
#     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
#     database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
#     user=os.environ.get("PGVECTOR_USER", "postgres"),
#     password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
# )

# Define a unique name for the collection in the database
COLLECTION_NAME = "NQNews"

# Create a PGVector instance to interact with the database
db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Query the database to find similar documents using similarity search
query = "What are the latest financial news?"
docs_with_score = db.similarity_search_with_score(query)

# Display the results with their similarity scores
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

# Query the database using max marginal relevance search for improved diversity
docs_with_score = db.max_marginal_relevance_search_with_score(query)

# Display the results with their relevance scores
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

# Create a new PGVector instance for document storage
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# Add a new document to the collection
store.add_documents([Document(page_content="foo")])

# Perform similarity search on the added document
docs_with_score = db.similarity_search_with_score("foo")

# [Note: The following lines seem to be testing or debugging lines and may not be necessary]
docs_with_score[0]
docs_with_score[1]

# Create a new PGVector instance with the option to delete the existing collection
db = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)

# Perform similarity search on the new collection
docs_with_score = db.similarity_search_with_score("foo")
docs_with_score[0]

# Convert the store to a retriever for fetching documents
retriever = store.as_retriever()
print(retriever)
