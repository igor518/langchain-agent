import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_NAME = "medium-blog-embeddings-v2"
SOURCE = Path("data/raw/medium-article.txt")

if __name__ == "__main__":
    document = Document(
        page_content=SOURCE.read_text(encoding="utf-8"),
        metadata={"source": str(SOURCE)},
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents([document])
    print("Injesting data into Pinecone...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)
    print("Done!")
