import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initialization components...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)
llm = ChatOpenAI(model="gpt-4.1-mini")

vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
promt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    print("Retrieving...")
    # Query
    query = "what is Pinecone in machine learning?"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke(query)
    print("\nAnswer:")
    print(result_raw.content)
