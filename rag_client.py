from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
import os
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex


load_dotenv()
pc = Pinecone(api_key="96103b4e-bd80-473e-82a5-bee9b39530eb")

pinecone_index = pc.Index("llamaindex-rag-fs")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = "2023-07-01-preview"

llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="gpt4o-mini",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embed",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

details = pinecone_index.describe_index_stats()
print(details)

index = vector_store.



# query = "Describe the auth library in ballerina"
# query_embedding = embed_model.get_query_embedding(query)
# #print(query_embedding)

# response = index.query(top_k=3,vector=query_embedding)
# print(response)

# vector_list = []
# for i in range(len(response["matches"])):
#     vector_list.append(response["matches"][i]["score"])
# #print(vector_list)
