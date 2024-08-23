from pinecone import Pinecone
from dotenv import load_dotenv
import os
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import json
from openai import AzureOpenAI


load_dotenv()
pc = Pinecone(api_key="96103b4e-bd80-473e-82a5-bee9b39530eb")

index = pc.Index("llamaindex-rag-fs")

api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = "2023-07-01-preview"
azure_deployment="gpt4o-mini"

llm = AzureOpenAI(
    api_key=api_key,
    azure_deployment=azure_deployment,
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

details = index.describe_index_stats()

query = "briefly explain the ballerina auth library"
query_embedding = embed_model.get_query_embedding(query)

response = index.query(top_k=3,vector=query_embedding,include_metadata=True)
response_dict = response.to_dict()
   
json_data = json.loads(response_dict["matches"][1]["metadata"]["_node_content"])

GROUNDED_PROMPT = """You are a knowledgeable assistant specialized in providing step-by-step solutions in the Ballerina programming language. When a user queries how something can be accomplished, answer the query using only the information provided in the sources below. Do not provide any information, suggestions, or explanations that are not explicitly stated in the sources. If the provided sources do not contain enough information to answer the query, respond with: "I don't know." Ensure that your response is fully grounded in the listed sources and maintain accuracy.
Query: {query}
Sources:{sources}
"""

response = llm.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": GROUNDED_PROMPT.format(query=query, sources=json_data['text'])
        }
    ],
    model="gpt-4o-mini",
)

print(response.choices[0].message.content)