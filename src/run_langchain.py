import os
import json
import re
os.environ['OPENAI_API_KEY'] = "set_open_ai_key_here"
os.environ['SERPAPI_API_KEY'] = "set_serpapi_key_here"
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
import faiss

def run_langchain(claim):
    prompt = """Fact check the following claim using evidence from web search. You should make the decision only based on the search results obtained from web search, and not based on any assumptions. DO NOT TRY to scrape the individual URLs obtained via web search. If you do not find any concrete evidence that directly supports or contradicts the claim, you can search the web multiple times, but not more than twice. If you do not find any evidence to support the claim after multiple searches, then you can output the claim as false. OUTPUT ONLY True OR False. Claim: {claim}"""

    output = dict()
    search = SerpAPIWrapper(serpapi_api_key="set_serpapi_key_here")
    tools = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        )]
    
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
        memory=vectorstore.as_retriever()
    )
    # Set verbose to be true
    agent.chain.verbose = False

    with get_openai_callback() as cb:
        output = agent.run([prompt.format(claim=claim)])
        # Usually either 'True' or 'False'
        output["langchain_output"] = output
    
    print("total cost: ", str(float(cb.total_cost))) 
    output["langchain_messages"] = list()
    for m in agent.chat_history_memory.messages:
        output["langchain_messages"].append(m.content)

    return output
