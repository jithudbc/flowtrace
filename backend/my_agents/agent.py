import asyncio

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

load_dotenv("my_agents/.env")  # Load environment variables from .env file



retry_config = types.HttpRetryOptions(
    attempts=3,
    initial_delay=1.0,  # seconds
    http_status_codes=[429, 500, 502, 503, 504],
    exp_base = 7
)

def log_model_call(callback_context, llm_request):
    print(f"[{callback_context.agent_name}] calling the model...")
    # None = continue normally


research_agent = Agent(
    name="ResearchAgent",
    description="An agent that can perform research tasks using Google Gemini and other tools.",
    model=Gemini(model="gemini-3.5-flash-lite",retry_config_options=retry_config),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1024,
    ),
    tools=[google_search],
    output_key="research_results",
    instruction="You are a research agent. Use the tools available to gather information and provide detailed results.",
    include_contents="none",         # ignore earlier chat history
    before_model_callback=log_model_call,
)

async def main():
    print("Starting agent...")
    runner = InMemoryRunner(agent=research_agent)
    results = await runner.run_debug("What are the latest advancements in AI research?")
    print("Results:", results)


asyncio.run(main())