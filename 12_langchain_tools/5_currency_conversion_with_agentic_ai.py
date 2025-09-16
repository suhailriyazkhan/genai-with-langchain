# tool create
from langchain_core.tools import InjectedToolArg, tool
from typing import Annotated
import requests
from utils.gemini_client import GoogleGenAIChatClient
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType

llm = GoogleGenAIChatClient().model

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
  response = requests.get(url)
  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """
  return base_currency_value * conversion_rate

# get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
# convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})
#
# llm_with_tools = llm.bind_tools([get_conversion_factor, convert])
#
# messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]
#
# ai_message = llm_with_tools.invoke(messages)
# messages.append(ai_message)
# print(ai_message.tool_calls)


# Step 5: Initialize the Agent ---
agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)

# --- Step 6: Run the Agent ---
user_query = "What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd?"
response = agent_executor.invoke({"input": user_query})
print(response)