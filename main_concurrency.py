from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import tool
from langchain.tools.render import render_text_description
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('CONCURRENCY_API_KEY')


@tool()
def convert_currency(query: str) -> str:
    """
    This function converts an amount from one currency to another using real-time exchange rates.
    Args:
        query (str): The conversion query in the format 'amount from_currency to to_currency' (e.g., '100 USD to CAD').
    Returns:
        str: The converted amount and exchange rate.
    """
    try:
        amount, from_currency, _, to_currency = query.split()
        amount = float(amount)
        # ref: https://www.exchangerate-api.com/docs/pair-conversion-requests
        response = requests.get(f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{from_currency}/{to_currency}")
        if response.status_code == 200:
            data = response.json()
            rate = data.get("conversion_rate")
            if rate:
                converted_amount = amount * rate
                return f"{amount} {from_currency} is {converted_amount:.2f} {to_currency} at an exchange rate of {rate:.4f}."
            else:
                return "Conversion rate not found."
        else:
            return "Failed to fetch the exchange rate."
    except Exception as e:
        return str(e)

def find_tool_by_name(tool_list, tool_name):
    for tool in tool_list:
        if tool.name == tool_name:
            return  tool
    raise ValueError(f"tool {tool_name} not found!")


if __name__ == '__main__':
    print('ReAct LangChain: Currency Conversion')
    tools = [convert_currency]

    # Generate the description and names for the tools
    tools_description = render_text_description(tools=tools)
    tools_names = ", ".join([t.name for t in tools])

    template= """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
   
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tools_description,
        tool_names=tools_names
    )

    # tells the model to stop generating text when it encounters these specific strings.
    llm = ChatOpenAI(temperature=0, model_kwargs = {"stop":["\nObservation","Observation"]})
    intermediate_step = []

    agent = {"input": lambda x: x["input"],
             "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} \
            | prompt \
            | llm \
            | ReActSingleInputOutputParser()

    # res = agent.invoke(input={"input": "what is the length of 'freedom!'"})
    # print(">>>>> Answer after parser: ", res)

    # Execution
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        user_input = input("Enter your currency conversion query (e.g., '100 USD to CAD') or 'exit' to quit: ")

        if user_input.lower() == 'exit':
            break

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(input={"input": user_input,
                                                                          "agent_scratchpad": intermediate_step})
        print(">>>>> Answer after execution: ", agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool # find the tool to run rom the list of tools
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(tool_input)
            print(f"observation: {observation}")
            intermediate_step.append((agent_step, observation))

    if isinstance(agent_step,AgentFinish):
        print("Final Answer: ",agent_step.return_values['output'])



