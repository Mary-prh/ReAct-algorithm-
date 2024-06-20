from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import tool
from langchain.tools.render import render_text_description
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union
import requests
from dotenv import load_dotenv

load_dotenv()

# response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en")
# fact = response.json().get("text", "Could not fetch a random fact.")
# print(fact)

@tool()
def get_random_fact() -> str:
    """
    This function fetches a random fact from an online API.
    Returns:
        str: A random fact.
    """
    response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en")
    if response.status_code == 200:
        fact = response.json().get("text", "Could not fetch a random fact.")
        return fact
    else:
        return "Failed to fetch a random fact."

def find_tool_by_name(tool_list, tool_name):
    for tool in tool_list:
        if tool.name == tool_name:
            return  tool
    raise ValueError(f"tool {tool_name} not found!")


if __name__ == '__main__':
    print('ReAct LangChain')
    # print(get_text_length("Hey Langchain!"))

    tools = [get_random_fact]

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
    # partial: pre-fill the prompt template with descriptions and names of the tools available to the agent.
    # This makes the prompt more informative and helps guide the agent in using the tools correctly.
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tools_description,
        tool_names=tools_names
    )

    # tells the model to stop generating text when it encounters these specific strings.
    llm = ChatOpenAI(temperature=0, model_kwargs = {"stop":["\nObservation","Observation"]})
    intermediate_step = []

    # When you ask a question, the prompt will guide the llm on how to think, act, and observe using the provided tools.
    # The AI will follow the structure, uses the tool to get the result, and provides the final answer.
    # agent = {"input": lambda x: x["input"]} | prompt | llm

    # res = agent.invoke(input={"input": "what is the length of 'freedom!'"})
    # print(">>>>> Answer before parser: ",res)

    # so far, we can see how in ReAct algorithm, the query part plugs into the agent and then the agent ellaborate prompt
    # which is sent to llm and the llm works as a reasoning agent to select the correct tool and
    # returns "res" as the output about all the information on what tool is used and so on

    # now we need to parse this output using ReActSingleInputOutputParser
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
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(input={"input": f"Give me a random fact",
                                                                          "agent_scratchpad": intermediate_step})
        # print(">>>>> Answer after execution: ", agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool # find the tool to run rom the list of tools
            tool_to_use = find_tool_by_name(tools, tool_name)
            observation = tool_to_use.func()
            # print(f"Funny Fact: {observation}")
            intermediate_step.append((agent_step, str(observation)))

    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke(input={"input": "what is the length of 'freedom!'",
    #                                                                   "agent_scratchpad": intermediate_step})
    #         print(">>>>> Answer after update: ", agent_step)
    if isinstance(agent_step,AgentFinish):
        print("Final Funny Fact: ",agent_step.return_values['output'])

    # the next step is too check the observation and keep the history of what has been done so far
    # the part Thought in template has been empty so far
    # now we add {agent_scratchpad}
    # and we need to add it also to the dictionary that we feed to prompt: "agent_scratchpad": lambda x: x["agent_scratchpad"]


