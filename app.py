import chainlit as cl
from chainlit.playground.providers.openai import stringify_function_call

from langchain_core.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents import AgentExecutor
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import AzureChatOpenAI
from langchain.memory import PostgresChatMessageHistory
from langchain_core.runnables import RunnableConfig
import os
from uuid import uuid4
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2023-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    model="gpt-4-turbo",
)

db = SQLDatabase.from_uri(database_uri=os.environ["POSTGRES_CONNECTION_STRING"])
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""
suffix = "I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."


tools = toolkit.get_tools()
prefix = prefix.format(dialect=toolkit.dialect, top_k=5)

messages = [
    SystemMessage(content=prefix),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=suffix),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
input_variables = ["input", "agent_scratchpad"]
prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        )
    )
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

@cl.oauth_callback
def auth_callback(provider_id: str, token: str, raw_user_data, default_app_user):
    print(default_app_user)
    if provider_id == "azure-ad":
        if "@microsoft.com" in raw_user_data["mail"]:
            return default_app_user


@cl.on_chat_start
def start():
    app_user = cl.user_session.get("user")
    message_history = PostgresChatMessageHistory(
        connection_string=os.environ["POSTGRES_CONNECTION_STRING"],
        session_id=app_user.identifier + str(uuid4()),
    )
    cl.user_session.set("message_history", message_history)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    agent_executor = cl.user_session.get("agent_executor")
    config = RunnableConfig(
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
    )
    async with cl.Step(name="SQL Agent") as step_p:
        async for chunk in agent_executor.astream(
            {"input": message.content, "chat_history": message_history.messages}
        ):  # , config = config):
            # Agent Action
            if "actions" in chunk:
                for action in chunk["actions"]:
                    async with cl.Step(name=action.tool) as step_a:
                        step_a.output = f"Calling Tool ```{action.tool}``` with input ```{action.tool_input}```"
            # Observation
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    async with cl.Step(name=action.tool) as step_s:
                        step_s.output = f"```{step.observation}```"
                        step_s.language = "sql"
            # Final result
            elif "output" in chunk:
                message_history.add_user_message(message.content)
                message_history.add_ai_message(chunk["output"])
                await cl.Message(content=chunk["output"]).send()
            else:
                raise ValueError
