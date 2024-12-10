from langchain_core.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.output_parsers.openai_functions import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

import os
from dotenv import load_dotenv
load_dotenv()

llm = AzureChatOpenAI(api_key = os.environ["AZURE_OPENAI_KEY"],  
                      api_version = "2023-12-01-preview",
                      azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
                      model= "gpt-4-turbo")

db = SQLDatabase.from_uri(database_uri = os.environ['POSTGRES_CONNECTION_STRING'])
list_sql_database_tool = ListSQLDatabaseTool(db=db)
info_sql_database_tool_description = (
    "Input to this tool is a comma-separated list of tables, output is the "
    "schema and sample rows for those tables. "
    "Be sure that the tables actually exist by calling "
    f"{list_sql_database_tool.name} first! "
    "Example Input: table1, table2, table3"
)
info_sql_database_tool = InfoSQLDatabaseTool(
    db=db, description=info_sql_database_tool_description
)
query_sql_database_tool_description = (
    "Input to this tool is a detailed and correct SQL query, output is a "
    "result from the database. If the query is not correct, an error message "
    "will be returned. If an error is returned, rewrite the query, check the "
    "query, and try again. If you encounter an issue with Unknown column "
    f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
    "to query the correct table fields."
)
query_sql_database_tool = QuerySQLDataBaseTool(
    db=db, description=query_sql_database_tool_description
)
query_sql_checker_tool_description = (
    "Use this tool to double check if your query is correct before executing "
    "it. Always use this tool before executing a query with "
    f"{query_sql_database_tool.name}!"
)
query_sql_checker_tool = QuerySQLCheckerTool(
    db=db, llm=llm, description=query_sql_checker_tool_description
)

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


tools = [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]
prefix = prefix.format(dialect=db.dialect, top_k=5)

messages = [
    SystemMessage(content=prefix),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=suffix),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
input_variables = ["input", "agent_scratchpad"]
prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

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
response = agent_executor.invoke(
    {
        "input": "How many applicants are there with income less than 1000",
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
    }
)
print(response['output'])