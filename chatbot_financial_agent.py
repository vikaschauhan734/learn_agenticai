import openai
from phi.agent import Agent
from phi.model.groq import Groq
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

# Load envirnoment variables from .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

# Web search agent
web_search_agent = Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()], # We can use multiple tools
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                          company_news=True)],
    instructions=["Use tables to display the data"], # Give basic instructions
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("chatbot_financial_agent:app", reload=True) # here chatbot_financial_agent is the file name and app (line 40) is from where our program will start 
                                                                    # reload = True is like debug = True