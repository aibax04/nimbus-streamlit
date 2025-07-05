import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# === Load Environment Variables ===
load_dotenv()

# === Initialize Model ===
groq_model = Groq(id="llama-3.3-70b-versatile")

# === Define Agents ===
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# === Multi-Agent Team ===
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=groq_model,
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# === Streamlit UI ===
st.set_page_config(page_title="NIMBUSAI", layout="wide")
st.title("NIMBUS AI")

st.markdown("""
This assistant can:
- 🔍 Search the web with DuckDuckGo  
- 📈 Fetch financial data, news, and analyst recommendations
""")

user_query = st.text_input("💬 Enter your query", placeholder="E.g., Summarize analyst recommendation and share the latest news for NVDA")

if st.button("Run Agent") and user_query:
    st.markdown("### 🤖 AI Response")
    with st.spinner("Thinking..."):
        try:
            response = multi_ai_agent.run(user_query)
            st.markdown(response.messages[-1].content)
        except Exception as e:
            st.error(f"❌ Error: {e}")

