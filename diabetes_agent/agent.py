"""
Diabetes Risk Assessment Agent - Educational healthcare assistant
"""

import os

from google.adk.agents import Agent
from google.adk.tools import google_search
from .prompts import AGENT_DESCRIPTION, AGENT_INSTRUCTIONS
# Add in BQ tool support
# NOTE (ADK 2.x): these moved from google.adk.tools.bigquery to
# google.adk.integrations.bigquery. The old path still imports but warns.
from google.adk.integrations.bigquery import BigQueryCredentialsConfig
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.integrations.bigquery.config import WriteMode
from google.genai import types
import google.auth
from google.adk.tools.agent_tool import AgentTool

# Model for both agents. "gemini-flash-latest" tracks the current Flash
# generation; set AGENT_MODEL to pin an explicit version for a class.
MODEL = os.environ.get("AGENT_MODEL", "gemini-flash-latest")

# 1) Application Default Credentials (ADC) — run: gcloud auth application-default login
#    Locally this is you. On Agent Runtime it is the Reasoning Engine service
#    agent, which needs BigQuery roles granted to it — see 4_deploy.sh.
adc, _ = google.auth.default()
bq_creds = BigQueryCredentialsConfig(credentials=adc)

# 2) Configure the BigQuery toolset (BLOCKED prevents writes while you test)
bq_cfg = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
bigquery_toolset = BigQueryToolset(
    credentials_config=bq_creds,
    bigquery_tool_config=bq_cfg,
)


search_agent = Agent(
    name="search_agent",
    model=MODEL,
    description="Google Search helper",
    instruction="Use Google Search to find relevant diabetes related information.",
    tools=[google_search],   # allowed: multiple search tools, if you add more
)

search_tool = AgentTool(agent=search_agent)

# Create the agent with Google Search capability
root_agent = Agent(
    name="diabetes_agent",
    model=MODEL,
    description=AGENT_DESCRIPTION,
    instruction=AGENT_INSTRUCTIONS,
    tools=[bigquery_toolset, search_tool],  # Add Google Search tool
)



# Debug output when run directly
if __name__ == "__main__":
    print(f"✅ Diabetes agent configured")
    print(f"📋 Name: {root_agent.name}")
    print(f"🧠 Model: {root_agent.model}")
    print(f"🛠️ Tools: {len(root_agent.tools)} configured")
