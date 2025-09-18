import vertexai
from diabetes_agent.agent import root_agent
from vertexai import agent_engines

vertexai.init(
    project="patrick-haggerty",
    location="us-central1",
    staging_bucket=f"gs://class-demo",
)



adk_app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)

remote_agent = agent_engines.create(
    adk_app,
    display_name=root_agent.name,
    requirements=[
        "google-adk (>=1.14.0,<2.0.0)",
        "google-cloud-aiplatform[agent_engines] (>=1.111.0,<2.0.0)",
        "google-genai (>=1.38.0,<2.0.0)",
        "pydantic (>=2.11.7,<3.0.0)",
    ],
    #        extra_packages=[""],
)
print(f"Created remote agent: {remote_agent.resource_name}")