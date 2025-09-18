# test.py
from google.adk.clients.agent_engine import AgentEngineClient

def main():
    # Replace with your full agent name
    agent_name = "projects/1055281703932/locations/us-central1/reasoningEngines/788239885952614400"

    # Create a client (auth is handled with ADC, e.g. gcloud auth application-default login)
    client = AgentEngineClient()

    # Create a session
    session = client.create_session(agent_name=agent_name)

    print(f"Created session: {session.name}")

    # Send a simple message
    response = client.send_message(
        session_name=session.name,
        message="Hello agent, this is a test!"
    )

    print("Agent response:")
    print(response)

if __name__ == "__main__":
    main()
