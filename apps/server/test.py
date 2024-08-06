import requests
from typing import List, Dict

from config import Config

# Define a function to send requests to the XAgent service
def xagent_request(task: str, model: str, config_file: str) -> str:
    xagent_url = "http://localhost:8090/api/run"  # URL of the running XAgent service
    headers = {"Content-Type": "application/json"}
    payload = {
        "task": task,
        "model": model,
        "config_file": config_file
    }

    try:
        response = requests.post(xagent_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("response", "No response from XAgent service.")
    except requests.RequestException as e:
        return f"Error communicating with XAgent service: {str(e)}"

# Define the agent factory function
def agent_factory():
    # This function can be expanded to include any initial setup required for the agent
    def run_agent(task: str) -> str:
        return xagent_request(task, "gpt-4", "assets/config.yml")
    return run_agent

agent = agent_factory()

# Mocking the Client class and run_on_dataset function for demonstration
class MockClient:
    def __init__(self):
        pass

def run_on_dataset(client: MockClient, dataset_name: str, llm_or_chain_factory, evaluation, concurrency_level: int, verbose: bool):
    # Mock function to simulate running evaluation on a dataset
    dataset = ["example task 1", "example task 2"]  # Placeholder dataset
    results = []

    for task in dataset:
        response = llm_or_chain_factory()(task)
        results.append({"task": task, "response": response})

    return results

client = MockClient()

eval_config = {
    "evaluators": ["qa", "helpfulness", "conciseness"],
    "input_key": "input",
    "eval_llm": "gpt-3.5-turbo",  # Placeholder for LLM configuration
}

# Running the evaluation
chain_results = run_on_dataset(
    client,
    dataset_name="test-dataset",
    llm_or_chain_factory=agent_factory,
    evaluation=eval_config,
    concurrency_level=1,
    verbose=True,
)

# Print the results
for result in chain_results:
    print(f"Task: {result['task']}, Response: {result['response']}")
