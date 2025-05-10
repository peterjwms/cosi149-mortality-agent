from run_agent import *
from train_mortality_models import load_model
from langchain.agents import create_tool_calling_agent



if __name__ == "__main__":
    # Load the model and tools
    model = load_model('mortality_model.pkl')
    tools = [predict_mortality_tool, impute_data_tool, calculate_statistics_tool, find_patient_data_tool]

    # create all the different pieces and then run a simulated prompt to experiment

    # Run the full agent
    run_full_agent(agent_with_memory)