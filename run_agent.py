import os
import sys
from typing import Union
from dotenv import load_dotenv
import joblib
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.sequential import SequentialChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
import torch

from langchain.tools import StructuredTool, Tool


sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN"))
sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN/tpatch_lib"))
sys.path.append(str(Path(__file__).resolve().parent / "t-patchGNN/tPatchGNN"))
from tPatchGNN.model.tPatchGNN import tPatchGNN
from tpatch_lib import utils
from tpatch_lib.parse_datasets import parse_datasets

IMP_MODEL, DATA_OBJ = None, None

MORT_MODEL = None


def load_ckpt(checkpt_path=None):
    if checkpt_path is None:
        checkpt_path = Path("t-patchGNN/tPatchGNN/experiments/experiment_48851.ckpt")

    checkpt = torch.load(checkpt_path, weights_only=False, map_location=torch.device("cpu"))
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dicts']

    model = tPatchGNN(ckpt_args)
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(state_dict)

    device = torch.device('cpu')
    model.to(device)
    return model, ckpt_args


def load_imputation_model():

    model, ckpt_args = load_ckpt()
    ckpt_args.batch_size = 1
    ckpt_args.n = 100
    ckpt_args.device = torch.device('cpu')

    data_obj = parse_datasets(ckpt_args, patch_ts=True)
    global MODEL, DATA_OBJ
    MODEL = model
    DATA_OBJ = data_obj


def impute_data(file_path):
    # get the correct rows of data that match the patient_id
    load_imputation_model()

    # df = pd.read_csv(file_path, encoding='utf-8')
    # df = df[df['icustayid'] == float(patient_id)]

    if 'train' in file_path:
        n_batches = DATA_OBJ['n_train_batches']
        dataloader = DATA_OBJ['train_dataloader']
    elif 'test' in file_path:
        n_batches = DATA_OBJ['n_test_batches']
        dataloader = DATA_OBJ['test_dataloader']

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)
        # print(batch_dict)
        if batch_dict is None:
            continue
        # print(batch_dict)

        pred_y = MODEL.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"]
        )
        print(batch_dict['observed_data'])
        print(pred_y)
        observed_data = batch_dict["observed_data"]
        observed_mask = batch_dict["observed_mask"].bool()
        imputed = observed_data.clone()
        print(imputed.shape, observed_mask.shape, pred_y.shape)
        print(imputed[~observed_mask].shape, pred_y[~observed_mask].shape)
        imputed[~observed_mask] = pred_y[~observed_mask]
        print(imputed)
    # return pred_y





def find_patient_data(file_path, patient_id):
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df[df['icustayid'] == float(patient_id)]
    return df


def load_mortality_model():
    global MORT_MODEL
    MORT_MODEL = joblib.load("mortality_models/RandomForestClassifier_mortality_prediction_model.pkl")
    print("Mortality model loaded successfully.")

    
def predict_mortality(file_path, patient_id):
    """Predict mortality for a given patient ID using the trained model.
    This predicts 90-day mortality for a given patient using different measurements from the patient during their stay in the ICU.
    1 """
    load_mortality_model()

    data = pd.read_csv(file_path)
    filtered_data = data[data['icustayid'] == float(patient_id)]
    # print(filtered_data)
    if filtered_data.empty:
        print(f"No data found for patient ID {patient_id}.")
        return None
    
    y_pred = MORT_MODEL.predict(filtered_data)
    y_pred_proba = MORT_MODEL.predict_proba(filtered_data)[:, 1]  # Probability of mortality
    
    most_common_prediction = pd.Series(y_pred).mode()[0]  # Get the most common prediction
    if most_common_prediction == 1:
        print(f"Predicted mortality for patient {patient_id}: Yes")
    else:
        print(f"Predicted mortality for patient {patient_id}: No")
    
    
    return most_common_prediction, y_pred_proba


# mortality_chain = SequentialChain(
#     chains=[
#         load_imputation_model,
#         predict_mortality,
#     ],
#     input_variables=["file_path", "patient_id"],
#     output_variables=["mortality_prediction", "mortality_probability"],
#     verbose=True
# )


def calculate_statistics(file_path):
            # calculate stats for each feature and save to another file
    if not Path(file_path).is_file():
        return f"File not found at {file_path}. Please try again."
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        stats = df.describe(include="number")

        stats.to_csv(Path(f"{os.path.splitext(file_path)[0]}_stats.csv"))
        
        return stats
    except Exception as e:
        return f"Error processing file: {str(e)}"


def retrieve_stats(file_path, column, statistic):
    if not Path(file_path).is_file():
        return f"File not found at {file_path}. Please try again."
    try:
        if "stats" in file_path:
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            df = calculate_statistics(file_path)
        if column not in df.columns:
            return f"Column '{column}' not found in the file."
        target_column = df[column]
        target_stat = target_column[statistic]
        
        return target_stat
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

class file_and_id_input(BaseModel):
    file_path: str = Field(description="CSV file path")
    patient_id: Union[float, int, str] = Field(default=None, description="Patient ID to predict mortality for")


class retrieve_stat_input(BaseModel):
    file_path: str = Field(description="CSV file path")
    column: str = Field(default=None, description="Column name to retrieve statistics for")
    statistic: str = Field(default=None, description="Statistic to retrieve (e.g., mean, median, std)")


predict_mortality_tool = StructuredTool(
    name="predict_mortality",
    func=predict_mortality,
    description="Predict 90-day mortality for a given patient ID using the trained model. Two parameters required: (file_path:str and patient_id:int).",
    args_schema=file_and_id_input
)

impute_data_tool = Tool(
    name="impute_data",
    func=impute_data,
    description="Impute data for all patients in a given file using the t-PatchGNN model. One parameter required: (file_path:str).",
    
)

calculate_statistics_tool = Tool(
    name="calculate_statistics",
    func=calculate_statistics,
    description="Calculate statistics for a given CSV file. One parameter required: (file_path:str).",
)

find_patient_data_tool = StructuredTool(
    name="find_patient_data",
    func=find_patient_data,
    description="Find patient data for a given patient ID using the t-PatchGNN model. Two parameters required: (file_path:str and patient_id:int).",
    args_schema=file_and_id_input
)

retrieve_stats_tool = StructuredTool(
    name="retrieve_stats",
    func=retrieve_stats,
    description="Retrieve a specific statistic for a given CSV file and column. Two parameters required: (file_path:str and column:str).",
    args_schema=retrieve_stat_input
)


def chat_with_agent(agent_with_memory, query: str) -> str:
    return agent_with_memory.invoke({
        "input": query,
    }, config={"configurable": {"session_id": "<foo>"}})


def run_full_agent(agent_with_memory):
    print("Welcome to the AI agent! Type 'exit' or 'quit' to end the conversation.")
    print("You can ask me about mortality prediction, data imputation, and statistical analysis.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = agent_with_memory.invoke({
                'input': user_input},
                config={"configurable": {"session_id": "<foo>"}})
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    tools = [predict_mortality_tool, 
             impute_data_tool, 
             calculate_statistics_tool,
             find_patient_data_tool,
             retrieve_stats_tool]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful medical assistant who can handle general conversations, statistical data analysis, and predicting values using existing ML models.
                      When given a CSV file, you can retrieve a patient's data, calculate statistics for numeric data, predict mortality for a given patient ID using a trained model, and impute missing data using the t-PatchGNN model."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """In memory implementation of chat message history."""

        messages: list[BaseMessage] = Field(default_factory=list)
        def add_messages(self, messages: list[BaseMessage]) -> None:
            """Add a list of messages to the store"""
            self.messages.extend(messages)
        def clear(self) -> None:
            self.messages = []


    memory = InMemoryHistory()

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,  
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    run_full_agent(agent_with_memory)

    

    test_filepath = Path("AI_agent_test_sepsis_features.csv")

    # load_imputation_model()
    # print(MODEL)
    # print(DATA_OBJ)

    # model = MODEL
    # dataloader = DATA_OBJ['train_dataloader']
    # n_batches = DATA_OBJ['n_train_batches']

    # for i in range(n_batches):
    #     batch_dict = utils.get_next_batch(dataloader)
    #     if batch_dict is None:
    #         continue
    #     print(batch_dict)
    #     if i == 5: break
    #     print(batch_dict["observed_data"].shape)
    #     print(batch_dict["observed_tp"].shape)
    #     print(batch_dict["observed_mask"].shape)
    #     print(batch_dict["tp_to_predict"].shape)
    #     pred_y = model.forecasting(batch_dict["tp_to_predict"], 
	# 		batch_dict["observed_data"], batch_dict["observed_tp"], 
	# 		batch_dict["observed_mask"]) 
    #     print(pred_y) 

    # print("Mortality prediction:")
    # patient_id = 262203  # Example patient ID
    # mortality_prediction, mortality_probability = predict_mortality(test_filepath, patient_id)
    # if mortality_prediction is not None:
    #     print(f"Predicted mortality for patient {patient_id}: {mortality_prediction}")
    #     print(f"Predicted probability of mortality for patient {patient_id}: {mortality_probability}")
    #     print(f"{sum(mortality_probability)}")
    # else:
    #     print(f"No prediction made for patient {patient_id}.")

