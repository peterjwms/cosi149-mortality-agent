# 安装必要库（先执行）
# !pip install langchain-core langchain-google-genai google-generativeai

import os
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, List, Dict
import csv
from io import StringIO
from langchain.tools import BaseTool, Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from typing import Optional, Type, List, Dict, Any, Union
import pandas as pd

from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# Access them like regular environment variables
api_key = os.getenv("GOOGLE_API_KEY")
# TODO: set API key in environment variables
print("Using Gemini API key:", api_key[:5] + "*****")

# 设置Gemini API密钥
# os.environ["GOOGLE_API_KEY"] 
from typing import Optional, Type
from pydantic import BaseModel, Field
import csv
from io import StringIO
import os
from langchain.tools import Tool
import pandas as pd
from langchain.tools import StructuredTool
# 修正后的CSV工具类

from tPatchGNN.run_samples import Inspector
from tpatch_lib.parse_datasets import parse_datasets

import torch
import tpatch_lib.utils as utils
import joblib

MODEL, DATA_OBJ = None, None
DF = None
MORTALITY_MODEL = None

class model_load_input(BaseModel):
    ckpt_path: str = Field(description="check model path")


def load_model(ckpt_path:str) -> str:
    print("Loading model from:", ckpt_path)
    model, args = Inspector.load_ckpt(ckpt_path, torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
    args.batch_size = 1
    args.n=100
    data_obj = parse_datasets(args, patch_ts=True)
    global MODEL, DATA_OBJ
    MODEL = model
    DATA_OBJ = data_obj
    return "Model loaded successfully."


class csv_load_input(BaseModel):
    csv_file_path: str = Field(description="CSV文件路径")
    ids: Union[float, int, str] = Field(default=None, description="需要查询的ID")

def load_csv(csv_file_path: str, ids: Union[float, int, str]=None) -> str:
    """
    读取CSV文件并执行查询。
    :param csv_input: CSV文件路径或直接的CSV文本内容
    :param query: 需要执行的查询
    :return: 查询结果
    """
    # 检查输入是否为文件路径

    df = pd.read_csv(csv_file_path, encoding='utf-8')
    print(df, ids)

    # 根据查询执行操作
    if ids != None:
        df = df[df['HADM_ID'] == float(ids)]
        global DF
        DF = df
        return df.to_string(index=False)  # 返回筛选后的数据
    else:
        return "不支持的查询类型，请提供有效的查询。"

def infer(*args):
    
    print('inside infer', *args)
    model = MODEL
    dataloader = DATA_OBJ['train_dataloader']
    n_batches = DATA_OBJ['n_train_batches']

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)
        if batch_dict is None:
            continue
        pred_y = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			batch_dict["observed_mask"]) 
        pprint(pred_y)

def predict_mortality(input_csv):
	# Load the trained model
	model = joblib.load(Path('tPatchGNN/mortality_prediction_model.pkl'))
	print("Model loaded successfully.")

	# Load the input data
	input_data = pd.read_csv(input_csv)

	# Make predictions
	predictions = model.predict(input_data)
	return predictions

predict_mortality_tool = Tool(
    name="predict_mortality",
    func=predict_mortality,
    description="predict mortality with the model that is specifically for morality prediction. One parameter is needed (input_csv:str).",
)

infer_tool = Tool(
    name="infer",
    func=infer,
    description="infer model with the data that is collected before. No parameter is needed.",
)

# single argument    
get_model_tool = Tool(
    name="load_model",
    func=load_model,
    description="load model for data imputation or computation (ckpt_path:str).",
)
# multiple arguments needs StructuredTool
get_data_tool = StructuredTool(
    name="load_filter_csvfile",
    func=load_csv,
    description="filter data if ids is given. Two parameters are needed (csv_file_path:str, ids:int).",
    args_schema=csv_load_input
)


# 初始化模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0.2
)
# 创建工具列表
tools = [get_data_tool, get_model_tool, infer_tool, predict_mortality_tool]

# 创建结构化提示模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful assistant who can handle both general conversations and CSV data analysis.
                             When a user mentions a CSV file or needs data analysis, use the corresponding tools"""), 
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),

])
# 创建Agent并添加内存

# 创建内存对象


agent = create_tool_calling_agent(llm, tools, prompt)
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
    lambda session_id: memory,  # this could be replaced with Redis or a DB
    input_messages_key="input",
    history_messages_key="chat_history"
)


# 5. 测试对话（正确传递所有变量）
def chat_with_agent(query: str) -> str:
    return agent_with_memory.invoke({
        "input": query,
    }, config={"configurable": {"session_id": "<foo>"}})

# 测试用例
# print(chat_with_agent("你好，今天怎么样？"))  # 常规对话
# print(chat_with_agent(r"get data from C:\Users\huzep\Desktop\149\t-PatchGNN\data\mimic\raw\test.csv. The ids is 100007"))  # 使用工具
# print(chat_with_agent(r"load model from C:\Users\huzep\Desktop\149\t-PatchGNN\tPatchGNN\experiments\experiment_48851.ckpt"))  # 使用工具
# print(chat_with_agent("infer the model"))  # 使用工具
print(chat_with_agent(r"predict mortality in the file C:\Users\peter\VSCodeProjects\brandeis\cosi149\project2\t-patchGNN\tPatchGNN\AI_agent_test_sepsis_features.csv"))  # 使用工具


def run_agent_chat():
    print("\n🤖 Gemini Smart CSV ChatBot")
    print("💬 Example questions:")
    print("   - Hi 👋")
    print("   - Set the file to 'employees.csv'")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            response = agent_with_memory.invoke({'input': user_input}, config={"configurable": {"session_id": "<foo>"}})
            print(f"\n🤖 Gemini:\n{response}\n")
        except Exception as e:
            print(f"❌ Agent Error: {e}")


run_agent_chat()