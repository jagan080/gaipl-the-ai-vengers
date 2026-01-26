import re

from typing import List
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from mock_actions import mock_restart_service, mock_get_service_status, mock_get_system_logs, mock_send_email

# ✅ Define State Schema
class AgentState(BaseModel):
    messages: List[BaseMessage]
    next: str  # Stores the next action to perform

# ✅ Define tool input schemas
class RestartServiceInput(BaseModel):
    service_name: str

class GetServiceStatusInput(BaseModel):
    service_name: str

class GetSystemLogsInput(BaseModel):
    service_name: str

class SendEmailInput(BaseModel):
    service_name: str

# ✅ Define structured tools
restart_service_tool = StructuredTool.from_function(
    func=mock_restart_service,
    name="Restart Service",
    description="Use this tool to restart a given service by providing its name.",
    args_schema=RestartServiceInput
)

get_service_status_tool = StructuredTool.from_function(
    func=mock_get_service_status,
    name="Get Service Status",
    description="Use this tool to get the status of a given service.",
    args_schema=GetServiceStatusInput
)

get_system_logs_tool = StructuredTool.from_function(
    func=mock_get_system_logs,
    name="Get System Logs",
    description="Use this tool to get the system logs.",
    args_schema=GetSystemLogsInput
)

send_email_tool = StructuredTool.from_function(
    func=mock_send_email,
    name="Send Email",
    description="Use this tool to email to engineering team.",
    args_schema=SendEmailInput
)

# ✅ Define the system prompt
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can manage servers/services."),
    ("human", "{input}")
])

# ✅ Function to determine next action
def decide_action(state: AgentState) -> AgentState:
    latest_message = state.messages[-1].content.lower()

    if "restart" in latest_message and "service" in latest_message:
        return AgentState(messages=state.messages, next="restart_service")
    
    elif "status" in latest_message and "service" in latest_message:
        return AgentState(messages=state.messages, next="get_service_status")
    
    elif "logs" in latest_message and "system" in latest_message:
        return AgentState(messages=state.messages, next="get_system_logs")
    
    elif "send" in latest_message and "email" in latest_message:
        return AgentState(messages=state.messages, next="send_email")
    
    else:
        return AgentState(messages=state.messages, next="respond")  # ✅ Default behavior when no task is found


# ✅ Function to handle normal responses
def respond(state: AgentState):
    ai_response = AIMessage(content="No Actions executed! You can ask me to restart a service or check its status.")
    return AgentState(messages=state.messages + [ai_response],next="decide")

# ✅ Function to restart service
def restart_service(state: AgentState):
    response = restart_service_tool.func(get_service_name_from_state(state))
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")

# ✅ Function to get server status
def get_service_status(state: AgentState):
    response = get_service_status_tool.func(get_service_name_from_state(state))
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")

# ✅ Function to get system logs
def get_system_logs(state: AgentState):
    response = get_system_logs_tool.func(get_service_name_from_state(state))
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")

# ✅ Function to send email
def send_email(state: AgentState):
    response = send_email_tool.func(get_service_name_from_state(state))
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")


# ✅ Create the LangGraph workflow
workflow = StateGraph(AgentState)

def get_service_name_from_state(state: AgentState) -> str:
    # Iterate through messages to find relevant content
    for message in state.messages:
        if isinstance(message, (HumanMessage, AIMessage)):  # Check if it's user input
            content = message.content.lower()
            words = content.split()
            
            # Example: If the user message contains "Order Service", extract it
            for word in words:
                if "service" in word.lower():  # Check for service naming pattern
                    return word  # Return the first match
            
    return "Delivery Service"  # Default if no service name found

# ✅ Define nodes
workflow.add_node("decide", RunnableLambda(decide_action))
workflow.add_node("respond", RunnableLambda(respond))
workflow.add_node("restart_service", RunnableLambda(restart_service))
workflow.add_node("get_service_status", RunnableLambda(get_service_status))
workflow.add_node("get_system_logs", RunnableLambda(get_system_logs))
workflow.add_node("send_email", RunnableLambda(send_email))

# ✅ Define transitions
workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", lambda state: state.next)

# ✅ Compile the graph
graph = workflow.compile()

# ✅ Function to invoke the agent
def run_agent(user_input: str):
    initial_state = AgentState(messages=[HumanMessage(content=user_input)], next="decide")
    response = graph.invoke(initial_state)

    # Convert AddableValuesDict to AgentState before accessing messages
    response_state = AgentState(**response)
    return response_state


# ✅ Running the agent
if __name__ == "__main__":
    user_inputs = [
        "Restart the UAT server",
        "What is the status of the UAT server?",
        "Hello, what can you do?"
    ]
    
    for user_input in user_inputs:
        print(f"\n🟢 User: {user_input}")
        response_state = run_agent(user_input)

        for msg in response_state.messages:
            print(f"🤖 AI: {msg.content}")
