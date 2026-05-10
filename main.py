from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"]:
        return END
    return ACT

######Graph Implementation#####

flow = StateGraph(MessagesState)

#Adding Nodes
flow.set_entry_point(AGENT_REASON)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

#Adding Edges
flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON,should_continue, {
    END:END,
    ACT:ACT})
flow.add_edge(ACT, AGENT_REASON)

#Generating Graph
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")



def main():
    print("Hello from project-langgraph!")


if __name__ == "__main__":
    main()
