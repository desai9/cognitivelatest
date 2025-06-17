import os
import logging
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
import pandas as pd
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Setup logging for transparency
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("empowerfin_guardian.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Verify imports
try:
    logger.debug("Successfully imported SystemMessage and HumanMessage")
except NameError as e:
    logger.error(f"Import error: {str(e)}")
    raise

# Initialize ChatGroq client
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    max_tokens=200
)

# Define the shared state for LangGraph
class AgentState(Dict):
    user_consent: bool = False
    bank_statement: Dict = {}
    transactions: List = []
    financial_goal: str = ""
    advice: str = ""
    risk_alert: str = ""
    scenario_plan: str = ""
    empathy_message: str = ""
    logs: List[str] = []
    uploaded_file: Any = None

# Async Groq API Client with retries and error handling using ChatGroq
async def call_groq_api(prompt: str, max_retries: int = 3) -> str:
    messages = [
        SystemMessage(content="You are a helpful financial assistant."),
        HumanMessage(content=prompt)
    ]
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Returning default response.")
                return "Error: Unable to get response from Groq API."
            await asyncio.sleep(2 ** attempt)
    return "Error: Unable to get response from Groq API."

# Node 1: Data Ingestion Agent (Process Bank Statement)
def process_bank_statement(state: AgentState) -> AgentState:
    if not state.get("user_consent", False):
        logger.error("User consent not provided")
        raise ValueError("User consent required to process bank statement")
    
    try:
        if "uploaded_file" not in state or state["uploaded_file"] is None:
            logger.error("No file uploaded")
            raise ValueError("No bank statement file uploaded")
        
        uploaded_file = state["uploaded_file"]
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        required_columns = ["date", "description", "amount"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}. Missing: {', '.join(missing_columns)}")
        
        transactions = df.to_dict(orient="records")
        
        for tx in transactions:
            desc = tx["description"].lower()
            if "grocery" in desc or "restaurant" in desc:
                tx["category"] = "discretionary"
            elif "rent" in desc or "utility" in desc:
                tx["category"] = "fixed"
            else:
                tx["category"] = "other"
        
        state["transactions"] = transactions
        state["bank_statement"]["balance"] = sum(tx["amount"] for tx in transactions)
        state["logs"].append("Bank statement processed: categorized transactions and calculated balance.")
        logger.info("Bank statement processed successfully")
    except Exception as e:
        logger.error(f"Error processing bank statement: {str(e)}")
        raise
    return state

# Node 2: FinTrust Agent (Ethical Financial Advisor - Idea 1)
async def fintrust_agent(state: AgentState) -> AgentState:
    try:
        total_expenses = sum(abs(tx["amount"]) for tx in state["transactions"] if tx["amount"] < 0)
        goal = state["financial_goal"]
        
        prompt = PromptTemplate(
            input_variables=["goal", "expenses"],
            template="Generate ethical financial advice for a user with goal '{goal}' and monthly expenses of ${expenses}."
        ).format(goal=goal, expenses=total_expenses)
        
        advice = await call_groq_api(prompt)
        state["advice"] = advice
        state["logs"].append(f"FinTrust Agent: Generated advice - {advice}")
        logger.info("FinTrust Agent completed successfully")
    except Exception as e:
        logger.error(f"FinTrust Agent error: {str(e)}")
        state["advice"] = "Error: Unable to generate financial advice."
        state["logs"].append("FinTrust Agent: Failed to generate advice due to API error.")
    return state

# Node 3: Wellness Agent (Proactive Risk Mitigation - Idea 2)
async def wellness_agent(state: AgentState) -> AgentState:
    try:
        balance = state["bank_statement"]["balance"]
        upcoming_bills = sum(abs(tx["amount"]) for tx in state["transactions"] if tx["category"] == "fixed")
        
        if balance - upcoming_bills < 0:
            prompt = f"Generate a risk alert for a balance of ${balance} and upcoming bills of ${upcoming_bills}."
            risk_alert = await call_groq_api(prompt)
        else:
            risk_alert = "No immediate financial risks detected."
        
        state["risk_alert"] = risk_alert
        state["logs"].append(f"Wellness Agent: Generated risk alert - {risk_alert}")
        logger.info("Wellness Agent completed successfully")
    except Exception as e:
        logger.error(f"Wellness Agent error: {str(e)}")
        state["risk_alert"] = "Error: Unable to generate risk alert."
        state["logs"].append("Wellness Agent: Failed to generate risk alert due to API error.")
    return state

# Node 4: Cognitive Twin Agent (Predictive Planning - Idea 3)
async def cognitive_twin_agent(state: AgentState) -> AgentState:
    try:
        goal = state["financial_goal"]
        balance = state["bank_statement"]["balance"]
        
        prompt = PromptTemplate(
            input_variables=["goal", "balance"],
            template="Generate a financial scenario plan for a user with goal '{goal}' and current balance of ${balance}."
        ).format(goal=goal, balance=balance)
        
        scenario_plan = await call_groq_api(prompt)
        state["scenario_plan"] = scenario_plan
        state["logs"].append(f"Cognitive Twin Agent: Generated scenario plan - {scenario_plan}")
        logger.info("Cognitive Twin Agent completed successfully")
    except Exception as e:
        logger.error(f"Cognitive Twin Agent error: {str(e)}")
        state["scenario_plan"] = "Error: Unable to generate scenario plan."
        state["logs"].append("Cognitive Twin Agent: Failed to generate scenario plan due to API error.")
    return state

# Node 5: Empathy Auditor Agent (Empathy Enhancement - Idea 4)
async def empathy_auditor_agent(state: AgentState) -> AgentState:
    try:
        risk_alert = state["risk_alert"]
        
        if "overdraft" in risk_alert.lower():
            prompt = f"Rephrase this risk alert to be more empathetic: {risk_alert}"
            empathy_message = await call_groq_api(prompt)
        else:
            empathy_message = "All messages are empathetic and supportive."
        
        state["empathy_message"] = empathy_message
        state["logs"].append(f"Empathy Auditor Agent: Generated empathy message - {empathy_message}")
        logger.info("Empathy Auditor Agent completed successfully")
    except Exception as e:
        logger.error(f"Empathy Auditor Agent error: {str(e)}")
        state["empathy_message"] = "Error: Unable to generate empathy message."
        state["logs"].append("Empathy Auditor Agent: Failed to generate empathy message due to API error.")
    return state

# Node 6: Output Agent (Just log the state, rendering moved to main)
def output_agent(state: AgentState) -> AgentState:
    logger.info("Generating final output")
    logger.debug(f"Final state contents: {state}")
    state["logs"].append("Output agent: Processed final state for display.")
    return state

# Define the LangGraph Workflow
def build_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("process_bank_statement", process_bank_statement)
    workflow.add_node("fintrust_agent", fintrust_agent)
    workflow.add_node("wellness_agent", wellness_agent)
    workflow.add_node("cognitive_twin_agent", cognitive_twin_agent)
    workflow.add_node("empathy_auditor_agent", empathy_auditor_agent)
    workflow.add_node("output_agent", output_agent)
    
    workflow.set_entry_point("process_bank_statement")
    workflow.add_edge("process_bank_statement", "fintrust_agent")
    workflow.add_edge("fintrust_agent", "wellness_agent")
    workflow.add_edge("wellness_agent", "cognitive_twin_agent")
    workflow.add_edge("cognitive_twin_agent", "empathy_auditor_agent")
    workflow.add_edge("empathy_auditor_agent", "output_agent")
    workflow.add_edge("output_agent", END)
    
    return workflow.compile()

# Function to render the dashboard in Streamlit
def render_dashboard(state: AgentState):
    st.header("EmpowerFin Guardian Dashboard")
    st.subheader(f"Financial Goal: {state.get('financial_goal', 'Not set')}")
    st.write(f"**Current Balance:** ${state.get('bank_statement', {}).get('balance', 0):.2f}")
    st.write(f"**Ethical Financial Advice:** {state.get('advice', 'Not available')}")
    st.write(f"**Risk Alert:** {state.get('risk_alert', 'Not available')}")
    st.write(f"**Scenario Plan:** {state.get('scenario_plan', 'Not available')}")
    st.write(f"**Empathy Adjustment:** {state.get('empathy_message', 'Not available')}")
    
    st.subheader("Transparency Logs")
    logs = state.get("logs", [])
    if logs:
        for log in logs:
            st.write(f"- {log}")
    else:
        st.write("No logs available.")

# Streamlit App
def main():
    st.title("EmpowerFin Guardian")
    st.write("A personalized financial advisor powered by AI")

    # User consent (hardcoded for now)
    user_consent = True

    # Input for financial goal
    financial_goal = st.text_input("Enter your financial goal (e.g., 'Save for a car in 2 years'):", value="")
    if not financial_goal:
        financial_goal = "Save for a house in 5 years"
        st.info(f"No goal provided. Using default goal: {financial_goal}")

    # File uploader for bank statement
    uploaded_file = st.file_uploader("Upload your bank statement (CSV)", type=["csv"])
    
    # Validate inputs before proceeding
    if st.button("Generate Financial Dashboard"):
        if not user_consent:
            st.error("User consent is required to proceed.")
            return
        
        if uploaded_file is None:
            st.error("Please upload a bank statement CSV file to proceed.")
            return

        with st.spinner("Processing your financial data..."):
            # Initialize state
            initial_state = AgentState(
                user_consent=user_consent,
                financial_goal=financial_goal,
                bank_statement={},
                transactions=[],
                logs=[],
                uploaded_file=uploaded_file
            )
            
            # Build and run the workflow
            app = build_workflow()
            try:
                result = asyncio.run(app.ainvoke(initial_state))
                st.success("Analysis complete!")
                # Render the dashboard with the final state
                render_dashboard(result)
            except Exception as e:
                logger.error(f"Workflow failed: {str(e)}")
                st.error(f"Error: Workflow failed - {str(e)}")

# Entry point for Streamlit
if __name__ == "__main__":
    main()
