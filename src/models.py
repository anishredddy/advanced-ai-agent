from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
from datetime import datetime


# ============================================================
# Message tracking
# ============================================================

class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime = datetime.now()
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ============================================================
# Company analysis models (UNCHANGED – SAFE)
# ============================================================

class CompanyAnalysis(BaseModel):
    """Structured output for LLM company analysis focused on developer tools"""
    pricing_model: str
    is_open_source: Optional[bool] = None
    tech_stack: List[str] = []
    description: str = ""
    api_available: Optional[bool] = None
    language_support: List[str] = []
    integration_capabilities: List[str] = []


class CompanyInfo(BaseModel):
    name: str
    description: str
    website: str
    pricing_model: Optional[str] = None
    is_open_source: Optional[bool] = None
    tech_stack: List[str] = []
    competitors: List[str] = []
    api_available: Optional[bool] = None
    language_support: List[str] = []
    integration_capabilities: List[str] = []
    developer_experience_rating: Optional[str] = None


# ============================================================
# TOOL CALL MODELS (ONE TOOL = ONE SCHEMA)
# ============================================================

class SearchToolsCall(BaseModel):
    """Call schema for search_tools"""
    query: str
    num_results: int = 3
    reasoning: str


class ResearchCompanyCall(BaseModel):
    """Call schema for research_company"""
    company_names: List[str]
    reasoning: str


class AnalyzeCompaniesCall(BaseModel):
    """Call schema for analyze_companies"""
    reasoning: str


class EndConversationCall(BaseModel):
    """Call schema for end_conversation"""
    reasoning: str


# ============================================================
# AGENT DECISION MODEL (NO UNIONS, NO anyOf)
# ============================================================

class AgentDecision(BaseModel):
    """
    Agent's decision at each step.
    Exactly ONE action must be populated based on decision_type.
    """

    decision_type: Literal[
        "respond",
        "ask_question",
        "search_tools",
        "research_company",
        "analyze_companies",
        "end"
    ]

    # Used for respond / ask_question
    message: Optional[str] = None

    # Tool-specific calls (schema enforced)
    search_tools: Optional[SearchToolsCall] = None
    research_company: Optional[ResearchCompanyCall] = None
    analyze_companies: Optional[AnalyzeCompaniesCall] = None
    end: Optional[EndConversationCall] = None


# ============================================================
# MAIN AGENT STATE
# ============================================================

class AgentState(BaseModel):
    # Conversation tracking
    conversation_history: List[Message] = []

    # User intent
    current_query: Optional[str] = None

    # Research data
    extracted_tools: List[str] = []
    researched_companies: List[CompanyInfo] = []

    # Control flags
    pending_action: Optional[str] = None
    awaiting_user_input: bool = True

    # Final output
    analysis: Optional[str] = None
    conversation_complete: bool = False

    
