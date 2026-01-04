from typing import List
from datetime import datetime
import time
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

from .models import (
    AgentState,
    Message,
    CompanyInfo,
    CompanyAnalysis,
    AgentDecision,
)
from .firecrawl import FireCrawlServices
from .prompts import DeveloperToolsPrompts

load_dotenv()

class AgentWorkflow:
    def __init__(self):
        self.firecrawl = FireCrawlServices()
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_API_TOKEN")
        )
        self.prompts = DeveloperToolsPrompts()
        print()

    # ============================================================
    # MAIN CONVERSATION LOOP
    # ============================================================

    def run_conversation(self, initial_message: str) -> AgentState:
        state = AgentState(
            conversation_history=[
                Message(
                    role="user",
                    content=initial_message,
                    timestamp=datetime.now()
                )
            ],
            current_query=initial_message,
            awaiting_user_input=False
        )

        print("\n" + "=" * 60)
        print("🤖 Developer Tools Agent")
        print("=" * 60 + "\n")

        while not state.conversation_complete:
            state = self.agent_decision_step(state)

            if state.awaiting_user_input and not state.conversation_complete:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("\n🤖 Agent: Goodbye!")
                    state.conversation_complete = True
                    break

                state.conversation_history.append(
                    Message(
                        role="user",
                        content=user_input,
                        timestamp=datetime.now()
                    )
                )
                state.awaiting_user_input = False

        return state

    # ============================================================
    # AGENT DECISION
    # ============================================================

    def agent_decision_step(self, state: AgentState) -> AgentState:
        messages = self._build_llm_messages(state)

        try:
            # Get structured output using OpenAI's response_format
            completion = self.client.chat.completions.create(
                model="gpt-4o",  # or your preferred model
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_decision",
                        "schema": AgentDecision.model_json_schema()
                    }
                }
            )
            
            decision_dict = json.loads(completion.choices[0].message.content)
            decision = AgentDecision(**decision_dict)

            # ---------------- RESPOND / ASK ----------------
            if decision.decision_type in {"respond", "ask_question"}:
                assert decision.message is not None, "Message missing"

                print(f"\n🤖 Agent: {decision.message}")
                state.conversation_history.append(
                    Message(
                        role="assistant",
                        content=decision.message,
                        timestamp=datetime.now()
                    )
                )
                state.awaiting_user_input = True

            # ---------------- SEARCH TOOLS ----------------
            elif decision.decision_type == "search_tools":
                assert decision.search_tools is not None, "search_tools args missing"

                args = decision.search_tools
                print(f"\n🔧 Agent searching tools: {args.query}")

                tools = self._tool_search_tools(
                    query=args.query,
                    num_results=args.num_results
                )
                state.extracted_tools.extend(tools)

                state.conversation_history.append(
                    Message(
                        role="tool",
                        content=f"Found tools: {', '.join(tools)}",
                        timestamp=datetime.now()
                    )
                )

            # ---------------- RESEARCH COMPANY ----------------
            elif decision.decision_type == "research_company":
                assert decision.research_company is not None, "research_company args missing"

                args = decision.research_company
                company_names = args.company_names
    
                # Print batch research message
                if len(company_names) == 1:
                    print(f"\n🔬 Agent researching: {company_names[0]}")
                else:
                    print(f"\n🔬 Agent researching {len(company_names)} companies: {', '.join(company_names)}")
                
                researched_count = 0
                for company_name in company_names:
                    print(f"\n  → Researching {company_name}...")
                    company = self._tool_research_company(company_name)
                    if company:
                        state.researched_companies.append(company)
                        researched_count += 1
                
                state.conversation_history.append(
                    Message(
                        role="tool",
                        content=f"Researched {researched_count}/{len(company_names)} companies: {', '.join(company_names)}",
                        timestamp=datetime.now()
                    )
                )

            # ---------------- ANALYZE COMPANIES ----------------
            elif decision.decision_type == "analyze_companies":
                assert decision.analyze_companies is not None, "analyze_companies args missing"

                print("\n📊 Agent analyzing companies")
                analysis = self._tool_analyze_companies(state)
                state.analysis = analysis

                state.conversation_history.append(
                    Message(
                        role="tool",
                        content="Analysis complete",
                        timestamp=datetime.now()
                    )
                )

                print(f"\n🤖 Agent: {analysis}")
                state.conversation_history.append(
                    Message(
                        role="assistant",
                        content=analysis,
                        timestamp=datetime.now()
                    )
                )
                state.awaiting_user_input = True

            # ---------------- END ----------------
            elif decision.decision_type == "end":
                assert decision.end is not None, "end args missing"

                print("\n🤖 Agent: Ending conversation.")
                state.conversation_complete = True

            else:
                raise ValueError(f"Unknown decision_type: {decision.decision_type}")

        except Exception as e:
            print(f"\n⚠️ Agent error: {e}")
            state.conversation_history.append(
                Message(
                    role="assistant",
                    content="I hit an internal error. Please rephrase.",
                    timestamp=datetime.now()
                )
            )
            state.awaiting_user_input = True

        return state

    # ============================================================
    # MESSAGE BUILDING
    # ============================================================

    def _build_llm_messages(self, state: AgentState):
        """Convert conversation history to OpenAI message format"""
        messages = [
            {"role": "system", "content": self.prompts.AGENT_SYSTEM_PROMPT}
        ]

        for msg in state.conversation_history:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                # OpenAI doesn't have a "tool" role in the same way, so use system
                messages.append({"role": "system", "content": f"[Tool Result] {msg.content}"})

        # Add current context as system message
        messages.append({
            "role": "system",
            "content": f"Current Context:\n{self._build_context_summary(state)}"
        })

        return messages

    def _build_context_summary(self, state: AgentState) -> str:
        summary = []

        if state.extracted_tools:
            summary.append(f"Extracted Tools: {', '.join(state.extracted_tools[:5])} \n"
                           f"→ Next step: Research these tools using 'research_company'"
                           )

        if state.researched_companies:
            summary.append(
                f"Researched Companies: {', '.join(c.name for c in state.researched_companies)} \n"
                f"→ Next step: {'Analyze if enough data' if len(state.researched_companies) >= 2 else 'Research more companies'}"
            )

        if state.analysis:
            summary.append("Analysis Complete: Yes")

        return "\n".join(summary) if summary else "No previous actions taken"

    # ============================================================
    # TOOLS (MATCHED TO STATIC WORKFLOW)
    # ============================================================

    def _tool_search_tools(self, query: str, num_results: int = 3) -> List[str]:
        """Extract tools from articles - matches static workflow logic"""
        start_time = time.time()
        
        print(f"Finding articles about: {query}")

        # Match static workflow: add "tools comparison best alternatives"
        article_query = f"{query} tools comparision best alternatives"
        search_results = self.firecrawl.search_companies(
            article_query,
            num_results=num_results
        )

        elapsed = time.time() - start_time
        print(f"\nFound articles about {query} in {elapsed:.2f} seconds\n")

        # Collect content from all search results
        all_content = ""
        for result in search_results.web or []:
            # Handle both url direct attribute and metadata.url
            url = getattr(result, "url", None)
            if not url and hasattr(result, "metadata"):
                url = getattr(result.metadata, "url", None)

            if url:
                scraped = self.firecrawl.scrape_company_pages(url)
                if scraped and getattr(scraped, "markdown", None):
                    # Match static workflow: limit to 1500 chars per page
                    all_content += scraped.markdown[:1500] + "\n\n"

        # Extract tool names using LLM
        messages = [
            {"role": "system", "content": self.prompts.TOOL_EXTRACTION_SYSTEM},
            {"role": "user", "content": self.prompts.tool_extraction_user(query, all_content)}
        ]

        extraction_start = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            response_content = completion.choices[0].message.content
            tool_names = [
                name.strip()
                for name in response_content.strip().split("\n")
                if name.strip()
            ]

            elapsed = time.time() - extraction_start
            print(f"\nExtracted tools in {elapsed:.2f} seconds\n")

            # Match static workflow: limit to 5 tools
            tools = tool_names[:5]
            print(f"✅ Found {len(tools)} tools: {', '.join(tools)}")
            return tools

        except Exception as e:
            print(f"⚠️ Tool extraction error: {e}")
            return []

    def _tool_research_company(self, company_name: str) -> CompanyInfo | None:
        """Research a specific company - matches static workflow logic"""
        start_time = time.time()
        
        print(f"Researching specific tool: {company_name}")

        # Match static workflow: search for "official site"
        tool_search_results = self.firecrawl.search_companies(
            f"{company_name} official site",
            num_results=1
        )

        elapsed = time.time() - start_time
        print(f"\nFound tool official site in {elapsed:.2f} seconds\n")

        if not tool_search_results or not tool_search_results.web:
            print(f"⚠️ No results found for {company_name}")
            return None

        # Get URL from first result
        result = tool_search_results.web[0]
        url = getattr(result, "url", None)
        if not url and hasattr(result, "metadata"):
            url = getattr(result.metadata, "url", None)

        if not url:
            print(f"⚠️ No URL found for {company_name}")
            return None

        # Initialize company object
        company = CompanyInfo(
            name=company_name,
            description="",
            website=url,
            tech_stack=[],
            competitors=[]
        )

        # Scrape and analyze the company page
        scraped = self.firecrawl.scrape_company_pages(url)
        if scraped and getattr(scraped, "markdown", None):
            content = scraped.markdown

            # Analyze the content
            analysis = self._analyze_company_content(company_name, content)

            # Populate company info from analysis
            company.pricing_model = analysis.pricing_model
            company.is_open_source = analysis.is_open_source
            company.tech_stack = analysis.tech_stack
            company.description = analysis.description
            company.api_available = analysis.api_available
            company.language_support = analysis.language_support
            company.integration_capabilities = analysis.integration_capabilities

        print(f"✅ Researched {company_name}")
        print(company)
        return company

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        """Analyze company content using structured LLM output - matches static workflow"""
        start_time = time.time()

        messages = [
            {"role": "system", "content": self.prompts.TOOL_ANALYSIS_SYSTEM},
            {"role": "user", "content": self.prompts.tool_analysis_user(company_name, content)}
        ]

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "company_analysis",
                        "schema": CompanyAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_dict = json.loads(completion.choices[0].message.content)
            analysis = CompanyAnalysis(**analysis_dict)

            elapsed = time.time() - start_time
            print(f"\nAnalysed {company_name} in {elapsed:.2f} seconds\n")

            return analysis

        except Exception as e:
            print(f"⚠️ Analysis error for {company_name}: {e}")
            # Return default analysis on error
            return CompanyAnalysis(
                pricing_model="Unknown",
                is_open_source=None,
                tech_stack=[],
                description="Analysis failed",
                api_available=None,
                language_support=[],
                integration_capabilities=[]
            )

    def _tool_analyze_companies(self, state: AgentState) -> str:
        """Generate final recommendations - matches static workflow logic"""
        if not state.researched_companies:
            return "No companies researched yet."

        print("Generating recommendations")

        # Match static workflow: serialize companies as JSON
        company_data = ", ".join([
            company.json() for company in state.researched_companies
        ])

        start_time = time.time()

        messages = [
            {"role": "system", "content": self.prompts.RECOMMENDATIONS_SYSTEM},
            {"role": "user", "content": self.prompts.recommendations_user(
                state.current_query,
                company_data
            )}
        ]

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        elapsed = time.time() - start_time
        print(f"\nLLM final output in {elapsed:.2f} seconds\n")

        return completion.choices[0].message.content