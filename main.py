from dotenv import load_dotenv
from src.workflow import AgentWorkflow

load_dotenv()
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_AI_STUDIO_KEY")
    print(os.getenv("GOOGLE_AI_STUDIO_KEY"))


def main():
    agent = AgentWorkflow()
    
    print("\n" + "="*70)
    print("🚀 Developer Tools Research Agent")
    print("="*70)
    print("\nI'm here to help you find and evaluate developer tools!")
    print("Just tell me what you're looking for, and I'll guide you through.")
    print("\nType 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        query = input("👤 You: ").strip()
        
        if query.lower() in {"quit", "exit", "bye"}:
            print("\n👋 Goodbye! Come back anytime you need tool recommendations.\n")
            break
        
        if query:
            # Start conversational workflow
            final_state = agent.run_conversation(query)
            
            # After conversation ends, show summary if research was done
            if final_state.researched_companies:
                print("\n" + "="*70)
                print("📋 RESEARCH SUMMARY")
                print("="*70)
                
                for i, company in enumerate(final_state.researched_companies, 1):
                    print(f"\n{i}. 🏢 {company.name}")
                    print(f"   🌐 {company.website}")
                    print(f"   💰 Pricing: {company.pricing_model}")
                    print(f"   📖 Open Source: {'Yes' if company.is_open_source else 'No' if company.is_open_source is False else 'Unknown'}")
                    
                    if company.tech_stack:
                        print(f"   🛠️  Tech: {', '.join(company.tech_stack[:5])}")
                    
                    if company.language_support:
                        print(f"   💻 Languages: {', '.join(company.language_support[:5])}")
                    
                    if company.api_available:
                        print(f"   🔌 API: ✅ Available")
                    
                    if company.integration_capabilities:
                        print(f"   🔗 Integrations: {', '.join(company.integration_capabilities[:4])}")
                
                if final_state.analysis:
                    print("\n" + "-"*70)
                    print("💡 RECOMMENDATION")
                    print("-"*70)
                    print(final_state.analysis)
            
            print("\n" + "="*70)
            print("🔄 Ready for your next question!")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()