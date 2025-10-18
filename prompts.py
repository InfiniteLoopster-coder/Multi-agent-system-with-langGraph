from langfuse import get_client
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

langfuse = get_client()

# Fetch by label (e.g., "production"); auto-cached by SDK
research_prompt_lf = langfuse.get_prompt("research_analyst", label="production", 
                                         fallback="You are a research analyst... {{user_query}}")
writer_prompt_lf   = langfuse.get_prompt("writer_synth", label="production",
                                         fallback="You are a precise writer... {{notes}}")

# Convert to LangChain templates + link prompt metadata so Langfuse ties runs to the exact prompt version
RESEARCH_PROMPT = ChatPromptTemplate.from_messages(
    research_prompt_lf.get_langchain_prompt()
)
RESEARCH_PROMPT.metadata = {"langfuse_prompt": research_prompt_lf}

WRITE_PROMPT = ChatPromptTemplate.from_messages(
    writer_prompt_lf.get_langchain_prompt()
)
WRITE_PROMPT.metadata = {"langfuse_prompt": writer_prompt_lf}
