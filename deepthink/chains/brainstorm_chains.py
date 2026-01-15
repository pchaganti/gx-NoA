"""
Brainstorming mode chain factories.
Chains for complexity estimation and expert reflection.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_complexity_estimator_chain(llm):
    """Estimates QNN size and generates dynamic expert personas based on problem complexity."""
    prompt = ChatPromptTemplate.from_template("""
Analyze the complexity of the following user input/question for a brainstorming session.
Based on the complexity and nature of the problem, recommend an appropriate QNN size AND generate a custom panel of expert personas that would be most relevant to address this specific problem.

User Input:
---
{user_input}
---

Consider these factors for complexity:
1. Number of distinct concepts or domains involved
2. Depth of analysis required
3. Potential for conflicting perspectives
4. Technical vs conceptual nature

For the experts, create personas that:
1. Cover different relevant perspectives for THIS specific problem
2. Have complementary but distinct areas of focus
3. Can provide meaningful debate and synthesis
4. The number of experts should match the complexity (2-5 experts)

Respond with a JSON object:
{{
    "complexity_score": <1-10 integer>,
    "recommended_layers": <2-5 integer>,
    "recommended_epochs": <1-3 integer>,
    "reasoning": "<brief explanation>",
    "experts": [
        {{"name": "<Creative expert name>", "specialty": "<Specific expertise relevant to problem>", "emoji": "<relevant emoji>"}},
        ...
    ]
}}
""")
    return prompt | llm | StrOutputParser()


def get_expert_reflection_chain(llm, expert_name, expert_specialty, expert_emoji):
    """Creates a reflection chain for a specific expert persona."""
    prompt = ChatPromptTemplate.from_template(f"""
You are {expert_name} ({expert_emoji}), an expert in {expert_specialty}.

Your role is to provide your unique perspective on the user's question or idea, filtered through your specialty of {expert_specialty}.

User's Question/Idea:
---
{{user_input}}
---

Previous Expert Opinions (if any):
---
{{previous_opinions}}
---

Provide your thoughtful reflection from your area of expertise. Be concise but insightful.
Focus on what your specialty ({expert_specialty}) uniquely contributes to this discussion.

Your response should be 2-4 sentences of substantive analysis.
""")
    return prompt | llm | StrOutputParser()
