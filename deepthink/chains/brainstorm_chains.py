"""
Brainstorming mode chain factories.
Chains for complexity estimation, expert reflection, and QNN brainstorming.
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


def get_opinion_synthesizer_chain(llm):
    """Synthesizes all expert opinions into a final coherent response."""
    prompt = ChatPromptTemplate.from_template("""
You are a master synthesizer. You have received opinions from multiple experts on a user's question.
Your task is to synthesize these diverse perspectives into a coherent, actionable response.

Original User Question:
---
{user_input}
---

Expert Opinions:
---
{all_opinions}
---

Create a synthesized response that:
1. Identifies key areas of agreement
2. Acknowledges valuable tensions or trade-offs
3. Provides a balanced, actionable conclusion
4. Is concise but comprehensive (3-5 sentences)

Synthesized Response:
""")
    return prompt | llm | StrOutputParser()


# ==================== QNN BRAINSTORMING CHAINS ====================

def get_brainstorming_agent_chain(llm):
    """
    Standard agent chain for brainstorming QNN node.
    Reflects on the input concept from the persona's perspective.
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
{system_prompt}
</System>

Concept to Explore / User Input:
---
{input}
---

Your Task:
Reflect on the input from your specific persona and expertise defined in the system prompt.
Provide a unique insight, a critical question, or a creative expansion.
Do NOT try to solve it algorithmically.
Explore the "why" and "what if".
Your output should be a single paragraph of deep reflection.

Your Reflection:
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_mirror_descent_chain(llm, learning_rate):
    """
    Evolves the agent's persona (system prompt) to encourage divergent thinking.
    """
    prompt = ChatPromptTemplate.from_template(f"""
You are a Persona Evolver. Your task is to slightly modify an agent's system prompt to encourage more detailed, divergent, or specific thinking based on their last output.

Original System Prompt:
---
{{current_prompt}}
---

Last Output from Agent:
---
{{last_output}}
---

Critique instructions:
- If the last output was too generic, make the prompt more specific to the persona's niche.
- If the last output was too algorithmic, modify the prompt to encourage philosophical or creative reasoning.
- If the output was good, reinforce the traits that led to it.
- The 'learning_rate' ({learning_rate}) determines the magnitude of change (0.0 = no change, 2.0 = radical reinvention).

Output ONLY the new system prompt. Do not add any explanation.
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_synthesis_chain(llm):
    """
    Synthesizes multiple agent reflections into a cohesive narrative for the epoch.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a Master Synthesizer of Ideas.
You have heard from a panel of experts who have reflected on a core concept.

Original Concept: {original_request}

Expert Reflections:
{agent_solutions}

Your Task:
Synthesize these perspectives into a rich, multi-faceted summary.
Highlight tensions, agreements, and novel insights.
Do NOT just list what they said. Weave a narrative that advances the understanding of the concept.

Synthesized Narrative:
""")
    return prompt | llm | StrOutputParser()
