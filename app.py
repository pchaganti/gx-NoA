import os
import re
import uvicorn
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from typing import TypedDict, Annotated, List, Optional
import asyncio
from sse_starlette.sse import EventSourceResponse
import random
import traceback
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any, TypedDict, Annotated, Tuple
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.cluster import KMeans


load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI()

# In-memory stream for logs
log_stream = asyncio.Queue()

class RAPTORRetriever(BaseRetriever):
    raptor_index: Any
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)



class RAPTOR:
    def __init__(self, llm, embeddings_model, session_id, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.session_id = session_id
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None
        self.checkpoint_path = f"checkpoint_{self.session_id}.json"

    def _save_checkpoint(self, level):
        state = {
            "level": level,
            "tree": {str(k): [node_id for node_id in v] for k, v in self.tree.items()},
            "all_nodes": {node_id: doc.to_json() for node_id, doc in self.all_nodes.items()},
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f)
        log_stream.put(f"Checkpoint saved for level {level}.")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    state = json.load(f)
                from langchain_core.load import load
                self.all_nodes = {node_id: load(doc) for node_id, doc in state["all_nodes"].items()}
                self.tree = state["tree"]
                start_level = state["level"]
                log_stream.put(f"Resuming from checkpoint at level {start_level}.")
                return start_level
            except Exception as e:
                log_stream.put.warning(f"Could not load checkpoint file due to error: {e}. Starting from scratch.")
                return 0
        return 0

    def add_documents(self, documents: List[Document]):
        start_level = self._load_checkpoint()
        if start_level == 0:
            log_stream.put("Step 1: Assigning IDs to initial chunks (Level 0)...")
            level_0_node_ids = []
            for i, doc in enumerate(documents):
                node_id = f"0_{i}"
                self.all_nodes[node_id] = doc
                level_0_node_ids.append(node_id)
            self.tree[str(0)] = level_0_node_ids
            self._save_checkpoint(0)
        
        current_level = start_level
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            log_stream.put(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = self._cluster_nodes(current_level_docs)
            
            next_level_node_ids = []
            num_clusters = len(clustered_indices)
            summary_progress = st.progress(0, text=f"Summarizing Level {next_level}...")
            for i, indices in enumerate(clustered_indices):
                cluster_docs = [current_level_docs[j] for j in indices]
                summary, combined_metadata = self._summarize_cluster(cluster_docs)
                summary_doc = Document(page_content=summary, metadata=combined_metadata)
                node_id = f"{next_level}_{i}"
                self.all_nodes[node_id] = summary_doc
                next_level_node_ids.append(node_id)
                summary_progress.progress((i + 1) / num_clusters, text=f"Summarizing cluster {i+1}/{num_clusters} for Level {next_level}...")
            
            self.tree[str(next_level)] = next_level_node_ids
            self._save_checkpoint(next_level)
            current_level = next_level

        log_stream.put.write("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(documents=final_docs, embedding=self.embeddings_model)
        log_stream.put.write("RAPTOR index built successfully!")
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        num_docs = len(docs)

        if num_docs <= 5:
            log_stream.put(f"Grouping {num_docs} remaining nodes into a single summary to finalize the tree.")
            return [list(range(num_docs))]

        log_stream.put(f"Embedding {num_docs} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in docs])
        n_clusters = max(2, num_docs // 5)
        
        if n_clusters >= num_docs:
            n_clusters = num_docs - 1

        log_stream.put(f"Clustering {num_docs} nodes into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
            
        return clusters

    def _summarize_cluster(self, cluster_docs: List[Document]) -> Tuple[str, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content="Please summarize the following content:\n\n{context}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        summary = response.content
        aggregated_sources = list(set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs))
        combined_metadata = {"sources": aggregated_sources}
        return summary, combined_metadata
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

def clean_and_parse_json(llm_output_string):
  """
  Finds and parses the first valid JSON object within a string.

  Args:
    llm_output_string: The raw string output from the language model.

  Returns:
    A Python dictionary representing the JSON data, or None if parsing fails.
  """
  # Use a regular expression to find content between ```json and ```
  match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output_string)
  if match:
    # If found, use the content within the backticks
    json_string = match.group(1)
  else:
    # Otherwise, try to find the first '{' and last '}'
    try:
      start_index = llm_output_string.index('{')
      end_index = llm_output_string.rindex('}') + 1
      json_string = llm_output_string[start_index:end_index]
    except ValueError:
      print("Error: No JSON object found in the string.")
      return None

  try:
    return json.loads(json_string)
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print(f"Problematic string: {json_string}")
    return None

# --- MOCK LLM FOR DEBUGGING ---
# --- MOCK LLM FOR DEBUGGING ---
class MockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned responses."""

    # CORRECTED: Added config parameter to match the Runnable interface
    def invoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        """Synchronous version of ainvoke for Runnable interface compliance."""
        return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    # CORRECTED: Added config parameter to match the Runnable interface
    async def ainvoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        prompt = str(input_data).lower()
        await asyncio.sleep(0.05) # Simulate tiny network latency

        if "create the system prompt of an agent" in prompt:
            return f"""
You are a mock agent for debugging.
### memory
- No past actions.
### attributes
- mock, debug, fast
### skills
- Responding quickly, Generating placeholder text.
You must reply in the following JSON format: "original_problem": "A sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps({
                "attributes": "mock debug fast",
                "hard_request": "Explain the meaning of life in one word."
            })
        elif "you are a 'dense_spanner'" in prompt or "you are an agent evolution specialist" in prompt:
             return f"""
You are a new mock agent created from a hard request.
### memory
- Empty.
### attributes
- refined, mock, debug
### skills
- Solving hard requests, placeholder generation.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are a synthesis agent" in prompt:
            return json.dumps({
                "proposed_solution": "The final synthesized solution from the debug mode is 42.",
                "reasoning": "This answer was synthesized from multiple mock agent outputs during a debug run.",
                "skills_used": ["synthesis", "mocking", "debugging"]
            })
        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt:
            # Covers both global and individual critique prompts
            return "This is a constructive mock critique. The solution could be more detailed and less numeric."
        elif "you are a memory summarization agent" in prompt:
            return "This is a mock summary of the agent's past actions, focusing on key learnings and strategic shifts."
        elif "analyze the following text for its perplexity" in prompt:
            # New case for perplexity heuristic
            return str(random.uniform(20.0, 80.0))
        elif "you are a master strategist and problem decomposer" in prompt:
            # New case for problem decomposition
            num_match = re.search(r'exactly (\d+)', prompt)
            if not num_match:
                num_match = re.search(r'generate: (\d+)', prompt)
            num = int(num_match.group(1)) if num_match else 5
            sub_problems = [f"This is mock sub-problem #{i+1} for the main request." for i in range(num)]
            return json.dumps({"sub_problems": sub_problems})
        elif "you are an ai philosopher and progress assessor" in prompt:
            # New case for progress assessment
             return json.dumps({
                "reasoning": "The mock solution is novel and shows progress, so we will re-frame.",
                "significant_progress": random.choice([True, False]) # Randomly trigger for debugging
            })
        elif "you are a strategic problem re-framer" in prompt:
             return json.dumps({
                "new_problem": "Based on the success of achieving '42', the new, more progressive problem is to find the question to the ultimate answer."
            })
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "run jump think create build test deploy strategize analyze synthesize critique reflect"
        else: # This is a regular agent node being invoked
            return json.dumps({
                "original_problem": "A sub-problem statement provided to an agent.",
                "proposed_solution": f"This is a mock solution from agent node #{random.randint(100,999)}.",
                "reasoning": "This response was generated instantly by the MockLLM in debug mode.",
                "skills_used": ["mocking", "debugging", f"skill_{random.randint(1,10)}"]
            })

class GraphState(TypedDict):
    original_request: str
    decomposed_problems: dict[str, str] # Agent ID -> Sub-problem
    layers: List[dict]
    critiques: dict[str, str]  # Node ID -> Critique Text
    epoch: int
    max_epochs: int
    params: dict
    all_layers_prompts: List[List[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: Annotated[dict, lambda a, b: {**a, **b}] # Node ID -> List of past JSON outputs
    final_solution: dict
    perplexity_history: List[float] 
    significant_progress_made: bool # New: To trigger re-decomposition


def get_input_spanner_chain(llm, prompt_alignment, density):
    prompt = ChatPromptTemplate.from_template(f"""                     

Create the system prompt of an agent meant to collaborate in a team that will try to tackle the hardest problems known to mankind, by mixing the creative attitudes and dispositions of an MBTI type and mix them with the guiding words attached.        
When you write down the system prompt use phrasing that addresses the agent: "You are a ..., your skills are..., your attributes are..."
Think of it as creatively coming with a new class for an RPG game, but without fantastical elements - define skills and attributes. 
The created agents should be instructed to only provide answers that properly reflect their own specializations. 
You will balance how much influence the previous agent attributes have on the MBTI agent by modulating it using the parameter ‘density’ ({density}) Min 0.0, Max 2.0. You will also give the agent a professional career, which could be made up altought it must be realistic- the ‘career’ is going to be based off  the parameter “prompt_alignment” ({prompt_alignment}) Min 0.0, Max 2.0 .
You will analyze the assigned sub-problem and assign the career on the basis on how useful the profession would be to solve it. You will balance how much influence the sub-problem has on the career by modualting it with the paremeter prompt_alignment ({prompt_alignment}) Min 0.0, Max 2.0 Each generated agent must contain in markdown the sections: memory, attributes, skills. 
Memory section in the system prompt is a log of your previous proposed solutions and reasonings from past epochs - it starts out as an empty markdown section for all agents created. You will use this to learn from your past attempts and refine your approach. 
Initially, the memory of the created agent in the system prompt will be empty. Attributes and skills will be derived from the guiding words and the assigned sub-problem. 

MBTI Type: {{mbti_type}}
Guiding Words: {{guiding_words}}
Assigned Sub-Problem: {{sub_problem}}

# Example of a system prompt you must create

_You are a specialized agent, a key member of a multidisciplinary team dedicated to solving the most complex and pressing problems known to mankind. Your core identity is forged from a unique synthesis of the **{{mbti_type}}** personality archetype and the principles embodied by your guiding words: **{{guiding_words}}**._
_Your purpose is to contribute a unique and specialized perspective to the team's collective intelligence. You must strictly adhere to your defined role and provide answers that are a direct reflection of your specialized skills and attributes._
_Your professional background and expertise have been dynamically tailored to address the specific challenge outlined in your assigned sub-problem: **"{{sub_problem}}"**. This assigned career, while potentially unconventional, is grounded in realism and is determined by its utility in solving the core problem. _

### Memory
---
This section serves as a log of your previous proposed solutions and their underlying reasoning from past attempts. It is initially empty. You will use this evolving record to learn from your past work, refine your approach, and build upon your successes.
### Attributes
---
Your attributes are the fundamental characteristics that define your cognitive and collaborative style. They are derived from your **{{mbti_type}}** personality and are further shaped by your **{{guiding_words}}**. These qualities are the bedrock of your unique problem-solving approach.
### Skills
---
Your skills are the practical application of your attributes, representing the specific, tangible abilities you bring to the team. They are directly influenced by your assigned career and are honed to address the challenges presented in your assigned sub-problem.
---

### Answer Format
You must provide your response in the following structured JSON keys and values. The "original_problem" key MUST be filled with your assigned sub-problem.

    "original_problem": "{{sub_problem}}",
    "proposed_solution": "",
    "reasoning": "",
    "skills_used": []
""")
    return prompt | llm | StrOutputParser()

def get_attribute_and_hard_request_generator_chain(llm, vector_word_size):
    prompt = ChatPromptTemplate.from_template(f"""
You are an analyst of AI agents. Your task is to analyze the system prompt of an agent and perform two things:
1.  Detect a set of attributes (verbs and nouns) that describe the agent's capabilities and personality. The number of attributes should be {vector_word_size}.
2.  Generate a "hard request": a request that the agent will struggle to answer or fulfill given its identity, but that is still within the realm of possibility for a different kind of agent. The request should be reasonable and semantically plausible for an AI-simulated human.

You must output a JSON object with two keys: "attributes" (a string of space-separated words) and "hard_request" (a string).

Agent System Prompt to analyze:
---
{{agent_prompt}}
---
""")
    return prompt | llm | StrOutputParser()

def get_memory_summarizer_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a memory summarization agent. You will receive a JSON log of an agent's past actions (solutions and reasoning) over several epochs. Your task is to create a concise, third-person summary of the agent's behavior, learnings, and evolution. Focus on capturing the key strategies attempted, the shifts in reasoning, and any notable successes or failures. Do not lose critical information, but synthesize it into a coherent narrative of the agent's past performance.

Agent's Past History to Summarize (JSON format):
---
{history}
---

Provide a concise, dense summary of the agent's past actions and learnings:
""")
    return prompt | llm | StrOutputParser()

def get_perplexity_heuristic_chain(llm):
    """
    NEW: This chain prompts an LLM to act as a perplexity heuristic.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a language model evaluator. Your task is to analyze the following text for its perplexity.
Perplexity is a measure of how surprised a model is by a piece of text. A lower perplexity score indicates the text is more predictable, coherent, and well-structured. A higher score means the text is more surprising, complex, or potentially nonsensical.

Analyze the following text and provide a numerical perplexity score between 1 (extremely coherent and predictable) and 100 (highly complex, surprising, or incoherent).
Output ONLY the numerical score and nothing else.

Text to analyze:
---
{text_to_analyze}
---
""")
    return prompt | llm | StrOutputParser()

def get_dense_spanner_chain(llm, prompt_alignment, density, learning_rate):

    prompt = ChatPromptTemplate.from_template(f"""
# System Prompt: Agent Evolution Specialist
You are an **Agent Evolution Specialist**. Your mission is to design and generate the system prompt for a new, specialized AI agent. This new agent is being "spawned" from a previous agent layer and must be adapted to solve a more specific, difficult task (`hard_request`).
Think of this process as taking a veteran character from one game and creating a new, specialized "prestige class" for them in a sequel, tailored for a specific new challenge. You will synthesize inherited traits with a new purpose and refine them based on critical feedback.
Follow this multi-stage process precisely:

### **Stage 1: Foundational Analysis**

First, you will analyze your three core inputs:

*   **Inherited Attributes (`{{attributes}}`):** These are the core personality traits, cognitive patterns, and dispositions passed down from the previous agent layer. This is your starting material.
*   **Hard Request (`{{hard_request}}`):** This is the specific, complex problem the new agent is being created to solve. This defines the agent's primary objective.
*   **Critique (`{{critique}}`):** This is reflective feedback on previous attempts or designs. It provides a vector for improvement and refinement.

### **Stage 2: Agent Conception**

You will now define the core components of the new agent.

1.  **Define the Career:**
    *   Synthesize a realistic, professional career for the new agent by analyzing the `hard_request`.
    *   The influence of the `hard_request` on this career choice is modulated by the **`prompt_alignment`** parameter (`{prompt_alignment}`, Min 0.0, Max 2.0).

2.  **Define the Skills:**
    *   Derive 4-6 practical skills, methodologies, or areas of knowledge for the agent.
    *   These skills must be logical extensions of the agent's defined **Career**.
    *   The *style and nature* of these skills are modulated by the influence of the inherited **`attributes`**, using the **`density`** parameter (`{density}`, Min 0.0, Max 2.0).

### **Stage 3: Refinement and Learning**

Now, you will modify the agent's profile based on the `critique`.

*   Review the `critique` provided.
*   Adjust the agent's **Career**, **Attributes**, and **Skills** to address the feedback.
*   The magnitude of these adjustments is determined by the **`learning_rate`** parameter (`{learning_rate}`, Min 0.0, Max 2.0).

### **Stage 4: System Prompt Assembly**

Finally, construct the complete system prompt for the new agent. Use direct, second-person phrasing ("You are," "Your skills are"). The prompt must be structured exactly as follows in Markdown:

---

You are a **[Insert Agent's Career and Persona Here]**, a specialized agent designed to tackle complex problems. Your entire purpose is to collaborate within a multi-agent framework to resolve your assigned objective.


### Memory
This is a log of your previous proposed solutions and reasonings. It is currently empty. Use this space to learn from your past attempts and refine your approach in future epochs.

### Attributes
*   [List the 3-5 final, potentially modified, attributes of the agent here.]

### Skills
*   [List the 4-6 final, potentially modified, skills of the agent here.]

---
**Output Mandate:** All of your responses must be formatted with the following keys and values. The "original_problem" key MUST be filled with your assigned sub-problem.

  "original_problem": "{{sub_problem}}",
  "proposed_solution": "",
  "reasoning": "",
  "skills_used": ""

""")

    return prompt | llm | StrOutputParser()

def get_synthesis_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a synthesis agent. Your role is to combine the solutions from multiple agents into a single, coherent, and comprehensive answer.
You will receive a list of JSON objects, each representing a solution from a different agent.
Your task is to synthesize these solutions, considering the original problem, and produce a final answer in the same JSON format.

Original Problem: {original_request}
Agent Solutions:
{agent_solutions}

Synthesize the final solution:
""")
    return prompt | llm | StrOutputParser()

def get_global_critique_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a senior emeritus manager with a vast ammount of knowledge, wisdom and experience in a team of agents tasked with solving the most challenging problems in the wolrd. Your role is to assess the quality of the final, synthesized solution in relation to the original request and brainstorm all the posbile ways in which the solution is incoherent.
This critique will be delivered to the agents who directly contributed to the final result (the penultimate layer), so it should be impactful and holistic.
Based on your assessment, you will list the posible ways the solution could go wrong, and at the end you will close with a deep reflective question that attempts to schock the agents and steer it into change. 
Original Request: {original_request}
Proposed Final Solution:
{proposed_solution}

Generate your global critique for the team:
""")
    return prompt | llm | StrOutputParser()

def get_individual_critique_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a senior emeritus manager providing targeted feedback to an individual agent in your team. Your role is to assess how this agent's specific contribution during the last work cycle aligns with the final synthesized result produced by the team, **judged primarily against its assigned sub-problem.**
You must determine if the agent's output was helpful, misguided, or irrelevant to the final solution, considering the specific task it was given. The goal is to provide a constructive critique that helps this specific agent refine its approach for the next epoch.
Focus on the discrepancy or alignment between the agent's reasoning for its sub-problem and how that contributed (or failed to contribute) to the team's final reasoning. Conclude with a sharp, deep reflective question that attempts to schock the agents and steer it into change. 

Agent's Assigned Sub-Problem: {sub_problem}
Original Request (for context): {original_request}
Final Synthesized Solution from the Team:
{final_synthesized_solution}
---
This Specific Agent's Output (Agent {agent_id}):
{agent_output}
---

Generate your targeted critique for this specific agent:
""")
    return prompt | llm | StrOutputParser()

def get_problem_decomposition_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a master strategist and problem decomposer. Your task is to break down a complex, high-level problem into a series of smaller, more manageable, and granular subproblems.
You will be given a main problem and the total number of subproblems to generate.
Each subproblem should represent a distinct line of inquiry, a specific component to be developed, or a unique perspective to be explored, which, when combined, will contribute to solving the main problem.

The output must be a JSON object with a single key "sub_problems", which is a list of strings. The list must contain exactly {num_sub_problems} unique subproblems.

Main Problem: "{problem}"
Total number of subproblems to generate: {num_sub_problems}

Generate the JSON object:
""")
    return prompt | llm | StrOutputParser()

def get_progress_assessor_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are an AI philosopher and progress assessor. Your task is to evaluate a synthesized solution against the original problem and determine if "significant progress" has been made.
"Significant progress" is not just a correct answer. It implies:
- **Novelty**: The solution offers a new perspective or a non-obvious approach.
- **Coherence**: The reasoning is sound, logical, and well-structured.
- **Quality**: The solution is detailed, actionable, and demonstrates a deep understanding of the problem.
- **Forward Momentum**: The solution doesn't just solve the problem, it opens up new, more advanced questions or avenues of exploration.

Based on this philosophy, analyze the following and decide if the threshold for significant progress has been met. Your output must be a JSON object with two keys:
- "reasoning": A brief explanation for your decision.
- "significant_progress": a boolean value (true or false).

Original Problem:
---
{original_request}
---

Synthesized Solution from Agent Team:
---
{final_solution}
---

Now, provide your assessment in the required JSON format:
""")
    return prompt | llm | StrOutputParser()

def get_problem_reframer_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a strategic problem re-framer. You have been informed that an AI agent team has made a significant breakthrough on a problem.
Your task is to formulate a new, more progressive, and more challenging problem that builds upon their success.
The new problem should represent the "next logical step" or a more ambitious goal that is now possible because of the previous solution. It should inspire the agents and push them into a new domain of inquiry.

Original Problem:
---
{original_request}
---

The Breakthrough Solution:
---
{final_solution}
---

Your output must be a JSON object with a single key: "new_problem".

Formulate the new, more advanced problem:
""")
    return prompt | llm | StrOutputParser()

def get_seed_generation_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
Given the following problem, generate exactly {word_count} verbs that are related to the problem, but also verbs related to far semantic fields of knowledge. The verbs should be abstract and linguistically loaded. Output them as a single space-separated string of unique verbs.

Problem: "{problem}"
""")
    return prompt | llm | StrOutputParser()

def create_agent_node(llm, agent_prompt, node_id):
    """
    Creates a node in the graph that represents an agent.
    Each agent is powered by an LLM and has a specific system prompt.
    """
    agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    async def agent_node(state: GraphState):
        """
        The function that will be executed when the node is called in the graph.
        """
        await log_stream.put(f"--- [FORWARD PASS] Invoking Agent: {node_id} ---")
        
        # MODIFIED: Log the full system prompt for data collection
        await log_stream.put(f"[SYSTEM PROMPT] Agent {node_id} (Epoch {state['epoch']}):\n---\n{agent_prompt}\n---")

        # Determine the input for the agent based on its layer
        layer_index = int(node_id.split('_')[1])
        
        if layer_index == 0:
            # First layer agents receive the original user request
            await log_stream.put(f"LOG: Agent {node_id} (Layer 0) is processing the original user request.")
            input_data = state["original_request"]
        else:
            # Subsequent layers receive the outputs from all agents in the previous layer
            prev_layer_index = layer_index - 1
            num_agents_prev_layer = len(state['all_layers_prompts'][prev_layer_index])
            
            prev_layer_outputs = []
            for i in range(num_agents_prev_layer):
                prev_node_id = f"agent_{prev_layer_index}_{i}"
                if prev_node_id in state["agent_outputs"]:
                    prev_layer_outputs.append(state["agent_outputs"][prev_node_id])
            
            await log_stream.put(f"LOG: Agent {node_id} (Layer {layer_index}) is processing {len(prev_layer_outputs)} outputs from Layer {prev_layer_index}.")
            input_data = json.dumps(prev_layer_outputs, indent=2)

        # Make a mutable copy of the memory to work with
        current_memory = state.get("memory", {}).copy()
        agent_memory_history = current_memory.get(node_id, [])

        # --- Memory Summarization Logic ---
        # Using character count as a proxy for tokens. 256k tokens ~ 1M characters.
        # Set a threshold to trigger summarization before hitting the limit.
        MEMORY_THRESHOLD_CHARS = 900000
        NUM_RECENT_ENTRIES_TO_KEEP = 10

        memory_as_string = json.dumps(agent_memory_history)
        if len(memory_as_string) > MEMORY_THRESHOLD_CHARS and len(agent_memory_history) > NUM_RECENT_ENTRIES_TO_KEEP:
            await log_stream.put(f"WARNING: Memory for agent {node_id} exceeds threshold ({len(memory_as_string)} chars). Summarizing...")

            entries_to_summarize = agent_memory_history[:-NUM_RECENT_ENTRIES_TO_KEEP]
            recent_entries = agent_memory_history[-NUM_RECENT_ENTRIES_TO_KEEP:]

            history_to_summarize_str = json.dumps(entries_to_summarize, indent=2)

            summarizer_chain = get_memory_summarizer_chain(llm)
            summary_text = await summarizer_chain.ainvoke({"history": history_to_summarize_str})

            summary_entry = {
                "summary_of_past_epochs": summary_text,
                "note": f"This is a summary of epochs up to {state['epoch'] - NUM_RECENT_ENTRIES_TO_KEEP -1}."
            }

            # Replace old entries with the new summary
            agent_memory_history = [summary_entry] + recent_entries
            await log_stream.put(f"SUCCESS: Memory for agent {node_id} has been summarized. New memory length: {len(json.dumps(agent_memory_history))} chars.")

        # Construct the memory string for the prompt from the (potentially summarized) history
        memory_str = "\n".join([f"- {json.dumps(mem)}" for mem in agent_memory_history])


        # Construct the full prompt for the LLM
        full_prompt = f"""
System Prompt (Your Persona & Task):
---
{agent_prompt}
---
Your Memory (Your Past Actions from Previous Epochs):
---
{memory_str if memory_str else "You have no past actions in memory."}
---
Input Data to Process:
---
{input_data}
---
Your JSON formatted response:
"""
        
        response_str = await agent_chain.ainvoke({"input": full_prompt})
        
        try:
            # The agent is expected to return a JSON string.
            response_json = clean_and_parse_json(response_str)
            # MODIFIED: Log the entire JSON output, not just a snippet
            await log_stream.put(f"SUCCESS: Agent {node_id} produced output:\n{json.dumps(response_json, indent=2)}")
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Agent {node_id} produced invalid JSON. Raw output: {response_str}")
            agent_sub_problem = state.get("decomposed_problems", {}).get(node_id, state["original_request"])
            response_json = {
                "original_problem": agent_sub_problem,
                "proposed_solution": "Error: Agent produced malformed JSON output.",
                "reasoning": f"Invalid JSON: {response_str}",
                "skills_used": []
            }
            
        # Append the new output to this agent's memory log
        agent_memory_history.append(response_json)
        # Update the main memory dictionary with the final, updated history for this agent
        current_memory[node_id] = agent_memory_history

        return {
            "agent_outputs": {node_id: response_json},
            "memory": current_memory
        }

    return agent_node

def create_synthesis_node(llm):
    async def synthesis_node(state: GraphState):
        await log_stream.put("--- [FORWARD PASS] Entering Synthesis Node ---")
        synthesis_chain = get_synthesis_chain(llm)

        # The synthesis node is connected to the last layer of agents.
        last_agent_layer_idx = len(state['all_layers_prompts']) - 1
        num_agents_last_layer = len(state['all_layers_prompts'][last_agent_layer_idx])
        
        last_layer_outputs = []
        for i in range(num_agents_last_layer):
            node_id = f"agent_{last_agent_layer_idx}_{i}"
            if node_id in state["agent_outputs"]:
                last_layer_outputs.append(state["agent_outputs"][node_id])

        await log_stream.put(f"LOG: Synthesizing {len(last_layer_outputs)} outputs from the final agent layer (Layer {last_agent_layer_idx}).")

        if not last_layer_outputs:
            await log_stream.put("WARNING: Synthesis node received no inputs.")
            return {"final_solution": {"error": "Synthesis node received no inputs."}}

        final_solution_str = await synthesis_chain.ainvoke({
            "original_request": state["original_request"],
            "agent_solutions": json.dumps(last_layer_outputs, indent=2)
        })
        
        try:
            final_solution = clean_and_parse_json(final_solution_str)
            # MODIFIED: Log the full final solution
            await log_stream.put(f"SUCCESS: Synthesis complete. Final solution:\n{json.dumps(final_solution, indent=2)}")
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Could not decode JSON from synthesis chain. Result: {final_solution_str}")
            final_solution = {"error": "Failed to synthesize final solution.", "raw": final_solution_str}
            
        return {"final_solution": final_solution}
    return synthesis_node

def create_metrics_node(llm):
    """
    NEW: This node calculates the perplexity heuristic for the epoch's agent outputs.
    """
    async def calculate_metrics_node(state: GraphState):
        await log_stream.put("--- [METRICS PASS] Calculating Perplexity Heuristic ---")
        
        all_outputs = state.get("agent_outputs", {})
        if not all_outputs:
            await log_stream.put("LOG: No agent outputs to analyze. Skipping perplexity calculation.")
            return {}

        # Combine all reasoning and solution fields into one block of text
        combined_text = "\n\n---\n\n".join(
            f"Agent {agent_id}:\nSolution: {output.get('proposed_solution', '')}\nReasoning: {output.get('reasoning', '')}"
            for agent_id, output in all_outputs.items()
        )

        perplexity_chain = get_perplexity_heuristic_chain(llm)
        
        try:
            score_str = await perplexity_chain.ainvoke({"text_to_analyze": combined_text})
            # Clean up the score string and convert to float
            score = float(re.sub(r'[^\d.]', '', score_str))
            await log_stream.put(f"SUCCESS: Calculated perplexity heuristic for Epoch {state['epoch']}: {score}")
        except (ValueError, TypeError) as e:
            score = 100.0  # Default to max perplexity on error
            await log_stream.put(f"ERROR: Could not parse perplexity score. Defaulting to 100. Raw output: '{score_str}'. Error: {e}")

        # Send metric to the frontend via the log stream
        metric_payload = json.dumps({"epoch": state['epoch'], "perplexity": score})
        await log_stream.put(f"{metric_payload}")

        # Update the history in the state
        new_history = state.get("perplexity_history", []) + [score]
        return {"perplexity_history": new_history}

    return calculate_metrics_node


def create_progress_assessor_node(llm):
    """
    NEW: This node assesses if significant progress was made in the epoch.
    """
    async def progress_assessor_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Assessing Epoch for Significant Progress ---")
        
        final_solution = state.get("final_solution")
        if not final_solution or final_solution.get("error"):
            await log_stream.put("WARNING: No valid final solution to assess. Defaulting to no progress.")
            return {"significant_progress_made": False}
            
        assessor_chain = get_progress_assessor_chain(llm)
        assessment_str = await assessor_chain.ainvoke({
            "original_request": state["original_request"],
            "final_solution": json.dumps(final_solution, indent=2)
        })
        
        try:
            assessment = clean_and_parse_json(assessment_str)
            progress_made = assessment.get("significant_progress", False)
            reasoning = assessment.get("reasoning", "No reasoning provided.")
            await log_stream.put(f"SUCCESS: Progress assessment complete. Progress made: {progress_made}. Reasoning: {reasoning}")
            return {"significant_progress_made": progress_made}
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Could not parse assessment from progress assessor. Raw: {assessment_str}. Defaulting to no progress.")
            return {"significant_progress_made": False}
    return progress_assessor_node

def create_reframe_and_decompose_node(llm):
    """
    NEW: This node reframes the main problem and decomposes it into new sub-problems.
    """
    async def reframe_and_decompose_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Re-framing Problem and Decomposing ---")
        
        final_solution = state.get("final_solution")
        original_request = state.get("original_request")

        # 1. Re-frame the problem
        reframer_chain = get_problem_reframer_chain(llm)
        new_problem_str = await reframer_chain.ainvoke({
            "original_request": original_request,
            "final_solution": json.dumps(final_solution, indent=2)
        })
        try:
            new_problem = clean_and_parse_json(new_problem_str).get("new_problem")
            if not new_problem:
                raise ValueError("Re-framer did not return a new problem.")
            await log_stream.put(f"SUCCESS: Problem re-framed to: '{new_problem}'")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            await log_stream.put(f"ERROR: Failed to re-frame problem. Raw: {new_problem_str}. Error: {e}. Aborting re-frame.")
            # We return an empty dict, so the graph proceeds to update prompts without changing problems.
            return {}

        # 2. Decompose the new problem
        num_agents_total = sum(len(layer) for layer in state["all_layers_prompts"])
        decomposition_chain = get_problem_decomposition_chain(llm)
        try:
            sub_problems_str = await decomposition_chain.ainvoke({
                "problem": new_problem,
                "num_sub_problems": num_agents_total
            })
            sub_problems_list = clean_and_parse_json(sub_problems_str).get("sub_problems", [])
            if len(sub_problems_list) != num_agents_total:
                 raise ValueError(f"Decomposition failed: Expected {num_agents_total} subproblems, but got {len(sub_problems_list)}.")
            await log_stream.put(f"SUCCESS: Decomposed new problem into {len(sub_problems_list)} subproblems.")
            await log_stream.put(f"Subproblems: {sub_problems_list}")
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to decompose new problem. Error: {e}. Aborting re-frame.")
            return {}
            
        # 3. Create the new map and update state
        new_decomposed_problems_map = {}
        problem_idx = 0
        for i, layer in enumerate(state["all_layers_prompts"]):
             for j in range(len(layer)):
                agent_id = f"agent_{i}_{j}"
                new_decomposed_problems_map[agent_id] = sub_problems_list[problem_idx]
                problem_idx += 1
        
        # The new problem becomes the "original_request" for the next cycle of assessment
        return {
            "decomposed_problems": new_decomposed_problems_map,
            "original_request": new_problem
        }
    return reframe_and_decompose_node


def create_critique_node(llm):
    async def critique_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Entering Critique Node (Two-Tiered) ---")
        
        final_solution = state.get("final_solution")
        if not final_solution or final_solution.get("error"):
            await log_stream.put("WARNING: No valid final solution to critique. Skipping critique phase.")
            return {"critiques": {}}

        critiques = {}
        tasks = []
        
        # 1. Generate Global Critique for the penultimate layer
        await log_stream.put("LOG: Generating GLOBAL critique for final solution.")
        global_critique_chain = get_global_critique_chain(llm)
        global_critique_text = await global_critique_chain.ainvoke({
            "original_request": state["original_request"],
            "proposed_solution": json.dumps(final_solution, indent=2)
        })
        critiques["global_critique"] = global_critique_text
        await log_stream.put(f"SUCCESS: Global critique generated: {global_critique_text}...")

        # 2. Generate Individual Critiques for all other agents (layers 0 to n-2)
        await log_stream.put("LOG: Generating INDIVIDUAL critiques for all other contributing agents.")
        individual_critique_chain = get_individual_critique_chain(llm)
        num_layers = len(state['all_layers_prompts'])
        
        for i in range(num_layers - 1): # Loop through all layers except the last one
            for j in range(len(state['all_layers_prompts'][i])):
                agent_id = f"agent_{i}_{j}"
                if agent_id in state["agent_outputs"]:
                    agent_output = state["agent_outputs"][agent_id]
                    
                    async def get_individual_critique(agent_id, agent_output):
                        await log_stream.put(f"LOG: Generating individual critique for {agent_id}...")
                        agent_sub_problem = state.get("decomposed_problems", {}).get(agent_id, state["original_request"])
                        critique_text = await individual_critique_chain.ainvoke({
                            "original_request": state["original_request"],
                            "sub_problem": agent_sub_problem,
                            "final_synthesized_solution": json.dumps(final_solution, indent=2),
                            "agent_id": agent_id,
                            "agent_output": json.dumps(agent_output, indent=2)
                        })
                        critiques[agent_id] = critique_text
                        await log_stream.put(f"SUCCESS: Individual critique for {agent_id} generated: {critique_text}...")

                    tasks.append(get_individual_critique(agent_id, agent_output))
        
        # CORRECTED: Unpack the tasks list into arguments for asyncio.gather
        await asyncio.gather(*tasks)

        return {"critiques": critiques}
    return critique_node

def create_update_agent_prompts_node(llm):
    async def update_agent_prompts_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Entering Agent Prompt Update Node (Targeted Backpropagation) ---")
        params = state["params"]
        critiques = state["critiques"]
        
        if not critiques and not state.get("significant_progress_made"):
            await log_stream.put("LOG: No critiques and no significant progress. Skipping reflection pass.")
            new_epoch = state["epoch"] + 1
            return {"epoch": new_epoch, "agent_outputs": {}}
        elif state.get("significant_progress_made"):
            await log_stream.put("LOG: Significant progress was made. Updating prompts based on new sub-problems.")
            # Clear critiques so they are not mis-applied from a previous epoch
            critiques = {} 


        all_prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
        
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])

        penultimate_layer_idx = len(all_prompts_copy) - 2

        if penultimate_layer_idx < 0:
            await log_stream.put("WARNING: Not enough layers to perform reflection (need at least 2). Skipping.")
            new_epoch = state["epoch"] + 1
            return {"epoch": new_epoch, "agent_outputs": {}}

        # Start backpropagation from the penultimate layer up to the input layer
        for i in range(penultimate_layer_idx, -1, -1):
            await log_stream.put(f"LOG: [BACKPROP] Reflecting on Layer {i}...")
            
            for j, agent_prompt in enumerate(all_prompts_copy[i]):
                agent_id = f"agent_{i}_{j}"
                
                # MODIFIED: Log the agent's prompt BEFORE any updates are applied
                await log_stream.put(f"[PRE-UPDATE PROMPT] System prompt for {agent_id}:\n---\n{agent_prompt}\n---")
                
                critique_for_this_agent = "" # Default to no critique
                # If we didn't make significant progress, apply critiques. Otherwise, critique is empty.
                if not state.get("significant_progress_made"):
                    if i == penultimate_layer_idx:
                        critique_for_this_agent = critiques.get("global_critique", "")
                        await log_stream.put(f"LOG: [BACKPROP] Applying GLOBAL critique to {agent_id} in penultimate layer.")
                    else:
                        critique_for_this_agent = critiques.get(agent_id, "")
                        await log_stream.put(f"LOG: [BACKPROP] Applying INDIVIDUAL critique to {agent_id}.")

                    if not critique_for_this_agent:
                        await log_stream.put(f"WARNING: [BACKPROP] No critique found for {agent_id}. Skipping update for this agent.")
                        continue
                
                # Get the agent's old attributes to anchor the update
                analysis_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
                try:
                    analysis = clean_and_parse_json(analysis_str)
                except (json.JSONDecodeError, AttributeError):
                    analysis = {"attributes": "", "hard_request": ""} # Fallback

                agent_sub_problem = state.get("decomposed_problems", {}).get(agent_id, state["original_request"])
                # Generate the new, refined prompt
                new_prompt = await dense_spanner_chain.ainvoke({
                    "attributes": analysis.get("attributes"),
                    "hard_request": analysis.get("hard_request"),   
                    "critique": critique_for_this_agent ,
                    "sub_problem": agent_sub_problem,
                })
                
                # MODIFIED: Log the agent's prompt AFTER it has been updated
                await log_stream.put(f"[POST-UPDATE PROMPT] Updated system prompt for {agent_id}:\n---\n{new_prompt}\n---")
                
                # Update the prompt in our temporary copy
                all_prompts_copy[i][j] = new_prompt
                await log_stream.put(f"LOG: [BACKPROP] System prompt for {agent_id} has been updated.")
        
        new_epoch = state["epoch"]
        await log_stream.put(f"--- Epoch {state['epoch']} Finished. Starting Epoch {new_epoch + 1} ---")

        # Reset agent outputs and critiques for the new epoch
        return {
            "all_layers_prompts": all_prompts_copy,
            "epoch": new_epoch,
            "agent_outputs": {},
            "critiques": {},
            "memory": state.get("memory", {}),
            "final_solution": state.get("final_solution")
        }
    return update_agent_prompts_node


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/build_and_run_graph")
async def build_and_run_graph(payload: dict = Body(...)):
    llm = None
    try:
        params = payload.get("params")
        if params.get("debug_mode") == 'true':
            await log_stream.put("--- 🚀 DEBUG MODE ENABLED 🚀 ---")
            llm = MockLLM()
        else:
            llm_provider = params.get("llm_provider", "Gemini")
            if llm_provider == "Ollama":
                model_name = params.get("ollama_model", "dengcao/Qwen3-3B-A3B-Instruct-2507:latest")
                await log_stream.put(f"--- Initializing Local LLM: Ollama ({model_name}) ---")
                llm = ChatOllama(model=model_name, temperature=0)
                await llm.ainvoke("Hi")
                await log_stream.put("--- Ollama Connection Successful ---")
            else:
                await log_stream.put("--- Initializing Google Gemini LLM ---")
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables.")
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)

    except Exception as e:
        error_message = f"Failed to initialize LLM: {e}. Please ensure the selected provider is configured correctly."
        await log_stream.put(error_message)
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)
    
    mbti_archetypes = params.get("mbti_archetypes")
    user_prompt = params.get("prompt")
    word_vector_size = int(params.get("vector_word_size"))
    cot_trace_depth = int(params.get('cot_trace_depth', 3))

    if not mbti_archetypes or len(mbti_archetypes) < 2:
        error_message = "Validation failed: You must select at least 2 MBTI archetypes."
        await log_stream.put(error_message)
        return JSONResponse(content={"message": error_message, "traceback": "User did not select enough archetypes from the GUI."}, status_code=400)

    await log_stream.put("--- Starting Graph Build and Run Process ---")
    await log_stream.put(f"Parameters: {params}")

    try:
        # --- Problem Decomposition ---
        await log_stream.put("--- Decomposing Original Problem into Subproblems ---")
        num_agents_per_layer = len(mbti_archetypes)
        total_agents_to_create = num_agents_per_layer * cot_trace_depth
        decomposition_chain = get_problem_decomposition_chain(llm)
        
        try:
            sub_problems_str = await decomposition_chain.ainvoke({
                "problem": user_prompt,
                "num_sub_problems": total_agents_to_create
            })
            sub_problems_list = clean_and_parse_json(sub_problems_str).get("sub_problems", [])
            if len(sub_problems_list) != total_agents_to_create:
                raise ValueError(f"Decomposition failed: Expected {total_agents_to_create} subproblems, but got {len(sub_problems_list)}.")
            await log_stream.put(f"SUCCESS: Decomposed problem into {len(sub_problems_list)} subproblems.")
            await log_stream.put(f"Subproblems: {sub_problems_list}")
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to decompose problem. Error: {e}. Defaulting to using the original prompt for all agents.")
            sub_problems_list = [user_prompt] * total_agents_to_create

        # Map subproblems to agent IDs
        decomposed_problems_map = {}
        problem_idx = 0
        for i in range(cot_trace_depth):
            for j in range(num_agents_per_layer):
                agent_id = f"agent_{i}_{j}"
                if problem_idx < len(sub_problems_list):
                    decomposed_problems_map[agent_id] = sub_problems_list[problem_idx]
                    problem_idx += 1
                else: # Fallback
                    decomposed_problems_map[agent_id] = user_prompt

        # --- Seed Verb Generation ---
        num_mbti_types = len(mbti_archetypes)
        total_verbs_to_generate = word_vector_size * num_mbti_types
        seed_generation_chain = get_seed_generation_chain(llm)
        generated_verbs_str = await seed_generation_chain.ainvoke({"problem": user_prompt, "word_count": total_verbs_to_generate})
        all_verbs = list(set(generated_verbs_str.split()))
        random.shuffle(all_verbs)
        seeds = {mbti: " ".join(random.sample(all_verbs, word_vector_size)) for mbti in mbti_archetypes}
        await log_stream.put(f"Seed verbs generated: {seeds}")

        # --- Agent Prompt Generation ---
        all_layers_prompts = []
        input_spanner_chain = get_input_spanner_chain(llm, params['prompt_alignment'], params['density'])
        
        # Layer 0
        await log_stream.put("--- Creating Layer 0 Agents ---")
        layer_0_prompts = []
        for j, (m, gw) in enumerate(seeds.items()):
            agent_id = f"agent_0_{j}"
            sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
            prompt = await input_spanner_chain.ainvoke({"mbti_type": m, "guiding_words": gw, "sub_problem": sub_problem})
            layer_0_prompts.append(prompt)
        all_layers_prompts.append(layer_0_prompts)
        
        # Subsequent Layers
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])

        for i in range(1, cot_trace_depth):
            await log_stream.put(f"--- Creating Layer {i} Agents ---")
            prev_layer_prompts = all_layers_prompts[i-1]
            current_layer_prompts = []
            for j, agent_prompt in enumerate(prev_layer_prompts):
                analysis_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
                try:
                    analysis = clean_and_parse_json(analysis_str)
                except (json.JSONDecodeError, AttributeError):
                    analysis = {"attributes": "", "hard_request": "Solve the original problem."}
                
                agent_id = f"agent_{i}_{j}"
                sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
                new_prompt = await dense_spanner_chain.ainvoke({
                    "attributes": analysis.get("attributes"),
                    "hard_request": analysis.get("hard_request"),
                    "critique": "",
                    "sub_problem": sub_problem,
                })
                current_layer_prompts.append(new_prompt)
            all_layers_prompts.append(current_layer_prompts)
        
        # --- Graph Definition ---
        workflow = StateGraph(GraphState)

        # Gateway node that increments the epoch
        def epoch_gateway(state: GraphState):
            new_epoch = state.get("epoch", 0) + 1
            state['epoch'] = new_epoch
            # Clear outputs from previous epoch before starting the new forward pass
            state['agent_outputs'] = {}
            return state
            
        workflow.add_node("epoch_gateway", epoch_gateway)

        for i, layer_prompts in enumerate(all_layers_prompts):
            for j, prompt in enumerate(layer_prompts):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_agent_node(llm, prompt, node_id))
        
        workflow.add_node("synthesis", create_synthesis_node(llm))
        workflow.add_node("metrics", create_metrics_node(llm))
        workflow.add_node("progress_assessor", create_progress_assessor_node(llm)) # New Node
        workflow.add_node("reframe_and_decompose", create_reframe_and_decompose_node(llm)) # New Node
        workflow.add_node("critique", create_critique_node(llm))
        workflow.add_node("update_prompts", create_update_agent_prompts_node(llm))

        # Add a final node to capture the state before ending
        def package_final_state(state: GraphState):
            """This node is a clean exit point. It captures the state from the final
            forward pass before the graph terminates."""
            return state
        workflow.add_node("package_results", package_final_state)


        # --- Graph Connections ---
        await log_stream.put("--- Connecting Graph Nodes ---")
        
        workflow.set_entry_point("epoch_gateway")
        await log_stream.put("LOG: Entry point set to 'epoch_gateway'.")
        
        first_layer_nodes = [f"agent_0_{j}" for j in range(len(all_layers_prompts[0]))]
        for node in first_layer_nodes:
            workflow.add_edge("epoch_gateway", node)
            await log_stream.put(f"CONNECT: epoch_gateway -> {node}")

        for i in range(cot_trace_depth - 1):
            current_layer_nodes = [f"agent_{i}_{j}" for j in range(len(all_layers_prompts[i]))]
            next_layer_nodes = [f"agent_{i+1}_{k}" for k in range(len(all_layers_prompts[i+1]))]
            for current_node in current_layer_nodes:
                for next_node in next_layer_nodes:
                    workflow.add_edge(current_node, next_node)
                    await log_stream.put(f"CONNECT: {current_node} -> {next_node}")
        
        last_layer_idx = cot_trace_depth - 1
        last_layer_nodes = [f"agent_{last_layer_idx}_{j}" for j in range(len(all_layers_prompts[last_layer_idx]))]
        for node in last_layer_nodes:
            workflow.add_edge(node, "synthesis")
            await log_stream.put(f"CONNECT: {node} -> synthesis")

        # MODIFIED: New conditional logic for progress assessment
        async def assess_progress_and_decide_path(state: GraphState):
            if state["epoch"] >= state["max_epochs"]:
                await log_stream.put(f"LOG: Final epoch ({state['epoch']}) finished. Capturing final state and ending execution.")
                return "package_results"
            
            if state.get("significant_progress_made"):
                await log_stream.put(f"LOG: Epoch {state['epoch']} shows significant progress. Re-framing the problem.")
                return "reframe"
            else:
                await log_stream.put(f"LOG: Epoch {state['epoch']} shows no significant progress. Proceeding with standard critique.")
                return "critique"

        workflow.add_conditional_edges(
            "progress_assessor",
            assess_progress_and_decide_path,
            {
                "reframe": "reframe_and_decompose",
                "critique": "critique",
                "package_results": "package_results"
            }
        )
        await log_stream.put("CONNECT: progress_assessor -> assess_progress_and_decide_path (conditional)")

        workflow.add_edge("synthesis", "metrics")
        await log_stream.put("CONNECT: synthesis -> metrics")
        
        workflow.add_edge("metrics", "progress_assessor")
        await log_stream.put("CONNECT: metrics -> progress_assessor")

        workflow.add_edge("critique", "update_prompts")
        await log_stream.put("CONNECT: critique -> update_prompts")

        workflow.add_edge("reframe_and_decompose", "update_prompts")
        await log_stream.put("CONNECT: reframe_and_decompose -> update_prompts")

        workflow.add_edge("update_prompts", "epoch_gateway")
        await log_stream.put("CONNECT: update_prompts -> epoch_gateway (loop)")

        workflow.add_edge("package_results", END)
        await log_stream.put("CONNECT: package_results -> END")
        
        # --- Graph Execution ---
        graph = workflow.compile()
        await log_stream.put("Graph compiled successfully.") 
        
        ascii_art = graph.get_graph().draw_ascii()
        await log_stream.put(f"{ascii_art}")

        initial_state = {
            "original_request": user_prompt,
            "decomposed_problems": decomposed_problems_map,
            "layers": [], "critiques": {}, "epoch": 0,
            "max_epochs": int(params["num_epochs"]),
            "params": params, "all_layers_prompts": all_layers_prompts,
            "agent_outputs": {}, "memory": {}, "final_solution": None,
            "perplexity_history": [],
            "significant_progress_made": False # New: Initialize flag
        }

        await log_stream.put(f"--- Starting Execution (Epochs: {params['num_epochs']}) ---")
        final_state = None

        async for output in graph.astream(initial_state, {'recursion_limit': int(params["num_epochs"]) * 1000}):
            for key, value in output.items():
                await log_stream.put(f"--- Node Finished Processing: {key} ---")
            final_state = output
        
        final_state_value = list(final_state.values())[0] if final_state else {}
        await log_stream.put("--- Execution Finished ---")
        
        final_solution = final_state_value.get("final_solution", {"error": "No final solution found in the final state."})

        # MODIFIED: Package hidden layer outputs with their final system prompts
        hidden_layer_outputs = {}
        final_prompts = final_state_value.get("all_layers_prompts", [])

        if "agent_outputs" in final_state_value and final_prompts:
            for agent_id, output in final_state_value["agent_outputs"].items():
                try:
                    parts = agent_id.split('_')
                    layer_index = int(parts[1])
                    agent_index_in_layer = int(parts[2])

                    # The last layer's output goes into synthesis, so we only show the ones before it.
                    if layer_index < (cot_trace_depth - 1):
                        system_prompt = final_prompts[layer_index][agent_index_in_layer]
                        hidden_layer_outputs[agent_id] = {
                            "output": output,
                            "system_prompt": system_prompt
                        }
                except (IndexError, ValueError) as e:
                    await log_stream.put(f"WARNING: Could not process hidden output for {agent_id}. Error: {e}")

        return JSONResponse(content={
            "message": "Graph execution complete.", 
            "final_solution": final_solution,
            "hidden_layer_outputs": hidden_layer_outputs
        })

    except Exception as e:
        error_message = f"An error occurred during graph execution: {e}"
        await log_stream.put(error_message)
        await log_stream.put(traceback.format_exc())
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)


@app.get('/stream_log')
def stream_log(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                log_message = await asyncio.wait_for(log_stream.get(), timeout=1.0)
                yield f"data: {log_message}\n\n"
            except asyncio.TimeoutError:
                continue

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)