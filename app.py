import os
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

load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI()

# In-memory stream for logs
log_stream = asyncio.Queue()

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
You must reply in the following JSON format: "original_problem": "", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps({
                "attributes": "mock debug fast",
                "hard_request": "Explain the meaning of life in one word."
            })
        elif "you are a 'dense_spanner'" in prompt:
             return f"""
You are a new mock agent created from a hard request.
### memory
- Empty.
### attributes
- refined, mock, debug
### skills
- Solving hard requests, placeholder generation.
You must reply in the following JSON format: "original_problem": "", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are a synthesis agent" in prompt:
            return json.dumps({
                "proposed_solution": "The final synthesized solution from the debug mode is 42.",
                "reasoning": "This answer was synthesized from multiple mock agent outputs during a debug run.",
                "skills_used": ["synthesis", "mocking", "debugging"]
            })
        elif "you are a critique agent" in prompt:
            return "This is a constructive mock critique. The solution could be more detailed and less numeric."
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "run jump think create build test deploy strategize analyze synthesize critique reflect"
        else: # This is a regular agent node being invoked
            return json.dumps({
                "original_problem": "A problem statement provided by the user.",
                "proposed_solution": f"This is a mock solution from agent node #{random.randint(100,999)}.",
                "reasoning": "This response was generated instantly by the MockLLM in debug mode.",
                "skills_used": ["mocking", "debugging", f"skill_{random.randint(1,10)}"]
            })

class GraphState(TypedDict):
    original_request: str
    layers: List[dict]
    critiques: dict[str, str]  # Node ID -> Critique Text
    epoch: int
    max_epochs: int
    params: dict
    all_layers_prompts: List[List[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: dict # Node ID -> List of past JSON outputs
    final_solution: dict

# --- LangChain Setup ---
# LLM is now initialized within the endpoint based on user selection

def get_input_spanner_chain(llm, prompt_alignment, density):
    prompt = ChatPromptTemplate.from_template(f"""
Create the system prompt of an agent meant to collaborate in a team that will try to tackle the hardest problems known to mankind, by mixing the creative attitudes and dispositions of an MBTI type and mix them with the guiding words attached.        
When you write down the system prompt use phrasing that addresses the agent: "You are a ..., your skills are..., your attributes are..."
Think of it as creatively coming up with a new class for an RPG game, but without fantastical elements - define skills and attributes. The created agents should be instructed to only provide answers that properly reflect their own specializations. You will balance how much influence the guiding words have on the MBTI agent by modulating it using the parameter ‘density’ ({density}) Min 0.0, Max 2.0. You will also give the agent a professional career, which could be made up altought it must be realistic- the ‘career’ is going to be based off  the parameter “prompt_alignment” ({prompt_alignment}) Min 0.0, Max 2.0 . You will analyze the prompt and assign the career on the basis on how useful the profession would be to solve the problem posed by the parameter ‘prompt’. You will balance how much influence the prompt has on the career by modualting it with the paremeter prompt_alignment ({prompt_alignment}) Min 0.0, Max 2.0  Each generated agent must contain in markdown the sections: memory, attributes, skills. 
Memory section in the system prompt is a log of your previous proposed solutions and reasonings from past epochs - it starts out as an empty markdown section for all agents created. You will use this to learn from your past attempts and refine your approach. 
Initially, the memory of the created agent in the system prompt will be empty. Attributes and skills will be derived from the guiding words and the prompt alignment. 
Each agent you generate should have an answer format in its system prompt with the following keys: “original_problem”: “”, “proposed_solution”: “”, “reasoning”: “”,  “skills_used”: “”.
MBTI Type: {{mbti_type}}
Guiding Words: {{guiding_words}}
Prompt: {{prompt}}
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


def get_dense_spanner_chain(llm, prompt_alignment, density, learning_rate):
    prompt = ChatPromptTemplate.from_template(f"""
Your task is to create the system prompt of new agent based on the attributes of a previous agent and a 'hard_request'.
This new agent will be part of a multi-layered graph of antropocentric agents working together to solve a problem.
When you write down the system prompt use phrasing that addresses the agent: "You are a ..., your skills are..., your attributes are..."
The new agent's personality, identity and skills as defined in the system prompt should be based on the identified attributes from the previous layer's agents and the 'hard_request'.
The influence of the hard_request on the new agent's career is modulated by 'prompt_alignment' ({prompt_alignment}). Min 0.0, Max 2.0
Each generated system prompt for an agent must contain in markdown the sections: memory, attributes, skills. 
Think of it as creatively coming up with a new class for an RPG game, but without fantastical elements - define skills and attributes. The created agents should be instructed to only provide answers that properly reflect their own specializations. 
Memory section in the system prompt is a log of your previous proposed solutions and reasonings from past epochs - it starts out as an empty markdown section for all agents created. You will use this to learn from your past attempts and refine your approach. 
Initially, the memory of the created agent in the system prompt will be empty. Attributes and skills will be derived from the guiding words while the career overriding the agent disposition is modulated by prompt alignment. 
The influence of the attributes on the new agent's skills is modulated by 'density' ({density}). Min 0.0, Max 2.0
The 'learning_rate' ({learning_rate}) (Min 0.0, Max 2.0) will determine how much the agent's attributes, career, and skills are modified based on the critique from the reflection pass.
Each agent you generate should have an answer format in its system prompt with the following keys: “original_problem”: “”, “proposed_solution”: “”, “reasoning”: “”,  “skills_used”: “”.
Attributes: {{attributes}}
Hard Request: {{hard_request}}
Critique: {{critique}}
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

def get_critique_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a senior emeritus manager with a vast ammount of knowledge, wisdom and experience in a team of agents tasked with solving the most challenging problems in the wolrd. Your role is to assess the quality of a proposed solution from a single agent in relation to the original request and brainstorm all the posbile ways in which the agents solution is incoherent.
Based on your assessment, you will list the posible ways the solution could go wrong, and at the end you will close with a deep reflective question that attempts to schock the agent and steer into change. 
Original Request: {original_request}
Proposed Solution from Agent {agent_id}:
{proposed_solution}

Generate your critique for this specific agent:
""")
    return prompt | llm | StrOutputParser()

def get_seed_generation_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
Given the following problem, generate exactly {word_count} verbs that are related to the problem, but also connect the problem with different semantic fields of knowledge. The verbs should be abstract and linguistically loaded. Output them as a single space-separated string of unique verbs.

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
        await log_stream.put(f"--- Invoking Agent: {node_id} ---")

        # Determine the input for the agent based on its layer
        layer_index = int(node_id.split('_')[1])
        
        if layer_index == 0:
            # First layer agents receive the original user request
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
            
            input_data = json.dumps(prev_layer_outputs, indent=2)

        # Retrieve the agent's memory from previous epochs
        agent_memory_history = state.get("memory", {}).get(node_id, [])
        memory_str = "\n".join([f"- Epoch {i}: {json.dumps(mem)}" for i, mem in enumerate(agent_memory_history)])

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
            response_json = json.loads(response_str)
            await log_stream.put(f"Agent {node_id} finished. Solution snippet: {str(response_json.get('proposed_solution'))[:80]}...")
        except json.JSONDecodeError:
            await log_stream.put(f"Error: Agent {node_id} produced invalid JSON. Raw output: {response_str}")
            response_json = {
                "original_problem": state["original_request"],
                "proposed_solution": "Error: Agent produced malformed JSON output.",
                "reasoning": f"Invalid JSON: {response_str}",
                "skills_used": []
            }
            
        # Update the memory for this node
        current_memory = state.get("memory", {}).copy()
        if node_id not in current_memory:
            current_memory[node_id] = []
        # Append the new output to this agent's memory log
        current_memory[node_id].append(response_json)

        return {
            "agent_outputs": {node_id: response_json},
            "memory": current_memory
        }

    return agent_node

def create_synthesis_node(llm):
    async def synthesis_node(state: GraphState):
        await log_stream.put("--- Entering Synthesis Node ---")
        synthesis_chain = get_synthesis_chain(llm)

        # The synthesis node is connected to the last layer of agents.
        last_agent_layer_idx = len(state['all_layers_prompts']) - 1
        num_agents_last_layer = len(state['all_layers_prompts'][last_agent_layer_idx])
        
        last_layer_outputs = []
        for i in range(num_agents_last_layer):
            node_id = f"agent_{last_agent_layer_idx}_{i}"
            if node_id in state["agent_outputs"]:
                last_layer_outputs.append(state["agent_outputs"][node_id])

        await log_stream.put(f"Synthesizing {len(last_layer_outputs)} outputs from layer {last_agent_layer_idx}.")

        if not last_layer_outputs:
            await log_stream.put("Warning: Synthesis node received no inputs.")
            return {"final_solution": {"error": "Synthesis node received no inputs."}}

        final_solution_str = await synthesis_chain.ainvoke({
            "original_request": state["original_request"],
            "agent_solutions": json.dumps(last_layer_outputs, indent=2)
        })
        
        try:
            final_solution = json.loads(final_solution_str)
            await log_stream.put(f"Synthesis complete. Final solution snippet: {str(final_solution.get('proposed_solution'))[:80]}...")
        except json.JSONDecodeError:
            await log_stream.put(f"Error: Could not decode JSON from synthesis chain. Result: {final_solution_str}")
            final_solution = {"error": "Failed to synthesize final solution.", "raw": final_solution_str}
            
        return {"final_solution": final_solution}
    return synthesis_node

def create_critique_node(llm):
    async def critique_node(state: GraphState):
        await log_stream.put("--- Entering Critique Node ---")
        critique_chain = get_critique_chain(llm)
        
        last_agent_layer_idx = len(state['all_layers_prompts']) - 1
        num_agents_last_layer = len(state['all_layers_prompts'][last_agent_layer_idx])
        
        critiques = {}
        tasks = []
        for i in range(num_agents_last_layer):
            node_id = f"agent_{last_agent_layer_idx}_{i}"
            if node_id in state["agent_outputs"]:
                agent_solution = state["agent_outputs"][node_id]
                
                async def get_critique(node_id, agent_solution):
                    critique_text = await critique_chain.ainvoke({
                        "original_request": state["original_request"],
                        "agent_id": node_id,
                        "proposed_solution": json.dumps(agent_solution, indent=2)
                    })
                    critiques[node_id] = critique_text
                    await log_stream.put(f"Critique generated for {node_id}: {critique_text[:100]}...")

                tasks.append(get_critique(node_id, agent_solution))
        
        await asyncio.gather(*tasks)

        return {"critiques": critiques}
    return critique_node

def create_update_agent_prompts_node(llm):
    async def update_agent_prompts_node(state: GraphState):
        await log_stream.put("--- Entering Agent Prompt Update Node (Reflection) ---")
        params = state["params"]
        critiques = state["critiques"]
        num_reflections = int(params.get('num_reflections', 1))

        if not critiques:
            await log_stream.put("No critiques available. Skipping reflection.")
            new_epoch = state["epoch"] + 1
            return {
                "epoch": new_epoch,
                "agent_outputs": {},
                "memory": state.get("memory", {}),
                "final_solution": state.get("final_solution") # Preserve solution
            }

        all_prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
        
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])

        last_layer_idx = len(all_prompts_copy) - 1
        penultimate_layer_idx = last_layer_idx - 1

        if penultimate_layer_idx < 0:
            await log_stream.put("Not enough layers to reflect upon. Skipping.")
            new_epoch = state["epoch"] + 1
            return {
                "epoch": new_epoch,
                "agent_outputs": {},
                "memory": state.get("memory", {}),
                "final_solution": state.get("final_solution") # Preserve solution
            }

        # --- Reflection Logic ---
        # This part of the logic remains the same
        
        new_epoch = state["epoch"] + 1
        await log_stream.put(f"--- Epoch {state['epoch']} Finished. Starting Epoch {new_epoch} ---")

        # *** BUG FIX: Preserve final_solution across epochs ***
        return {
            "all_layers_prompts": all_prompts_copy,
            "epoch": new_epoch,
            "agent_outputs": {},
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
        layer_0_prompts = [await input_spanner_chain.ainvoke({"mbti_type": m, "guiding_words": gw, "prompt": user_prompt}) for m, gw in seeds.items()]
        all_layers_prompts.append(layer_0_prompts)
        
        # Subsequent Layers
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])

        for i in range(1, cot_trace_depth):
            await log_stream.put(f"--- Creating Layer {i} Agents ---")
            prev_layer_prompts = all_layers_prompts[i-1]
            current_layer_prompts = []
            for agent_prompt in prev_layer_prompts:
                analysis_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
                try:
                    analysis = json.loads(analysis_str)
                except json.JSONDecodeError:
                    analysis = {"attributes": "", "hard_request": "Solve the original problem."}
                
                new_prompt = await dense_spanner_chain.ainvoke({
                    "attributes": analysis.get("attributes"),
                    "hard_request": analysis.get("hard_request"),
                    "critique": ""
                })
                current_layer_prompts.append(new_prompt)
            all_layers_prompts.append(current_layer_prompts)
        
        # --- Graph Definition ---
        workflow = StateGraph(GraphState)

        for i, layer_prompts in enumerate(all_layers_prompts):
            for j, prompt in enumerate(layer_prompts):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_agent_node(llm, prompt, node_id))
        
        workflow.add_node("synthesis", create_synthesis_node(llm))
        workflow.add_node("critique", create_critique_node(llm))
        workflow.add_node("update_prompts", create_update_agent_prompts_node(llm))

        # --- Graph Connections ---
        workflow.set_entry_point(f"agent_0_0")
        for j in range(len(all_layers_prompts[0])):
            workflow.add_edge(f"agent_0_{j}", f"agent_1_0")

        for i in range(1, cot_trace_depth - 1):
            for j in range(len(all_layers_prompts[i])):
                workflow.add_edge(f"agent_{i}_{j}", f"agent_{i+1}_0")
        
        last_layer_idx = cot_trace_depth - 1
        for j in range(len(all_layers_prompts[last_layer_idx])):
            workflow.add_edge(f"agent_{last_layer_idx}_{j}", "synthesis")

        workflow.add_edge("synthesis", "critique")
        workflow.add_edge("critique", "update_prompts")
        
        async def should_continue(state: GraphState):
            if state["epoch"] >= state["max_epochs"] - 1:
                return END
            return "agent_0_0"
        
        workflow.add_conditional_edges("update_prompts", should_continue, {
            "agent_0_0": "agent_0_0",
            END: END
        })
        
        # --- Graph Execution ---
        graph = workflow.compile()
        await log_stream.put("Graph compiled successfully.")
        
        # --- Generate and send ASCII graph ---
        ascii_art = graph.get_graph().draw_ascii()
        await log_stream.put(json.dumps({"type": "graph_ascii", "payload": ascii_art}))

        initial_state = {
            "original_request": user_prompt,
            "layers": [], "critiques": {}, "epoch": 0,
            "max_epochs": int(params["num_epochs"]),
            "params": params, "all_layers_prompts": all_layers_prompts,
            "agent_outputs": {}, "memory": {}, "final_solution": None
        }

        await log_stream.put(f"--- Starting Execution (Epochs: {params['num_epochs']}) ---")
        final_state = None
        async for output in graph.astream(initial_state):
            for key, value in output.items():
                await log_stream.put(f"--- Node Finished: {key} ---")
            final_state = output
        
        final_state_value = list(final_state.values())[0] if final_state else {}
        await log_stream.put("--- Execution Finished ---")
        
        final_solution = final_state_value.get("final_solution", {"error": "No final solution found in the final state."})

        return JSONResponse(content={
            "message": "Graph execution complete.", 
            "final_solution": final_solution,
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