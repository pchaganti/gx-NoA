import os
import uvicorn
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from typing import TypedDict, Annotated, List
import asyncio
from sse_starlette.sse import EventSourceResponse
import random

load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI()

# In-memory stream for logs
log_stream = asyncio.Queue()

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
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

def get_input_spanner_chain(prompt_alignment, density):
    prompt = ChatPromptTemplate.from_template(f"""
Create a generalist agent meant to collaborate in a team that will try to tackle any kind of problem that gets thrown at them, by mixing the creative attitudes and dispositions of an MBTI type and mix them with the guiding words attached. Think of it as creatively coming up with a new class for an RPG game, but without fantastical elements - define skills and attributes. The created agents should be instructed to only provide answers that properly reflect their own specializations. You will balance how much influence the guiding words have on the MBTI agent by modulating it using the parameter ‘density’ ({density}). You will also give the agent a professional career, which could be made up altought it must be realistic- the ‘career’ is going to be based off  the parameter “prompt_alignment” ({prompt_alignment}) . You will analyze the prompt and assign the career on the basis on how useful the profession would be to solve the problem posed by the parameter ‘prompt’. You will balance how much influence the prompt has on the career by modualting it with the paremeter prompt_alignment ({prompt_alignment})  Each generated agent must contain in markdown the sections: memory, attributes, skills. Memory is a log of your previous proposed solutions and reasonings from past epochs. You will use this to learn from your past attempts and refine your approach. Initially, your memory will be empty. Attributes and skills will be derived from the guiding words and the prompt alignment. Each agent should format its answer as a JSON with the following keys: “original_problem”: “”, “proposed_solution”: “”, “reasoning”: “”,  “skills_used”: “”.
MBTI Type: {{mbti_type}}
Guiding Words: {{guiding_words}}
Prompt: {{prompt}}
""")
    return prompt | llm | StrOutputParser()

def get_attribute_and_hard_request_generator_chain(vector_word_size):
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


def get_dense_spanner_chain(prompt_alignment, density, learning_rate):
    prompt = ChatPromptTemplate.from_template(f"""
You are a 'dense_spanner'. Your task is to create a new agent based on the attributes of a previous agent and a 'hard_request'.
This new agent will be part of a multi-layered graph of agents working to solve a problem.
The new agent's system prompt should be based on the identified attributes from the previous layer's agents and the 'hard_request'.
The influence of the hard_request on the new agent's career is modulated by 'prompt_alignment' ({prompt_alignment}).
The influence of the attributes on the new agent's skills is modulated by 'density' ({density}).
The 'learning_rate' ({learning_rate}) will determine how much the agent's attributes, career, and skills are modified based on the critique from the reflection pass.
The agent should have the same JSON output format as the input agents.
Attributes: {{attributes}}
Hard Request: {{hard_request}}
Critique: {{critique}}
""")
    return prompt | llm | StrOutputParser()

def get_synthesis_chain():
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

def get_critique_chain():
    prompt = ChatPromptTemplate.from_template("""
You are a critique agent. Your role is to assess the veracity, precision, usability, and truthfulness of a proposed solution from a single agent in relation to the original request.
Based on your assessment, you will generate a constructive critique that can be used to improve the agent's solution in the next iteration.

Original Request: {original_request}
Proposed Solution from Agent {agent_id}:
{proposed_solution}

Generate your critique for this specific agent:
""")
    return prompt | llm | StrOutputParser()

def get_seed_generation_chain():
    prompt = ChatPromptTemplate.from_template("""
Given the following problem, generate exactly {word_count} verbs that are related to the problem, but also connect the problem with different semantic fields of knowledge. The verbs should be abstract and linguistically loaded. Output them as a single space-separated string of unique verbs.

Problem: "{problem}"
""")
    return prompt | llm | StrOutputParser()


def create_agent_node(agent_prompt, node_id):
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

async def synthesis_node(state: GraphState):
    await log_stream.put("--- Entering Synthesis Node ---")
    synthesis_chain = get_synthesis_chain()

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


async def critique_node(state: GraphState):
    await log_stream.put("--- Entering Critique Node ---")
    critique_chain = get_critique_chain()
    
    last_agent_layer_idx = len(state['all_layers_prompts']) - 1
    num_agents_last_layer = len(state['all_layers_prompts'][last_agent_layer_idx])
    
    critiques = {}
    for i in range(num_agents_last_layer):
        node_id = f"agent_{last_agent_layer_idx}_{i}"
        if node_id in state["agent_outputs"]:
            agent_solution = state["agent_outputs"][node_id]
            critique_text = await critique_chain.ainvoke({
                "original_request": state["original_request"],
                "agent_id": node_id,
                "proposed_solution": json.dumps(agent_solution, indent=2)
            })
            critiques[node_id] = critique_text
            await log_stream.put(f"Critique generated for {node_id}: {critique_text[:100]}...")
            
    return {"critiques": critiques}


async def update_agent_prompts_node(state: GraphState):
    await log_stream.put("--- Entering Agent Prompt Update Node (Reflection) ---")
    params = state["params"]
    critiques = state["critiques"]
    num_reflections = int(params.get('num_reflections', 1))

    if not critiques:
        await log_stream.put("No critiques available. Skipping reflection.")
        new_epoch = state["epoch"] + 1
        return {"epoch": new_epoch, "agent_outputs": {}, "memory": state.get("memory", {})} # Preserve memory

    all_prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
    
    dense_spanner_chain = get_dense_spanner_chain(params['prompt_alignment'], params['density'], params['learning_rate'])
    attribute_chain = get_attribute_and_hard_request_generator_chain(params['vector_word_size'])

    last_layer_idx = len(all_prompts_copy) - 1
    
    # The critiques are for the last layer agents, which were generated by the penultimate layer agents.
    # So, we update the penultimate layer prompts based on the critiques of the last layer.
    penultimate_layer_idx = last_layer_idx - 1

    if penultimate_layer_idx < 0:
        await log_stream.put("Not enough layers to reflect upon. Skipping.")
        new_epoch = state["epoch"] + 1
        return {"epoch": new_epoch, "agent_outputs": {}, "memory": state.get("memory", {})} # Preserve memory

    # --- Reflection on Penultimate Layer ---
    await log_stream.put(f"--- Reflecting on Layer {penultimate_layer_idx} ---")
    prompts_to_update = all_prompts_copy[penultimate_layer_idx]
    
    # Determine which agent prompts to update based on num_reflections
    num_agents_in_layer = len(prompts_to_update)
    agent_indices_to_update = random.sample(range(num_agents_in_layer), min(num_reflections, num_agents_in_layer))
    
    await log_stream.put(f"Selected {len(agent_indices_to_update)} out of {num_agents_in_layer} agents in layer {penultimate_layer_idx} for reflection.")

    critiques_for_next_layer = {} # agent_index -> new_hard_request

    for i in agent_indices_to_update:
        agent_prompt = prompts_to_update[i]
        critique_key = f"agent_{last_layer_idx}_{i}"
        
        if critique_key not in critiques:
            await log_stream.put(f"Warning: No critique found for {critique_key}. Skipping update for agent {penultimate_layer_idx}_{i}.")
            continue

        critique = critiques[critique_key]
        await log_stream.put(f"Updating prompt for agent {penultimate_layer_idx}_{i} with critique from {critique_key}...")
        
        analysis_result_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
        try:
            analysis_result = json.loads(analysis_result_str)
            attributes = analysis_result.get("attributes", "")
        except json.JSONDecodeError:
            await log_stream.put(f"Could not decode attributes for agent {i}. Using empty attributes.")
            attributes = ""

        new_agent_prompt = await dense_spanner_chain.ainvoke({
            "attributes": attributes,
            "hard_request": "Based on the following critique, refine your approach and capabilities.",
            "critique": critique
        })
        
        all_prompts_copy[penultimate_layer_idx][i] = new_agent_prompt
        await log_stream.put(f"Prompt for agent {penultimate_layer_idx}_{i} updated.")

        # Generate a new "hard_request" from the newly created prompt to be used as critique for the previous layer.
        new_analysis_str = await attribute_chain.ainvoke({"agent_prompt": new_agent_prompt})
        try:
            new_analysis = json.loads(new_analysis_str)
            new_hard_request = new_analysis.get("hard_request", "")
            if new_hard_request:
                critiques_for_next_layer[i] = new_hard_request
        except json.JSONDecodeError:
            await log_stream.put(f"Could not decode new hard_request for updated agent {penultimate_layer_idx}_{i}.")

    # --- Backward Propagation to Earlier Layers ---
    for layer_idx in range(penultimate_layer_idx - 1, -1, -1):
        if not critiques_for_next_layer:
            await log_stream.put(f"No new critiques generated from layer {layer_idx + 1}. Stopping backward propagation.")
            break
        
        await log_stream.put(f"--- Reflecting on Layer {layer_idx} ---")
        
        prompts_to_update = all_prompts_copy[layer_idx]
        # The critiques are now sparse, keyed by the agent index from the layer above.
        agent_indices_to_update = list(critiques_for_next_layer.keys())
        
        new_critiques_for_next_layer = {}

        for i in agent_indices_to_update:
            if i >= len(prompts_to_update):
                await log_stream.put(f"Warning: Index {i} from previous layer's reflection is out of bounds for layer {layer_idx}. Skipping.")
                continue

            agent_prompt = prompts_to_update[i]
            critique = critiques_for_next_layer[i]
            
            await log_stream.put(f"Updating prompt for agent {layer_idx}_{i} with hard_request from updated agent {layer_idx+1}_{i}...")
            
            analysis_result_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
            try:
                analysis_result = json.loads(analysis_result_str)
                attributes = analysis_result.get("attributes", "")
            except json.JSONDecodeError:
                attributes = ""

            new_agent_prompt = await dense_spanner_chain.ainvoke({
                "attributes": attributes,
                "hard_request": "Based on the following critique, refine your approach and capabilities.",
                "critique": critique
            })
            
            all_prompts_copy[layer_idx][i] = new_agent_prompt
            await log_stream.put(f"Prompt for agent {layer_idx}_{i} updated.")

            new_analysis_str = await attribute_chain.ainvoke({"agent_prompt": new_agent_prompt})
            try:
                new_analysis = json.loads(new_analysis_str)
                new_hard_request = new_analysis.get("hard_request", "")
                if new_hard_request:
                    new_critiques_for_next_layer[i] = new_hard_request
            except json.JSONDecodeError:
                pass
        
        critiques_for_next_layer = new_critiques_for_next_layer

    new_epoch = state["epoch"] + 1
    await log_stream.put(f"--- Epoch {state['epoch']} Finished. Starting Epoch {new_epoch} ---")

    return {
        "all_layers_prompts": all_prompts_copy,
        "epoch": new_epoch,
        "agent_outputs": {}, # Reset outputs for the next epoch
        "memory": state.get("memory", {}) # Preserve memory across epochs
    }

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/build_and_run_graph")
async def build_and_run_graph(payload: dict = Body(...)):
    if not llm:
        return JSONResponse(content={"message": "LLM not initialized. Please check your API key."}, status_code=500)

    params = payload.get("params")
    user_prompt = params.get("prompt")
    word_vector_size = int(params.get("vector_word_size"))
    cot_trace_depth = int(params.get('cot_trace_depth', 4))
    num_agents_per_principality = 1

    await log_stream.put("--- Starting Graph Build and Run Process ---")
    await log_stream.put(f"Parameters: {params}")

    # --- Seed Verb Generation ---
    mbti_types = [
        "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    num_mbti_types = len(mbti_types)
    total_verbs_to_generate = word_vector_size * num_mbti_types

    await log_stream.put(f"--- Generating {total_verbs_to_generate} Seed Verbs ---")
    seed_generation_chain = get_seed_generation_chain()
    try:
        generated_verbs_str = await seed_generation_chain.ainvoke({
            "problem": user_prompt,
            "word_count": total_verbs_to_generate
        })
        all_verbs = list(set(generated_verbs_str.split())) # Ensure uniqueness
        
        if len(all_verbs) < total_verbs_to_generate:
            await log_stream.put(f"Warning: LLM generated {len(all_verbs)} unique verbs, required {total_verbs_to_generate}. Padding with repeats.")
            repeats_needed = total_verbs_to_generate - len(all_verbs)
            all_verbs.extend((all_verbs * (repeats_needed // len(all_verbs) + 1))[:repeats_needed])
        
        all_verbs = all_verbs[:total_verbs_to_generate] # Trim excess
        random.shuffle(all_verbs) # Shuffle to distribute diverse verbs

        seeds = {}
        for i, mbti_type in enumerate(mbti_types):
            start_index = i * word_vector_size
            end_index = start_index + word_vector_size
            seeds[mbti_type] = " ".join(all_verbs[start_index:end_index])
        
        await log_stream.put("Seed verbs generated successfully.")
    except Exception as e:
        await log_stream.put(f"Error generating seed verbs: {e}. Aborting.")
        return JSONResponse(content={"message": f"Failed to generate seed verbs: {e}"}, status_code=500)


    # --- Agent Prompt Generation ---
    all_layers_prompts = []
    input_spanner_chain = get_input_spanner_chain(params['prompt_alignment'], params['density'])
    
    # Layer 0 (Input Layer)
    await log_stream.put("--- Creating Layer 0 Agents ---")
    layer_0_prompts = []
    for mbti_type, guiding_words in seeds.items():
        for _ in range(num_agents_per_principality):
            agent_prompt = await input_spanner_chain.ainvoke({"mbti_type": mbti_type, "guiding_words": guiding_words, "prompt": user_prompt})
            layer_0_prompts.append(agent_prompt)
    all_layers_prompts.append(layer_0_prompts)
    await log_stream.put(f"Created {len(layer_0_prompts)} agents for Layer 0.")

    # Subsequent Dense Layers
    attribute_chain = get_attribute_and_hard_request_generator_chain(params['vector_word_size'])
    dense_spanner_chain = get_dense_spanner_chain(params['prompt_alignment'], params['density'], params['learning_rate'])

    for i in range(1, cot_trace_depth):
        await log_stream.put(f"--- Creating Layer {i} Agents ---")
        prev_layer_prompts = all_layers_prompts[i-1]
        current_layer_prompts = []
        for agent_prompt in prev_layer_prompts:
            analysis_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
            try:
                analysis = json.loads(analysis_str)
                attributes = analysis.get("attributes")
                hard_request = analysis.get("hard_request")
            except json.JSONDecodeError:
                await log_stream.put(f"Could not decode attributes/request. Using defaults.")
                attributes, hard_request = "", "Solve the problem."

            new_agent_prompt = await dense_spanner_chain.ainvoke({
                "attributes": attributes,
                "hard_request": hard_request,
                "critique": "" # No critique during initial build
            })
            current_layer_prompts.append(new_agent_prompt)
        all_layers_prompts.append(current_layer_prompts)
        await log_stream.put(f"Created {len(current_layer_prompts)} agents for Layer {i}.")

    # --- Graph Definition ---
    workflow = StateGraph(GraphState)

    # Add agent nodes for each layer
    for i, layer_prompts in enumerate(all_layers_prompts):
        for j, _ in enumerate(layer_prompts):
            node_id = f"agent_{i}_{j}"
            workflow.add_node(node_id, create_agent_node(all_layers_prompts[i][j], node_id))

    # Add synthesis, critique, and update nodes
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("update_prompts", update_agent_prompts_node)

    # --- Graph Connections ---
    # Set the entry point to the first agent of the first layer
    workflow.set_entry_point("agent_0_0")

    # Forward connections (fully connected)
    for i in range(cot_trace_depth - 1):
        for j in range(len(all_layers_prompts[i])):
            # Since all nodes in the next layer depend on all nodes from this layer,
            # we connect each node to the *first* node of the next layer.
            # The logic within the nodes handles the aggregation.
            workflow.add_edge(f"agent_{i}_{j}", f"agent_{i+1}_0")
    
    # Connect all nodes in the last layer to the synthesis node
    last_layer_idx = cot_trace_depth - 1
    for j in range(len(all_layers_prompts[last_layer_idx])):
        workflow.add_edge(f"agent_{last_layer_idx}_{j}", "synthesis")


    # Connect synthesis to critique
    workflow.add_edge("synthesis", "critique")
    
    # Connect critique to the prompt update node
    workflow.add_edge("critique", "update_prompts")

    # Conditional edge for epochs
    async def should_continue(state: GraphState):
        if state["epoch"] >= state["max_epochs"]:
            await log_stream.put("--- Max Epochs Reached. Ending Run. ---")
            return END
        else:
            # After updating prompts, we start the next epoch from the beginning.
            return "agent_0_0"

    workflow.add_conditional_edges("update_prompts", should_continue)
    
    # --- Graph Execution ---
    try:
        graph = workflow.compile()
        await log_stream.put("Graph compiled successfully.")
        
        initial_state = {
            "original_request": user_prompt,
            "layers": [], 
            "critiques": {},
            "epoch": 0,
            "max_epochs": int(params["num_epochs"]),
            "params": params,
            "all_layers_prompts": all_layers_prompts,
            "agent_outputs": {},
            "memory": {},
            "final_solution": {}
        }

        await log_stream.put(f"--- Starting Execution (Epochs: {params['num_epochs']}) ---")
        # The stream method is good for observing the flow state by state
        async for output in graph.astream(initial_state):
            # output is a dictionary where keys are node names
            for key, value in output.items():
                await log_stream.put(f"--- Node: {key} ---")
                # Avoid printing the entire memory state every time
                if "memory" in value:
                    await log_stream.put(f"Output for {key} processed. Memory updated.")
                else:
                    await log_stream.put(f"Output: {value}")

        
        # After the stream is done, the final state is in the last output
        final_state = list(output.values())[-1]
        await log_stream.put("--- Execution Finished ---")
        
        final_solution = final_state.get("final_solution", {"error": "No final solution found."})

        mermaid_graph = f"""
        <pre class="mermaid">
            {graph.get_graph().draw_mermaid()}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
        
        return JSONResponse(content={
            "message": "Graph execution complete.", 
            "final_solution": final_solution,
            "mermaid_graph": mermaid_graph,
            "graph_structure": {
                "nodes": list(graph.nodes.keys()),
                "edges": [{"source": s, "target": t} for s, t in graph.edges]
            }
        })

    except Exception as e:
        error_message = f"An error occurred: {e}"
        await log_stream.put(error_message)
        import traceback
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