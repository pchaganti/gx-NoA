![thumbnail](https://github.com/user-attachments/assets/13694758-a5c9-40c5-9c07-c7a168e660cf)

# Network of Agents (NoA): Democratizing Deep Thought 🧠

I've been thinking a lot about how we, as people, develop ideas. It's rarely a single, brilliant flash of insight. Our minds are shaped by the countless small interactions we have throughout the day—a conversation here, an article there. This environment of constant, varied input seems just as important as the act of thinking itself.

I wanted to see if I could recreate a small-scale version of that "soup" required for true insight, for local LLMs. The result is this project, Network of Agents (NoA).

## ⚠️ Alpha Software - We Need Your Help! ⚠️
Please be aware that NoA is currently in an alpha stage. It is experimental research software, and you may encounter bugs, unexpected behavior, or breaking changes.

Your feedback is invaluable. If you run into issues, have ideas, or want to contribute, please **open an issue** on our GitHub repository. Helping us identify and squash bugs is one of the most important contributions you can make right now!

## **Is true "deep thinking" only for trillion-dollar companies?**
NoA is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepThink offer powerful reasoning by giving their massive models more "thinking time" in a closed environment, NoA explores a different path: **emergent intelligence**. We simulate a society of AI agents that collaborate, critique, and evolve their understanding collectively. The best part is that you don't need a supercomputer. NoA is designed to turn even a modest 32gb RAM laptop into a powerful "thought mining" rig. 💻⛏️ By leveraging efficient local models (like a quantized 30B-a3b parameter version of Qwen), you can leave the graph-network running for hours or even days, allowing it to iteratively refine its approach and "mine" a sophisticated solution to a hard problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.  

https://github.com/user-attachments/assets/009abf33-9083-4d6c-a5fa-3936bba48243


## Changelog

*   **Interactive RAG Chat & Diagnostic Tool:** The process now pauses after the final epoch, allowing you to directly **chat with the generated RAG index**. This powerful diagnostic feature lets you interrogate the massive "cube of thinking text" from all hidden layers, ask follow-up questions, and gain extra insights beyond the automated academic questions. Your entire chat conversation is then archived and included in the final knowledge harvest, enriching the final report.

*   **Final Knowledge Harvest & RAG:** The run doesn't just end with a final answer anymore. All agent outputs from all epochs are now archived into a multi-layered RAPTOR (Recursive Abstractive Processing over Trees of RAG) index. Upon completion, an `interrogator` agent generates a series of expert-level questions about the original problem, and a `paper_formatter` agent uses the RAG index to write a formal academic-style paper answering each question. The final output is now a downloadable ZIP file containing this collection of research papers.

*   **Dynamic Critique Annealing:** The network previously static "loss function" now evolves. A new meta-process analyzes the collective output of the hidden-layer agents after each epoch to determine their collective affinity. It then selects a "pseudo-neurotransmitter" that dynamically rewrites the system prompt of the critique agent, changing its persona (e.g., from a 'senior manager' to a 'cynical philosopher-king') to provide a different style of feedback for the next epoch.

*   **Dynamic Problem Re-framing:** The network can now assess its own progress. If it determines it has made a "significant breakthrough," it formulates a new, more advanced problem that builds upon its success. This turns the process from simple refinement into a genuine journey of discovery.

*   **Divide and Conquer - Automatic Problem Decomposition:** NoA now starts by breaking down the user's initial problem into smaller, granular sub-problems. Each agent is assigned a unique piece of the puzzle, ensuring a more focused and diverse approach from the start.

*   **Keeping an Eye on Things - Perplexity Metrics & Chart:** A new `metrics` node calculates the average perplexity of all agent outputs after each epoch. We plot this on a live chart in the GUI, giving you a visual heuristic for the network's coherence over time.

*   **Better Memory for the Long Haul - Dynamic Summarization:** To support extra-long mining sessions, a specialized chain now automatically creates a concise summary of an agent's older memories if its memory log gets too long, preserving key insights without overflowing the context window.

  

## The Core Idea: Qualitative Backpropagation

Here’s the part that I'm most curious about. I was inspired by the concept of backpropagation in neural networks. It's a numerical algorithm, of course, but I wondered if the core principle could be applied qualitatively. What if, instead of sending back a numerical error signal, you sent back a "reflection"?

After the network produces a solution, a "critique" agent reviews it and provides criticism. This feedback is then used to **automatically re-write the core system prompts** of the agents that contributed. The goal is for the network to "learn" from its mistakes over multiple cycles, refining not just its answers, but its own internal structure and approach.

### The Trade-Off: Speed for Depth

The obvious trade-off here is speed. It’s the opposite of instantaneous. A 6-layer network with 6 agents per layer, running for 20 cycles, can easily take 12 hours to complete. You're trading quick computation for a slow, iterative process of refinement.

The algorithm does really well in problems where creativity and insight override pure precision. It can come up with new frameworks for the social sciences, for instance. Physics and math, not so much.


## The NoA Algorithm: From Individual Agents to a Collective Mind

The core of NoA is a novel algorithm that orchestrates LLM agents into a dynamic, layered network. This architecture facilitates a "forward pass" for problem-solving, a unique "reflection pass" for learning, and a final "harvest pass" for knowledge extraction.

### The Forward Pass

In NoA, the "weights" and "biases" of the network are not numerical values but the rich, descriptive personas of its agents, defined in natural language.

1.  **Input Layer & Decomposition**: The process starts with a user's high-level problem. A `master strategist` node first **decomposes this problem into smaller, distinct sub-problems**. NoA then generates abstract "seed verbs" and assigns them to diverse MBTI personality archetypes. An `input-spanner` chain forges the first layer of agents, each with a unique persona, a specialized career, a distinct set of skills, and its own **assigned sub-problem**.

2.  **Building Depth with Dense Layers**: A `dense-spanner` chain analyzes the agents of the preceding layer, identifies their core attributes, and formulates a "hard request"—a tailored challenge. It then spawns a new agent in the next layer, specifically engineered to tackle this challenge, progressively increasing the system's intellectual depth.

3.  **Action**: A user's prompt initiates a cascade of information through the network. The input layer agents process their assigned sub-problems, and their structured JSON outputs are broadcast to every agent in the subsequent layer. This dense, layer-by-layer processing continues until the final layer is reached, constituting a full "forward pass" of collaborative inference.

### The Reflection Pass: Learning, Critiquing, and Evolving Goals

This is where NoA truly differs from a simple multi-agent system. The reflection pass is a multi-stage process where the network assesses its own performance and decides how to adapt.

1.  **Synthesis and Metrics**: After the forward pass, a `synthesis_node` merges the outputs from the final agent layer into a single, coherent solution. Immediately after, a `metrics_node` analyzes all agent outputs from the epoch to calculate a perplexity score.

2.  **Dynamic Annealing of Critique**: Before any critique is generated, an `update_critique_prompt` node analyzes the tone and content of the hidden-layer agents' outputs. It uses a unique heuristic to select a new persona (e.g., a wise mentor, a harsh drill sergeant) and dynamically rewrites the system prompt for the critique agent. This ensures the *style* of feedback adapts to the network's current state.

3.  **The Crossroads of Progress**: The synthesized solution is passed to a `progress_assessor` node. This AI philosopher evaluates whether the solution represents "significant progress" based on novelty, coherence, and forward momentum. This decision dictates the course of the next epoch.

4.  **Path A: The "Eureka!" Path (Significant Progress)**: If a breakthrough is achieved, the network's goal shifts to advancement. A `problem_reframer` node formulates a **new, more challenging problem** that builds on the recent success. This new problem is then decomposed and assigned to the agents.

5.  **Path B: The "Refinement" Path (No Significant Progress)**: If the solution is not a major leap forward, the system focuses on iterative improvement. The dynamically-configured `critique_chain` scrutinizes the solution and generates global and individual critiques.

6.  **Updating the "Neural" Weights**: This feedback—either a new mission or a detailed critique—propagates backward through the network. An `update_agent_prompts_node` uses this signal to modify each agent's core system prompt, refining their skills, attributes, and roles.

This entire cycle is one "epoch." By running multiple epochs, the network engages in a process of collective sense-making that can now not only deepen its understanding but also evolve its own objectives.

### The Final Harvest Pass: Consolidating Knowledge

When the final epoch is complete, the process is not over. The network enters a final phase to extract and structure the knowledge it has generated.

1.  **Archival and RAG Indexing**: An `archive_epoch_outputs` node gathers every single agent output from every epoch of the entire run. This collection of documents is then used to build a comprehensive, multi-layered RAPTOR RAG index, creating a searchable knowledge base of the entire thought process.

2.  **Pause for Interactive Chat**: At this point, the network pauses. The user is presented with a chat interface, allowing them to directly query the newly created RAG index. This serves as a powerful diagnostic tool, enabling the user to probe the network's collective "mind," ask clarifying questions, and explore threads of reasoning before the final summarization.

3.  **Interrogation and Synthesis**: When the user concludes the chat session, the entire conversation is converted into documents and added to the knowledge base, which is re-indexed. A `final_harvest` node then takes over. It uses an `interrogator` agent to generate a series of deep, expert-level questions about the original problem. For each question, it performs a retrieval query against the RAG index and feeds the context to a `paper_formatter` agent.

4.  **Generating the Final Report**: The `paper_formatter` synthesizes the retrieved context into a formal, academic-style markdown document. The final output of the entire run is a downloadable ZIP archive containing this collection of research papers, representing the network's total accumulated knowledge on the topic.

## Vision & Long-Term Roadmap: Training a World Language Model

Beyond solving individual problems, every NoA run generates an incredibly valuable asset: a complete, structured trace of a multi-agent collaborative process. This isn't just a log file; it's a dataset capturing the evolution of thought. It includes the initial agent personas, the layer-by-layer Chain-of-Thought (CoT) traces, the critiques, the diff of how each agent's core prompts evolved, and now, a complete RAG index of the entire process.

We conceptualize these collected traces as a new form of data: **powerful, multi-factorial, and multi-dimensional data for training next-generation reasoning models.** Unlike traditional datasets which capture static information, this data captures the *dynamics* of reasoning. It shows how diverse viewpoints clash, converge, and synthesize a solution.

Our ultimate objective is to use this emergent data to train a true **"World Language Model" (WLM)**.

A WLM, as we envision it, moves beyond predicting the next token. It would be a model trained on the fundamental patterns of collaboration, critique, and collective intelligence. It would learn the implicit "language" of how diverse agents interact to build something greater than the sum of their parts. By training a foundation model on thousands of these solution-mining runs, we hypothesize it could develop a more robust, generalized reasoning capability—one that intrinsically understands context, causality, and problem decomposition from a systemic perspective.

This represents the grand ambition of the NoA project: to not only solve hard problems but to create a data flywheel that can be used to forge a new paradigm of reasoning AI.

## Mid-Term Research Goals

These are the core research avenues we are actively exploring to enhance the network's capabilities.


*   **Enhance Combinatorial Heuristics**: The philosophical underpinning of NoA is that complex solutions emerge from the combinatorial power of language. The current "pseudo-neurotransmitter" system is a first step. We plan to research and implement more advanced heuristics to guide the LLM agents, improving their ability to reason symbolically and generate novel solutions.

*   **Develop More Sophisticated Metrics**: While perplexity provides a starting point, we plan to research and implement more multi-faceted metrics to track the network's health, cognitive diversity, and the quality of its emergent solutions over time.
*   **Peer-to-Peer Networking for Distributed Mining:** To truly democratize deep thought, we will add an optional P2P networking layer. This will allow multiple users to connect their NoA instances, distributing the agent computation across a network of machines to collectively "mine" a solution.


## Hyperparameters Explained: Tuning Your Theory Crafting Mining Rig ⚙️


The behavior of the NoA network is governed by several key hyperparameters, allowing you to architect the collaborative "mining" process.

*   **`CoT trace depth`**: The number of layers in your agent network. Deeper networks allow for more steps of abstraction.
*   **`Number of epochs`**: One full cycle of a forward pass (inference) and a reflection pass (learning). More epochs allow the network more time to "mine" a solution.
*   **`Vector word size`**: The number of "seed verbs" for initial agent creation. A larger size provides a richer starting point.
*   **`Number of Questions for Final Harvest`**: The number of expert-level questions the `interrogator` agent will generate to create the final report.
*   **`Prompt alignment` (0.1 - 2.0)**: Controls how strongly an agent's career is influenced by the user's prompt. Higher values lead to more specialization.
*   **`Density` (0.1 - 2.0)**: Modulates the influence of the previous layer when creating new agents. High density leads to refinement; low density encourages novelty.
*   **`Learning rate` (0.1 - 2.0)**: Controls the magnitude of change an agent makes to its prompt in response to critique.

## Technical Setup

The application is built with a Python backend and a vanilla HTML/CSS/JS frontend.

*   **Backend**: FastAPI, LangChain, LangGraph, Ollama
*   **Frontend**: HTML, CSS, JavaScript

### Installation and Execution

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd NoA
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:** `pip install -r requirements.txt`.

4.  **Install and Run Ollama**: This application requires a running Ollama server for local inference.
    *   Follow the official instructions to install Ollama.
    *   Download the primary model. The default is `dengcao/Qwen3-30B-A3B-Instruct-2507:latest`.
        ```bash
        ollama pull dengcao/Qwen3-30B-A3B-Instruct-2507:latest
        ```
    *   **Download the compulsory summarization model.** NoA requires `qwen3:1.7b` for its internal summarization tasks (memory, RAG indexing). This is not optional.
        ```bash
        ollama pull qwen3:1.7b
        ```
    *   Ensure the Ollama application is running in the background.

5.  **Run the application:**
    ```bash
    uvicorn app:app --reload
    ```

6.  **Access the GUI:** Open your web browser and navigate to `http://127.0.0.1:8000`.

## How It Works

1.  **Architect the Network**: Use the GUI to set the hyperparameters that define your network's structure and learning capacity.
2.  **Pose a Problem**: Enter the high-level prompt you want the agent network to solve.
3.  **Build and Run**: Click the "Build and Run Graph" button to initiate the process.
4.  **Observe the Emergence**: The backend dynamically constructs the agent network using LangGraph. You can monitor the entire process—agent creation, forward inference, and reflective learning—in the real-time log viewer.
5.  **Chat and Diagnose**: Once the epochs are complete, the GUI will present a chat interface. Use this to directly query the RAG index of the network's entire thought process. Ask follow-up questions and probe for deeper insights.
6.  **Harvest and Download**: When you are finished chatting, click the "HARVEST" button. This will incorporate your chat history and generate the final report. A download link for a ZIP file containing the research papers will appear.

## Let's Collaborate: Building a P2P Network

This is where I'd love to get some community input.

My long-term vision is to go beyond a single machine. I dream of building a P2P networking layer that would allow multiple users to connect their instances of the app. We could create a shared, distributed network where our machines could collaborate to tackle truly massive problems.

However, my background is in Python and AI, and I'm not an expert in distributed systems. **If you're someone who knows about peer-to-peer networking and this idea sounds at all interesting to you, I would genuinely love to hear from you and potentially collaborate.**

It’s an open-source experiment, and I’d be grateful for any thoughts, feedback, or ideas you might have.

Thanks.