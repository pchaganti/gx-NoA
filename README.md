![thumbnail](https://github.com/user-attachments/assets/13694758-a5c9-40c5-9c07-c7a168e660cf)

# Network of Agents (NoA): Democratizing Deep Thought 🧠

I've been thinking a lot about how we, as people, develop ideas. It's rarely a single, brilliant flash of insight. Our minds are shaped by the countless small interactions we have throughout the day—a conversation here, an article there. This environment of constant, varied input seems just as important as the act of thinking itself.

I wanted to see if I could recreate a small-scale version of that "soup" required for true insight, for local LLMs. The result is this project, Network of Agents (NoA).

**Is true "deep thinking" only for trillion-dollar companies?**

NoA is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepThink offer powerful reasoning by giving their massive models more "thinking time" in a closed environment, NoA explores a different path: **emergent intelligence**. We simulate a society of AI agents—like individual neurons in a larger network—that collaborate, critique, and evolve their understanding collectively. The best part is that you don't need a supercomputer. NoA is designed to turn even a modest 32gb RAM laptop into a powerful "thought mining" rig. 💻⛏️

By leveraging efficient local models (like `dengcao/Qwen3-30B-A3B-Instruct-2507:latest`), you can leave the graph-network running for hours or even days, allowing it to iteratively refine its approach and "mine" a framework or solution to a multifactorial problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.

https://github.com/user-attachments/assets/009abf33-9083-4d6c-a5fa-3936bba48243

This project is an open invitation to explore the frontiers of collective AI, putting the power of deep, multi-day problem-solving into the hands of everyone. 🌍

## The Core Idea: Qualitative Backpropagation

Here’s the part that I'm most curious about. I was inspired by the concept of backpropagation in neural networks. It's a numerical algorithm, of course, but I wondered if the core principle could be applied qualitatively. What if, instead of sending back a numerical error signal, you sent back a "reflection"?

After the network produces a solution, a "critique" agent reviews it and provides criticism. This feedback is then used to **automatically re-write the core system prompts** of the agents that contributed. The goal is for the network to "learn" from its mistakes over multiple cycles, refining not just its answers, but its own internal structure and approach.

### The Trade-Off: Speed for Depth

The obvious trade-off here is speed. It’s the opposite of instantaneous. A 6-layer network with 6 agents per layer, running for 20 cycles, can easily take 12 hours to complete. You're trading quick computation for a slow, iterative process of refinement.

The algorithm does really well in problems where creativity and insight override pure precision. It can come up with new frameworks for the social sciences, for instance. Physics and math, not so much.

## The NoA Algorithm: From Individual Agents to a Collective Mind

The core of NoA is a novel algorithm that orchestrates LLM agents into a dynamic, layered network. This architecture facilitates a "forward pass" for problem-solving and a unique "reflection pass" for learning and goal evolution.

### The Forward Pass

In NoA, the "weights" and "biases" of the network are not numerical values but the rich, descriptive personas of its agents, defined in natural language.

1.  **Input Layer & Decomposition**: The process starts with a user's high-level problem. A `master strategist` node first **decomposes this problem into smaller, distinct sub-problems**. NoA then generates abstract "seed verbs" and assigns them to diverse MBTI personality archetypes. An `input-spanner` chain forges the first layer of agents, each with a unique persona, a specialized career, a distinct set of skills, and its own **assigned sub-problem**. This ensures both cognitive and functional diversity from the very beginning.

2.  **Building Depth with Dense Layers**: To create a truly deep and specialized network, a `dense-spanner` chain analyzes the agents of the preceding layer. It identifies their core attributes and formulates a "hard request"—a tailored challenge designed to push the boundaries of that agent's expertise. It then spawns a new agent in the next layer, specifically engineered to tackle this challenge. This process mirrors the creation of specialized hidden layers in a neural network, progressively increasing the system's intellectual depth.

3.  **Action**: A user's prompt initiates a cascade of information through the network. The input layer agents process their assigned sub-problems, and their structured JSON outputs are broadcast to every agent in the subsequent layer. This dense, layer-by-layer processing continues until the final layer is reached, constituting a full "forward pass" of collaborative inference.

### The Reflection Pass: Learning, Critiquing, and Evolving Goals

This is where NoA truly differs from a simple multi-agent system. The reflection pass is a multi-stage process where the network assesses its own performance and decides how to adapt.

1.  **Synthesis and Metrics**: After the forward pass, a `synthesis_node` merges the outputs from the final agent layer into a single, coherent solution. Immediately after, a `metrics_node` analyzes all agent outputs from the epoch to calculate a perplexity score, offering a quantitative glimpse into the network's state.

2.  **The Crossroads of Progress**: The synthesized solution is then passed to a `progress_assessor` node. This AI philosopher acts as a critical juncture, evaluating whether the solution represents "significant progress" based on novelty, coherence, and forward momentum. This decision dictates the entire course of the next epoch.

3.  **Path A: The "Eureka!" Path (Significant Progress)**: If a breakthrough is achieved, the network's goal shifts from refinement to advancement. A `problem_reframer` node formulates a **new, more challenging problem** that builds on the recent success. This new problem is then decomposed and assigned to the agents. In this path, direct critique is bypassed in favor of a new mission.

4.  **Path B: The "Refinement" Path (No Significant Progress)**: If the solution is not a major leap forward, the system focuses on iterative improvement. A `critique_chain` scrutinizes the solution and generates a global, constructive critique. It also generates targeted, individual critiques for each agent in the hidden layers, assessing their contribution relative to their sub-problem.

5.  **Updating the "Neural" Weights**: This feedback—either a new mission or a detailed critique—propagates backward through the network. An `update_agent_prompts_node` uses this signal to modify each agent's core system prompt, refining their skills, attributes, and roles. This is how the network learns and adapts—not through a central controller, but through a distributed process of assessment, challenge, and evolution.

This entire cycle is one "epoch." By running multiple epochs, the network engages in a process of collective sense-making that can now not only deepen its understanding but also evolve its own objectives.

## Changelog

*   **Dynamic Problem Re-framing:** This is the big one. The network now has a new ability to assess its own progress at the end of an epoch. If it decides it has made a "significant breakthrough," it doesn't just continue to refine the old solution. Instead, it **formulates a new, more advanced problem** that builds upon its success. It essentially gives itself a harder challenge, turning the process from simple refinement into a genuine journey of discovery.
*   **Divide and Conquer - Automatic Problem Decomposition:** Instead of every agent in the first layer tackling the same high-level prompt, NoA now starts by breaking down the user's initial problem into smaller, more granular sub-problems. Each agent is assigned a unique piece of the puzzle from the very beginning, ensuring a more focused and diverse approach right out of the gate.
*   **Keeping an Eye on Things - Perplexity Metrics & Chart:** We've added a new `metrics` node that calculates the average perplexity of all agent outputs after each epoch. We now plot this on a live chart in the GUI, giving you a visual heuristic for the network's coherence over time.
*   **Better Memory for the Long Haul - Dynamic Summarization:** To support those extra-long, multi-day mining sessions, we've implemented a memory summarizer. If an agent's memory log gets too long, a specialized chain now automatically creates a concise summary of its older memories.
*   **Opening the Black Box - Richer Data Collection:** We're now logging and displaying much more data from the final run, including the final evolved system prompts and JSON outputs for the hidden-layer agents.

## Vision & Long-Term Roadmap: Training a World Language Model

Beyond solving individual problems, every NoA run generates an incredibly valuable asset: a complete, structured trace of a multi-agent collaborative process. This isn't just a log file; it's a dataset capturing the evolution of a more socially driven kind of thought.

Our ultimate objective is to use this emergent data to train a true **"World Language Model" (WLM)**. A WLM, as we envision it, moves beyond predicting the next token. It would be a model trained on the fundamental patterns of collaboration, critique, and social intelligence. By training a foundation model on thousands of these solution-mining runs, we hypothesize it could develop a more robust, generalized reasoning capability.

## Research & Development Goals

### Mid-Term Research Goals
*   **Implement Cyclical Hierarchical Sparse Connections:** Explore a more sophisticated architecture to see the emergence of "leader" agents and specialized "micro-teams," improving scalability.
*   **Give Agents Tools:** Look into frameworks like open-source command-line tools to give each "neuron" an execution environment so it can code and perform actions, though this adds significant complexity.
*   **Scale to a "World-of-Agents"**: On more powerful hardware, use complex **"institutional directives"** as the foundational vectors, allowing the network to model and solve large-scale societal or organizational problems.
*   **Enhance Combinatorial Heuristics**: Research and implement more advanced heuristics to guide the LLM agents, improving their ability to reason symbolically.
*   **Develop More Sophisticated Metrics**: Implement more multi-faceted metrics to track the network's health, cognitive diversity, and the quality of its emergent solutions.

### Immediate Development Goals
*   **Peer-to-Peer Networking for Distributed Mining:** To truly democratize deep thought, we will add an optional P2P networking layer. This will allow multiple users to connect their NoA instances to collectively "mine" a solution.
*   **Conduct High-VRAM Scalability Tests:** Benchmark the system on a high-end machine with at least 128GB of VRAM to test the upper limits of the metaheuristic.

## Hyperparameters Explained: Tuning Your Solution Mining Rig ⚙️

*   **`CoT trace depth`**: The number of layers in your agent network.
*   **`Number of epochs`**: One full cycle of a forward pass (inference) and a reflection pass (learning).
*   **`Vector word size`**: The number of "seed verbs" for initial agent creation.
*   **`Prompt alignment` (0.1 - 2.0)**: Controls how strongly an agent's career is influenced by the user's prompt.
*   **`Density` (0.1 - 2.0)**: Modulates the influence of the previous layer when creating new agents.
*   **`Learning rate` (0.1 - 2.0)**: Controls the magnitude of change an agent makes to its prompt in response to critique.

## Technical Setup

The application is built with a Python backend and a vanilla HTML/CSS/JS frontend.

*   **Backend**: FastAPI, LangChain, LangGraph, Google Gemini / Ollama
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

4.  **Set up your environment variables:**
    *   Create a file named `.env` in the project root.
    *   If using Google Gemini, add your API key: `GEMINI_API_KEY="YOUR_API_KEY_HERE"`

5.  **Run the application:**
    ```bash
    uvicorn app:app --reload
    ```

6.  **Access the GUI:** Open your web browser and navigate to `http://127.0.0.1:8000`.

### Using Local Inference with Ollama

This application supports running inference locally using an Ollama server.

1.  **Install Ollama**: Follow the official instructions to install Ollama.
2.  **Download a Model**: Pull the desired model. The default is `dengcao/Qwen3-30B-A3B-Instruct-2507:latest`.
    ```bash
    ollama pull dengcao/Qwen3-30B-A3B-Instruct-2507:latest
    ```
3.  **Run Ollama Server**: Make sure the Ollama application is running in the background.
4.  **Select in GUI**: In the web interface, select "Local Ollama".

## How It Works

1.  **Choose your LLM**: Select between Google Gemini or Local Ollama.
2.  **Architect the Network**: Use the GUI to set the hyperparameters.
3.  **Pose a Problem**: Enter the high-level prompt you want the network to solve.
4.  **Build and Run**: Click "Build and Run Graph" to initiate the process.
5.  **Observe the Emergence**: Monitor the process in the real-time log viewer.
6.  **Receive the Synthesized Result**: Once all epochs are complete, the final answer is displayed.

## Let's Collaborate: Building a P2P Network

This is where I'd love to get some community input.

My long-term vision is to go beyond a single machine. I dream of building a P2P networking layer that would allow multiple users to connect their instances of the app. We could create a shared, distributed network where our machines could collaborate to tackle truly massive problems.

However, my background is in Python and AI, and I'm not an expert in distributed systems. **If you're someone who knows about peer-to-peer networking and this idea sounds at all interesting to you, I would genuinely love to hear from you and potentially collaborate.**

It’s an open-source experiment, and I’d be grateful for any thoughts, feedback, or ideas you might have.

Thanks.