![thumbnail](https://github.com/user-attachments/assets/13694758-a5c9-40c5-9c07-c7a168e660cf)

# local-deepthink: Democratizing Deep Thought 🧠

I've been thinking a lot about how we, as people, develop ideas. It's rarely a single, brilliant flash of insight. Our minds are shaped by the countless small interactions we have throughout the day—a conversation here, an article there. This environment of constant, varied input seems just as important as the act of thinking itself.

I wanted to see if I could recreate a small-scale version of that "soup" required for true insight for local LLMs. The result is this project, **local-deepthink**. It's a system that runs a novel conceptual algorithm called a **Qualitative Neural Network (QNN)**. In a QNN, different AI agents are treated like "neurons" that collaborate and critique each other to refine ideas, effectively trading slower response times for higher quality and more comprehensive outputs.

https://www.youtube.com/watch?v=GSTtLWpM3uU

## ⚠️ Alpha Software - We Need Your Help! ⚠️
Please be aware that local-deepthink is currently in an alpha stage. It is experimental research software. With the recent addition of a **code execution sandbox**, you should proceed very carefully. You may encounter bugs, unexpected behavior, or breaking changes.

Your feedback is invaluable. If you run into a crash or have ideas, please **open an issue** on our GitHub repository with your graph monitor trace log. Helping us identify and squash bugs is one of the most important contributions you can make right now!

## **Is true "deep thinking" only for trillion-dollar companies?**
**local-deepthink** is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepMind offer powerful reasoning by giving their massive models more "thinking time" in a closed environment (for a high price), local-deepthink explores a different path: **emergent intelligence on affordable local hardware**. We simulate a society of AI agents that collaborate, critique, and evolve their understanding collectively.

Essentially, you can think of this project as a way to **max out a model's performance by trading response time for quality**. The best part is that you don't need a supercomputer. local-deepthink is designed to turn even a modest 32gb RAM CPU-only laptop into a powerful "thought mining" rig. 💻⛏️ By leveraging efficient local models, you can leave the network running for hours or even days, allowing it to "mine" a sophisticated solution to a hard problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.

## Use Cases
The **Qualitative Neural Network (QNN)** algorithm that powers this system is great for problems where the only clue you have is a vague question or for one-shotting very long prompting sessions.

*   **For Non-Programmers: Ultra-Long Creative Generation**
    Think of local-deepthink as a way to handle prompts that require ultra-long or deeply creative outputs. Do you want to theory-craft a complex system or design the lore of an entire RPG world? Normally, this requires prompting a model repeatedly and figuring out different system prompts. With local-deepthink, you give the system a high-level goal, and the QNN figures out the rest. At the end of the run, it delivers a comprehensive, queryable knowledge base, and an interrogator chain can use your points of interest to generate an exhaustive final report.

*   **For Programmers & Researchers: Conceptual Algorithm Design**
    If you are a researcher building algorithms from scratch, you can use local-deepthink to conceptualize new, SOTA (State-of-the-Art) algorithms. With the new code execution sandbox, agents can even write and test vanilla code snippets. It's an ideal tool for exploring novel solutions where external dependencies can be plugged in later.

## Changelog

*   **Code Execution Sandbox:** Agents now have access to a code sandbox, allowing them to write and execute code to solve problems, perform calculations, or prototype solutions.
*   **Interactive RAG Chat & Diagnostic Tool:** The process now pauses after the final epoch, allowing you to directly **chat with the generated RAG index**. This powerful diagnostic feature lets you interrogate the massive "cube of thinking text" from all hidden layers, ask follow-up questions, and gain extra insights.
*   **Final Knowledge Harvest & RAG:** The run now archives all agent outputs into a multi-layered RAPTOR index. Upon completion, `interrogator` and `paper_formatter` agents use this RAG index to write a formal academic-style paper answering expert-level questions about the problem.
*   **Dynamic Critique Annealing:** The network's "loss function" now evolves. A meta-process analyzes the collective output and selects a "pseudo-neurotransmitter" that rewrites the critique agent's persona (e.g., from a 'senior manager' to a 'cynical philosopher-king') to provide different feedback for the next epoch and avoid local minima.
*   **Dynamic Problem Re-framing:** The network can now assess its own progress. If it determines it has made a "significant breakthrough," it formulates a new, more advanced problem that builds upon its success.
*   **Divide and Conquer - Automatic Problem Decomposition:** local-deepthink now starts by breaking down the user's initial problem into smaller, granular sub-problems, assigning each agent a unique piece of the puzzle.
*   **Perplexity Metrics & Chart:** A `metrics` node calculates the average perplexity of all agent outputs after each epoch, plotted on a live chart in the GUI.
*   **Dynamic Summarization:** A specialized chain now automatically creates a concise summary of an agent's older memories if its memory log gets too long, preserving key insights.

## The Core Idea: Qualitative Backpropagation

The core experiment is the **Qualitative Neural Network (QNN)**, an algorithm inspired by backpropagation in traditional neural networks. It's a numerical algorithm, of course, but what if the principle could be applied qualitatively? Instead of sending back a numerical error signal, you send back a "reflection."

After the network produces a solution, a "critique" agent reviews it and provides criticism. This feedback is then used to **automatically re-write the core system prompts** of the agents that contributed. The goal is for the network to "learn" from its mistakes over multiple cycles, refining not just its answers, but its own internal structure and approach. QNNs are also extremely human-interpretable, unlike their numerical counterparts.

### The Trade-Off: Speed for Depth

The obvious trade-off here is speed. A 6-layer network with 6 agents per layer, running for 20 cycles, can easily take 12 hours to complete. You're trading quick computation for a slow, iterative process of refinement. The algorithm excels in problems where creativity and insight override pure precision, like developing new frameworks in the social sciences.

## The QNN Algorithm: From Individual Agents to a Collective Mind

The core of local-deepthink is the novel QNN algorithm that orchestrates LLM agents into a dynamic, layered network. This architecture facilitates a "forward pass" for problem-solving, a unique "reflection pass" for learning, and a final "harvest pass" for knowledge extraction.

### The Forward Pass

In a QNN, the "weights" and "biases" of the network are not numerical values but the rich, descriptive personas of its agents, defined in natural language.

1.  **Input Layer & Decomposition**: The process starts with a user's high-level problem. A `master strategist` node first **decomposes this problem into smaller, distinct sub-problems**. These are then assigned to the first layer of agents.
2.  **Building Depth with Dense Layers**: A `dense-spanner` chain analyzes the agents of the preceding layer and spawns a new agent in the next layer, specifically engineered to tackle a tailored challenge.
3.  **Action**: A user's prompt initiates a cascade of information through the network until the final layer is reached, constituting a full "forward pass" of collaborative inference.

### The Reflection Pass: Learning, Critiquing, and Evolving Goals

This is where a QNN truly differs from a simple multi-agent system.

1.  **Synthesis and Metrics**: A `synthesis_node` merges the final outputs into a single solution, and a `metrics_node` calculates a perplexity score for the epoch.
2.  **Dynamic Annealing of Critique**: An `update_critique_prompt` node analyzes the hidden-layer outputs and dynamically rewrites the critique agent's system prompt. This global loss function dynamically swaps personas—from a "lazy manager" to a "philosopher king" or a "harsh drill sergeant"—to prevent the network from getting stuck in local minima.
3.  **The Crossroads of Progress**: A `progress_assessor` node evaluates whether the solution represents "significant progress."
4.  **Path A: The "Eureka!" Path**: If a breakthrough is achieved, a `problem_reframer` node formulates a new, more challenging problem.
5.  **Path B: The "Refinement" Path**: If not, the `critique_chain` scrutinizes the solution and generates detailed critiques.
6.  **Updating the "Neural" Weights**: This feedback propagates backward, and an `update_agent_prompts_node` modifies each agent's core system prompt for the next cycle.

### The Final Harvest Pass: Consolidating Knowledge

1.  **Archival and RAG Indexing**: All agent outputs from every epoch are used to build a comprehensive RAPTOR RAG index.
2.  **Pause for Interactive Chat & Diagnosis**: The network pauses, allowing you to directly query the RAG index. Because QNNs are highly interpretable, you can even diagnose a specific "neuron" by asking the chat about `agent_1_1` to get that specific agent's entire history.
3.  **Interrogation and Synthesis**: When you're done, your chat is added to the knowledge base. An `interrogator` agent then formulates expert-level questions about the original problem based on your points of interest.
4.  **Generating the Final Report**: A `paper_formatter` agent uses the RAG index to answer these questions, synthesizing the information into formal research papers. The final output is a downloadable ZIP archive of this report.

## Vision & Long-Term Roadmap: Training a World Language Model

Every local-deepthink run generates a complete, structured trace of a multi-agent collaborative process—a dataset capturing the evolution of thought. We see this as **powerful, multi-dimensional data for training next-generation reasoning models.**

Our ultimate objective is to use this data to train a true **"World Language Model" (WLM)**. A WLM would move beyond predicting the next token to understanding the fundamental patterns of collaboration, critique, and collective intelligence. The exciting possibility is that fine-tuning a model on thousands of these QNN logs might make static system prompts obsolete, as the trained LLM would learn to implicitly figure them out and dynamically switch its reasoning process on the fly.

## Mid-Term Research Goals & How You Can Help
This is still alpha software, and we need your help. Besides the value you get after "mining" a solution, it's also super entertaining to watch the neurons interact with each other! If you have the hardware, please consider helping us benchmark.

*   **Hunt Bugs**: If you run into a crash, please open an issue with your graph monitor trace log.
*   **Deep Runs & Benchmarking**: I don't have access to systems like Google's DeepMind, so it would be fantastic if someone with a powerful local rig could run and benchmark moderate-to-large QNNs.
*   **Thinking Models Support**: Help integrate support for dedicated "thinking models".
*   **P2P Networking for Distributed Mining:** My background is in Python and AI, not distributed systems. A long-term vision is a P2P networking layer to allow multiple users to connect their instances and collectively "mine" a solution to a massive problem. If you have experience here, I would love to collaborate.
*   **Checkpoint Import/Export**: Implement a mechanism that allows you to save a QNN run and pick it up where you left off, making the system more crash-resistant.

## What's Next?
The current focus is on polishing and debugging existing features to reach a beta phase. After that, the next iteration will introduce specialized modes and advanced capabilities:

*   **Story Telling Mode:** A dedicated mode for generating complex narratives, characters, and plots.
*   **World Simulation Mode:** For creating and simulating intricate worlds with consistent lore, physics, and histories, perfect for RPGs and theoretical systems.
*   **Recursive Module Stitching:** An advanced feature enabling the system to design, code, and recursively assemble different software modules to create complex, full-stack applications from a high-level prompt.

## Hyperparameters & Hardware Guidelines ⚙️

*   **`CoT trace depth`**: The number of layers in your agent network.
*   **`Number of epochs`**: One full cycle of a forward and reflection pass.
*   **`Vector word size`**: The number of "seed verbs" for initial agent creation.
*   **`Number of Questions for Final Harvest`**: The number of questions the `interrogator` agent generates.
*   **`Prompt alignment` (0.1 - 2.0)**: How strongly an agent's career is influenced by the user's prompt.
*   **`Density` (0.1 - 2.0)**: Modulates the influence of the previous layer when creating new agents.
*   **`Learning rate` (0.1 - 2.0)**: Controls the magnitude of change an agent makes to its prompt.

#### Hardware Recommendations:
*   **CPU-Only Laptop (32GB RAM)**: 2x2 or 4x4 networks with 3-4 epochs are ideal.
*   **High-End Rig (64GB RAM + 24GB GPU)**: 6x6 up to 10x10 networks with 2-10 epochs should be doable in 20-45 minutes.

## Technical Setup

*   **Backend**: FastAPI, LangChain, LangGraph, Ollama
*   **Frontend**: HTML, CSS, JavaScript

### Installation and Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andres-ulloa-de-la-torre/local-deepthink
    cd local-deepthink
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
4.  **Install and Run Ollama**:
    *   Follow the official instructions to install Ollama.
    *   Download the primary model (default is `dengcao/Qwen3-30B-A3B-Instruct-2507:latest`).
        ```bash
        ollama pull dengcao/Qwen3-30B-A3B-Instruct-2507:latest
        ```
    *   **Download the compulsory summarization model.** local-deepthink requires `qwen3:1.7b`.
        ```bash
        ollama pull qwen3:1.7b
        ```
    *   Ensure the Ollama application is running.
5.  **Run the application:**
    ```bash
    uvicorn app:app --reload
    ```
6.  **Access the GUI:** Open your browser to `http://127.0.0.1:8000`.

## How It Works

1.  **Architect the Network**: Use the GUI to set the hyperparameters for your QNN.
2.  **Pose a Problem**: Enter the high-level prompt you want the network to solve.
3.  **Build and Run**: Click the "Build and Run Graph" button.
4.  **Observe the Emergence**: Monitor the process in the real-time log viewer.
5.  **Chat and Diagnose**: Once epochs are complete, use the chat interface to query the RAG index of the network's entire thought process.
6.  **Harvest and Download**: When finished chatting, click "HARVEST" to generate and download the final ZIP report.

It’s an open-source experiment, and I’d be grateful for any thoughts, feedback, or ideas you might have. Please support the repo if you want to see more open-source work like this!

Thanks.