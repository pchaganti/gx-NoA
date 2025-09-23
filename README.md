![thumbnail](https://github.com/user-attachments/assets/13694758-a5c9-40c5-9c07-c7a168e660cf)

# local-deepthink: Democratizing Deep Algorithmic Thought 🧠

I've been thinking a lot about how we, as people, develop complex ideas and algorithms. It's rarely a single, brilliant flash of insight. Our minds are shaped by the countless small interactions we have—a conversation here, an article there. This environment of constant, varied input seems just as important as the act of thinking itself.

I wanted to see if I could recreate a small-scale version of that "soup" required for true algorithmic insight for local LLMs. The result is this project, **local-deepthink**. It's a system that runs a novel conceptual algorithm called a **Qualitative Neural Network (QNN)**. In a QNN, different AI agents are treated like "neurons" that collaborate and critique each other to refine complex solutions, effectively trading slower response times for higher quality and more robust outputs.

## ⚠️ Alpha Software - We Need Your Help! ⚠️
Please be aware that local-deepthink is currently in an alpha stage. It is experimental research software. You may encounter bugs, unexpected behavior, or breaking changes.

Your feedback is invaluable. If you run into a crash or have ideas, please **open an issue** on our GitHub repository with your graph monitor trace log. Helping us identify and squash bugs is one of the most important contributions you can make right now!


## **Is true "deep thinking" only for trillion-dollar companies?**

**local-deepthink** is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepMind offer powerful reasoning by giving their massive models more "thinking time" in a closed environment (for a high price), local-deepthink explores a different path: **emergent intelligence on affordable local hardware**. We simulate a society of AI agents that collaborate, evolve, and deepen their understanding of a complex problem collectively over time.

Essentially, you can think of this project as a way to **max out a model's performance on complex algorithmic tasks by trading response time for quality**. The best part is that you don't need a supercomputer. local-deepthink is designed to turn even a modest 32gb RAM CPU-only laptop into a powerful "thought mining" rig. 💻⛏️ By leveraging efficient local models, you can leave the network running for hours or even days, allowing it to "mine" a sophisticated solution to a hard algorithmic problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.

## Use Case: Advanced Algorithm Generation
The **Qualitative Neural Network (QNN)** algorithm that powers this system is great for complex problems where the only clue you have is a vague question or a high-level conceptual goal. With the system now refocused exclusively on code and algorithm generation, its primary use case is to tackle difficult programming challenges.

*   **For Programmers & Researchers: Full Stack App Generation**: This is an experimental feature. As of right now the system can accumulate attempts and offer one final synthetized solution. This works well for algorithm design. Now we are trying to decompose the synthesis in modules that are aftwerads stitched together through abstract interface cards. It wont work on a few epochs, so if you can test it on high epochs and open an issue, it would be appreciated.

## Changelog

*  **Hidden-layer-fixed**: Issue with meta-prompting in the hidden layer fixed. Agents are now moderately divergent from a strict skill alignment, as originally intended. Specialization is one thing; the individual that serves as recipient for the toolset is another. Keeping both distinct is important to make answers smoother.
*   **QNN Export/Import:** You can now export the entire state of a trained agent network (QNN) to a JSON file. This QNN can be imported and used for inference on new problems without rerunning the entire epoch process.
*   **Code Generation & Sandbox:** The system can now generate, synthesize, and safely execute Python code. A new `code_execution` node validates the final code, and successful modules provide context for future epochs.
*   **Dynamic Problem Re-framing:** The network can now assess its own progress. After each cycle (epoch), it formulates a new, more advanced problem that builds upon its previous solution, forcing the agents to continuously deepen their understanding.
*   **Divide and Conquer - Automatic Problem Decomposition:** local-deepthink now starts by breaking down the user's initial problem into smaller, granular sub-problems, assigning each agent a unique piece of the puzzle.
*   **Perplexity Metrics & Chart:** A `metrics` node calculates the average perplexity of all agent outputs after each epoch, plotted on a live chart in the GUI.
*   **Dynamic Summarization:** A specialized chain now automatically creates a concise summary of an agent's older memories if its memory log gets too long, preserving key insights while managing context length.

## The Core Idea: Qualitative Backpropagation

The core experiment is the **Qualitative Neural Network (QNN)**, an algorithm inspired by backpropagation in traditional neural networks. It's a numerical algorithm, of course, but what if the principle could be applied qualitatively? Instead of sending back a numerical error signal, you send back a "reflection."

After the network produces a solution, a "reflection pass" analyzes the result and **automatically re-writes the core system prompts** of the agents that contributed. The goal is for the network to "learn" from its own output over multiple cycles (epochs), refining not just its answers, but its own internal structure and approach. QNNs are also extremely human-interpretable, unlike their numerical counterparts.

### The Trade-Off: Speed for Depth

The obvious trade-off here is speed. A 6-layer network with 6 agents per layer, running for 20 epochs, can easily take 12 hours to complete. You're trading quick computation for a slow, iterative process of refinement. The algorithm excels in problems where creativity and insight override pure precision, like developing new frameworks in the social sciences.

## The QNN Algorithm: From Individual Agents to a Collective Mind

The core of local-deepthink is the novel QNN algorithm that orchestrates LLM agents into a dynamic, layered network. This architecture facilitates a "forward pass" for problem-solving, a "reflection pass" for learning, and a final "harvest pass" for knowledge extraction.

### The Forward Pass

In a QNN, the "weights" and "biases" of the network are not numerical values but the rich, descriptive personas of its agents, defined in natural language.

1.  **Input Layer & Decomposition**: The process starts with a user's high-level problem. A `master strategist` node first **decomposes this problem into smaller, distinct sub-problems**. These are then assigned to the first layer of agents.
2.  **Building Depth with Dense Layers**: A `dense-spanner` chain analyzes the agents of the preceding layer and spawns a new agent in the next layer, specifically engineered to tackle a tailored challenge.
3.  **Action**: A user's prompt initiates a cascade of information through the network until the final layer is reached, constituting a full "forward pass" of collaborative inference.

### The Reflection Pass: Learning Through Evolving Goals

This is where a QNN truly differs from a simple multi-agent system. Instead of simply correcting errors, the network learns by continuously raising the bar.

1.  **Synthesis and Metrics**: A `synthesis_node` merges the final outputs into a single solution, and a `metrics_node` calculates a perplexity score for the epoch.
2.  **Problem Re-framing**: The core of the learning loop. A `problem_reframer` node analyzes the synthesized solution and formulates a new, more ambitious problem that represents the "next logical step." This prevents the network from stagnating and pushes it toward deeper insights.
3.  **Decomposition of the New Problem**: The newly framed problem is then broken down again into a new set of granular sub-problems.
4.  **Updating the "Neural" Weights**: This new set of sub-problems is propagated backward through the network. An `update_agent_prompts_node` modifies each agent's core system prompt to align with its new, more advanced task for the next epoch.

### The Final Harvest Pass: Consolidating Knowledge

1.  **Archival and RAG Indexing**: All agent outputs from every epoch are used to build a comprehensive RAPTOR RAG index.
2.  **Pause for Interactive Chat & Diagnosis**: The network pauses, allowing you to directly query the RAG index. Because QNNs are highly interpretable, you can even diagnose a specific "neuron" by asking the chat about `agent_1_1` to get that specific agent's entire history.
3.  **Interrogation and Synthesis**: When you're done, your chat is added to the knowledge base. An `interrogator` agent then formulates expert-level questions about the original problem based on your points of interest.
4.  **Generating the Final Report**: A `paper_formatter` agent uses the RAG index to answer these questions, synthesizing the information into formal research papers. The final output is a downloadable ZIP archive of this report.

## Vision & Long-Term Roadmap: Training a World Language Model

Every local-deepthink run generates a complete, structured trace of a multi-agent collaborative process—a dataset capturing the evolution of thought. With the new export feature, these QNN JSON files can now be collected. We see this as **powerful, multi-dimensional data for training next-generation reasoning models.**

Our ultimate objective is to use this data to train a true **"World Language Model" (WLM)**. A WLM would move beyond predicting the next token to understanding the fundamental patterns of collaboration, critique, and collective intelligence. The exciting possibility is that fine-tuning a model on thousands of these QNN logs might make static system prompts obsolete, as the trained LLM would learn to implicitly figure them out and dynamically switch its reasoning process on the fly.

## Mid-Term Research Goals & How You Can Help
This is still alpha software, and we need your help. Besides the value you get after "mining" a solution, it's also super entertaining to watch the neurons interact with each other! If you have the hardware, please consider helping us benchmark.

*   **Hunt Bugs**: If you run into a crash, please open an issue with your graph monitor trace log.
*   **Deep Runs & Benchmarking**: I don't have access to systems like Google's DeepMind, so it would be fantastic if someone with a powerful local rig could run and benchmark moderate-to-large QNNs.
*   **Thinking Models Support**: Help integrate support for dedicated "thinking models".
*   **P2P Networking for Distributed Mining:** My background is in Python and AI, not distributed systems. A long-term vision is a P2P networking layer to allow multiple users to connect their instances and collectively "mine" a solution to a massive problem. If you have experience here, I would love to collaborate.
*   **Checkpoint Import/Export**: A basic version is implemented, but expanding this to allow saving a run mid-epoch would make the system more crash-resistant.

## What's Next?
The current focus is on polishing and debugging existing features to reach a beta phase. After that, the next iteration will introduce specialized modes and advanced capabilities:

*   **Recursive Module Stitching:** The initial implementation allows code validation and context feedback. The next step is to enable the system to design, code, and recursively assemble different software modules to create complex, full-stack applications from a high-level prompt.
*   **Export your QNN:** This is now implemented! You can import and export your QNN in plain JSON format, so other people can prompt it, at just a few MBs of size.

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
    git clone https://github.com/iblameandrew/local-deepthink
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
    *   Download a primary model for the agents (default is `dengcao/Qwen3-3B-A3B-Instruct-2507:latest`).
        ```bash
        ollama pull dengcao/Qwen3-3B-A3B-Instruct-2507:latest
        ```
    *   **Download the compulsory summarization model.** local-deepthink requires `qwen3:1.7b` for its internal processes.
        ```bash
        ollama pull qwen3:1.7b
        ```
    *   Ensure the Ollama application is running.
5.  **Run the application:**
    ```bash
    uvicorn new:app --reload
    ```
6.  **Access the GUI:** Open your browser to `http://127.0.0.1:8000`.

## How It Works

1.  **Architect the Network**: Use the GUI to set the hyperparameters for your QNN.
2.  **Pose a Problem**: Enter the high-level prompt you want the network to solve.
3.  **Build and Run**: Click the "Build and Run Graph" button.
4.  **Observe the Emergence**: Monitor the process in the real-time log viewer.
5.  **Chat and Diagnose**: Once epochs are complete, use the chat interface to query the RAG index of the network's entire thought process.
6.  **Harvest and Download**: When finished chatting, click "HARVEST" to generate and download the final ZIP report.
7.  **(Optional) Export, Import, and Infer**: Use the `Export QNN` button to save your network. Later, use the `Import QNN` button to load it and run new prompts against the trained agent structure.

It’s an open-source experiment, and I’d be grateful for any thoughts, feedback, or ideas you might have. Please support the repo if you want to see more open-source work like this!

Thanks.