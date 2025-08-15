# Network of Agents (NoA): Democratizing Deep Thought 🧠

**Is true "deep thinking" only for trillion-dollar companies?**

NoA is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepThink offer powerful reasoning by giving their massive models more "thinking time" in a closed environment, NoA explores a different path: **emergent intelligence**. We simulate a society of AI agents that collaborate, critique, and evolve their understanding collectively. The most exciting part? You don't need a supercomputer. NoA is designed to turn even a modest 32gb RAM laptop into a powerful "solution mining" rig. 💻⛏️ By leveraging efficient local models (like a quantized 30B-a3b parameter version of Qwen), you can leave the graph-network running for hours or even days, allowing it to iteratively refine its approach and "mine" a sophisticated solution to a hard problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.

This project is an open invitation to explore the frontiers of collective AI, putting the power of deep, multi-day problem-solving into the hands of everyone. 🌍

## The NoA Algorithm: From Individual Agents to a Collective Mind

The core of NoA is a novel algorithm that orchestrates LLM agents into a dynamic, layered network. This architecture facilitates a "forward pass" for problem-solving and a unique "reflection pass" for learning, which is conceptually analogous to backpropagation in ANNs.

### The Forward Pass

In NoA, the "weights" and "biases" of the network are not numerical values but the rich, descriptive personas of its agents, defined in natural language.

1.  **Input Layer**: The process starts with a user's high-level problem. NoA generates a set of abstract "seed verbs" related to this problem and assigns them to diverse MBTI personality archetypes. An `input-spanner` chain then forges the first layer of agents, each with a unique persona, a specialized career aligned with the problem, and a distinct set of skills. This ensures cognitive diversity from the very beginning.

2.  **Building Depth with Dense Layers**: To create a truly deep and specialized network, a `dense-spanner` chain analyzes the agents of the preceding layer. It identifies their core attributes and formulates a "hard request"—a tailored challenge designed to push the boundaries of that agent's expertise. It then spawns a new agent in the next layer, specifically engineered to tackle this challenge. This process mirrors the creation of specialized hidden layers in a neural network, progressively increasing the system's intellectual depth.

3.  **The Flow of Understanding**: A user's prompt initiates a cascade of information through the network. The input layer agents process the prompt, and their structured JSON outputs are broadcast to every agent in the subsequent layer. This dense, layer-by-layer processing continues until the final layer is reached, constituting a full "forward pass" of collaborative inference.

### The Reflection Pass: Learning as a Distributed Phenomenon

Traditional backpropagation adjusts numerical weights based on a calculated error gradient. NoA simulates this process using a metaheuristic approach where critique and challenge drive adaptation across the network.

1.  **Synthesis and Critique as a "Loss Function"**: After the forward pass, a `synthesis_node` intelligently merges the outputs from the final agent layer into a single, coherent solution. This synthesized answer is then scrutinized by a `critique_chain`, which acts as the network's "loss function." It assesses the solution's quality against the original problem and generates a global, constructive critique—a roadmap for improvement.

2.  **Propagating Insight Backwards**: The reflection pass is where collective learning occurs. The global critique is sent to the *penultimate* layer of agents, who use it as a catalyst to refine their core system prompts. The true innovation lies in what happens next: the critique chain issues a critique on the measure of presumed responsability the agent in the hidden layer had over the final response.

3.  **Updating the "Neural" Weights**: This chain reaction continues backward through the entire network. At each step, an `update_agent_prompts_node` uses the incoming adptation signal (the critique) to modify the receiving agents' system prompts. It refines their skills, attributes, and even their professional roles. This is how the network learns—not through a central controller, but through a distributed process of peer-to-peer challenge and adaptation.

This entire cycle of a forward and reflection pass is one "epoch." By running multiple epochs, the network engages in a process of collective sense-making, iteratively refining its agent personas to generate progressively more insightful and robust solutions.

## Hyperparameters Explained: Tuning Your Solution Mining Rig ⚙️

The behavior of the NoA network is governed by several key hyperparameters, allowing you to architect the collaborative "mining" process.

*   **`CoT trace depth`**: This is the number of layers in your agent network. A deeper network allows for more steps of abstraction and refinement, similar to how deeper neural networks can model more complex functions. Each layer builds upon the insights of the last.
*   **`Number of epochs`**: An epoch is one full cycle of a forward pass (inference) and a reflection pass (learning). This is the key to "solution mining." Running more epochs allows the agents to iteratively refine their personas and collective strategy based on feedback, leading to more sophisticated solutions over time. Set it high and let your laptop mine for a breakthrough overnight! 🌙
*   **`Vector word size`**: This determines the number of "seed verbs" that form the initial skill set for each agent in the first layer. A larger size provides a richer, more diverse starting point for the agents' abilities.
*   **`Prompt alignment` (0.1 - 2.0)**: This parameter controls how strongly an agent's assigned "career" is influenced by the user's prompt. A higher value means agents will be more narrowly specialized to the problem domain, while lower value allows for more creative, tangential, and potentially innovative professional roles.
*   **`Density` (0.1 - 2.0)**: Density modulates the influence of the previous layer's attributes when creating a new, deeper layer of agents. High density means new agents will be very similar to their predecessors, leading to specialized refinement. Low density encourages more novelty and exploration in the agent creation process.
*   **`Learning rate` (0.1 - 2.0)**: Analogous to its counterpart in ANNs, this parameter controls the magnitude of change an agent makes to its system prompt in response to critique during the reflection pass. A higher learning rate leads to more drastic and rapid evolution, while a lower rate results in more subtle, gradual refinement.

## Roadmap and Future Directions (TODO)

NoA is in its early stages of development, and this README represents a research vision as much as a finished product. We are actively exploring several key areas to enhance the network's capabilities and efficiency. Community involvement and testing are crucial for making progress on these frontiers.

*   **Implement Cyclical Hierarchical Sparse Connections:** The current model uses a densely connected structure where every agent in a layer receives input from all agents in the previous layer. We plan to explore a more sophisticated and efficient architecture. By creating sparse, hierarchical connections, we hope to see the **emergence of "leader" agents and specialized "micro-teams,"** improving scalability and mirroring real-world collaboration more closely.

*   **Scale to a "World-of-Agents"**: On more powerful hardware, the current metaheuristic could be scaled significantly. We aim to move beyond using simple "seed verbs" for initial agent creation. The next step is to use complex **"institutional directives"** as the foundational vectors, allowing the network to model and solve large-scale societal or organizational problems.

*   **Enhance Combinatorial Heuristics**: The philosophical underpinning of NoA is that complex human solutions emerge from the combinatorial power of language and symbols. We plan to research and implement more advanced heuristics to guide the LLM agents, improving their ability to reason symbolically and generate novel solutions through the synthesis of diverse concepts.

## Immediate Roadmap (Short-Term TODO)

These are the immediate, actionable goals we are focused on for the next development cycle:

*   **Dynamic Memory Summarization:** To support long-running epochs and LLMs with small context windows, an agent's memory (list of its past JSON outputs) can grow too large. We will implement a dynamic summarization layer that automatically condenses an agent's memory history when it approaches a token limit, preserving key insights while preventing context overflow.

*   **Live Graph Terminal Visualization:** The current log provides a linear stream of events. We will build a second, parallel terminal interface that renders the agent network as a graph using live-updating text. This will allow users to visualize the flow of data during the forward pass and the backpropagation of critiques during the reflection pass in real-time, preserving the graph's shape for intuitive monitoring.

*   **Peer-to-Peer Networking for Distributed Mining:** To truly democratize deep thought, we will add an optional P2P networking layer. This will allow multiple users to connect their NoA instances, distributing the agent computation across a network of machines. This enables a group to pool their resources and "mine" a solution to a very hard problem collectively.

*   **Conduct High-VRAM Scalability Tests:** We will benchmark the system on a high-end machine with at least 128GB of VRAM. The goal is to test the upper limits of the metaheuristic by dramatically increasing the number of agents, the `CoT trace depth`, and the complexity of the "institutional directives" to understand its scaling properties.

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

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.
    ```
    fastapi
    uvicorn
    python-dotenv
    langchain-google-genai
    langgraph
    sse-starlette
    ollama
    langchain-community
    ```

4.  **Set up your environment variables:**
    *   Create a file named `.env` in the project root.
    *   If using Google Gemini, add your API key to the file:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

5.  **Run the application:**
    ```bash
    uvicorn app:app --reload
    ```

6.  **Access the GUI:**
    Open your web browser and navigate to `http://127.0.0.1:8000`.

### Using Local Inference with Ollama

This application supports running inference locally using an Ollama server, allowing you to use open-source models without needing a cloud API key.

1.  **Install Ollama**: Follow the official instructions to install Ollama on your system.
2.  **Download a Model**: Pull the desired model from the Ollama library. The default model for this application is `dengcao/Qwen3-3B-A3B-Instruct-2507:latest`. You can pull it by running:
    ```bash
    ollama pull dengcao/Qwen3-3B-A3B-Instruct-2507:latest
    ```
3.  **Run Ollama Server**: Make sure the Ollama application is running in the background. It will serve the models locally.
4.  **Select in GUI**: In the web interface, use the "LLM Provider" dropdown and select "Local Ollama". You can change the model name from the default if you have other models pulled.

## How It Works

1.  **Choose your LLM**: Select between Google Gemini or Local Ollama.
2.  **Architect the Network**: Use the GUI to set the hyperparameters that define your network's structure and learning capacity.
3.  **Pose a Problem**: Enter the high-level prompt you want the agent network to solve.
4.  **Build and Run**: Click the "Build and Run Graph" button to initiate the process.
5.  **Observe the Emergence**: The backend dynamically constructs the agent network using LangGraph. You can monitor the entire process—agent creation, forward inference, and reflective learning—in the real-time log viewer.
6.  **Receive the Synthesized Result**: Once all epochs are complete, the final, synthesized answer from the collective is displayed.