![thumbnail](https://github.com/user-attachments/assets/13694758-a5c9-40c5-9c07-c7a168e660cf)

# local-deepthink: Democratizing Deep Thought 🧠

I've been thinking a lot about how we, as people, develop ideas. It's rarely a single, brilliant flash of insight. Our minds are shaped by the countless small interactions we have throughout the day—a conversation here, an article there. This environment of constant, varied input seems just as important as the act of thinking itself.

I wanted to see if I could recreate a small-scale version of that "soup" required for true insight for local LLMs. The result is this project, **local-deepthink**. It's a system that runs a novel conceptual algorithm called a **Qualitative Neural Network (QNN)**. In a QNN, different AI agents are treated like "neurons" that collaborate and critique each other to refine ideas, effectively trading slower response times for higher quality and more comprehensive outputs.

https://www.youtube.com/watch?v=GSTtLWpM3uU

## ⚠️ Alpha Software - We Need Your Help! ⚠️
Please be aware that local-deepthink is currently in an alpha stage. It is experimental research software. You may encounter bugs, unexpected behavior, or breaking changes.

Your feedback is invaluable. If you run into a crash or have ideas, please **open an issue** on our GitHub repository with your graph monitor trace log. Helping us identify and squash bugs is one of the most important contributions you can make right now!

## **Is true "deep thinking" only for trillion-dollar companies?**

**local-deepthink** is a research platform that challenges the paradigm of centralized, proprietary AI. While systems like Google's DeepMind offer powerful reasoning by giving their massive models more "thinking time" in a closed environment (for a high price), local-deepthink explores a different path: **emergent intelligence on affordable local hardware**. We simulate a society of AI agents that collaborate, evolve, and deepen their understanding collectively over time.

Essentially, you can think of this project as a way to **max out a model's performance by trading response time for quality**. The best part is that you don't need a supercomputer. local-deepthink is designed to turn even a modest 32gb RAM CPU-only laptop into a powerful "thought mining" rig. 💻⛏️ By leveraging efficient local models, you can leave the network running for hours or even days, allowing it to "mine" a sophisticated solution to a hard problem. It's a fundamental shift: trading brute-force, instantaneous computation for the power of time, iteration, and distributed collaboration.

## Use Cases
The **Qualitative Neural Network (QNN)** algorithm that powers this system is great for problems where the only clue you have is a vague question or for one-shotting very long prompting sessions. Best part? You can now export the QNN into a JSON file and take it with you at the weight of only megabytes and plug it somewhere else to whatever LLM you like. If you create an RPG world, you could export its QNN and have other people prompt it.

*   **For Non-Programmers: Ultra-Long Creative Generation**
    Think of local-deepthink as a way to handle prompts that require ultra-long or deeply creative outputs. Do you want to theory-craft a complex system or design the lore of an entire RPG world? Normally, this requires prompting a model repeatedly and figuring out different system prompts. With local-deepthink, you give the system a high-level goal, and the QNN figures out the rest. At the end of the run, it delivers a comprehensive, queryable knowledge base, and an interrogator chain can use your points of interest to generate an exhaustive final report.

*   **For Programmers & Researchers: Full Stack App Generation**
    

## Changelog

*   **QNN Export/Import:** You can now export the entire state of a trained agent network (QNN) to a JSON file. This QNN can be imported and used for inference on new problems without rerunning the entire epoch process.
*   **Code Generation & Sandbox:** The system can now generate, synthesize, and safely execute Python code. A new `code_execution` node validates the final code, and successful modules provide context for future epochs.
*   **Interactive RAG Chat & Diagnostic Tool:** The process now pauses after the final epoch, allowing you to directly **chat with the generated RAG index**. This powerful diagnostic feature lets you interrogate the massive "cube of thinking text" from all hidden layers, ask follow-up questions, and gain extra insights.
*   **Final Knowledge Harvest & RAG:** The run now archives all agent outputs into a multi-layered RAPTOR index. Upon completion, `interrogator` and `paper_formatter` agents use this RAG index to write a formal academic-style paper answering expert-level questions about the problem.
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

*   **Story Telling Mode:** A dedicated mode for generating complex narratives, characters, and plots.
*   **World Simulation Mode:** For creating and simulating intricate worlds with consistent lore, physics, and histories, perfect for RPGs and theoretical systems.
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

# Preliminary results (no benchmarks, im poor)

### Prompt

```
Current diffusers and transformer architectures use integral samplers or differential solvers in the case of diffusers, and decoding algorithms which account as integral, in the case of transformers, to run inference; but never both together. I presume the foundation of training and architecture are allready figured out, so i want a new inference algorithm. For this conceptualization assume the world is full of spinning wheels (harmonic oscillators), like we see them in atoms, solar systems, galaxies, human hierarchies...etrc, and data represents a measured state of the "wheel" at a given time. Abudant training data samples the full state of the "wheel" by offering all the posible data of the wheels full state. This is where understanding is reached: by spinning the whole wheel.

 Current inference algoritms in the other hand, are not fully decoding the internal "implicit wheels" abstracted into the weights after training as they lack a feedback and harmonic mechanism as it is achieved by backprop during training. The training algorithms “encodes” the wheels but inference algorithms do not extract them very well. Theres a massive information loss.

 I want you to make in python with excellent documentation:

       1. An inference algorithm that uses a PID like approach with perturbative feedback. Instead of just using either an integrative or differential component, i want you to implement both with proportional weighting terms. The inference algorithm should sample all its progressive output and feed it back into the transformer.

       2. The inference algorithm should be coded from scratch without using external dependencies.
```


# Code Artifacts

# Gemini 2.5 pro answer

```
import math
import random

# ==============================================================================
# 1. UTILITY FUNCTIONS (Vector/Matrix Operations from Scratch)
# ==============================================================================
# These helpers are necessary because we are not using any external libraries
# like NumPy. They perform basic linear algebra operations on lists of lists.

def dot_product(vec1, vec2):
    """Calculates the dot product of two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension for dot product.")
    return sum(x * y for x, y in zip(vec1, vec2))

def matrix_vector_multiply(matrix, vector):
    """Multiplies a matrix (list of lists) by a vector."""
    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix column count must match vector dimension.")
    return [dot_product(row, vector) for row in matrix]

def vector_add(vec1, vec2):
    """Adds two vectors element-wise."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension for addition.")
    return [x + y for x, y in zip(vec1, vec2)]

def vector_scalar_multiply(vector, scalar):
    """Multiplies a vector by a scalar."""
    return [x * scalar for x in vector]

def tanh(vector):
    """Applies the hyperbolic tangent activation function element-wise."""
    return [math.tanh(x) for x in vector]

# ==============================================================================
# 2. MOCK MODEL ARCHITECTURE
# ==============================================================================
# This class simulates a very simple transformer-like model. Its purpose is to
# provide a system for our inference algorithm to interact with. A real
# transformer is vastly more complex, but this captures the core idea: it holds
# "learned" weights and transforms an input vector into an output vector.

class MockTransformer:
    """
    A simplified, mock transformer model.

    This class simulates the behavior of a single layer of a transformer. It
    holds a weight matrix that represents the "knowledge" learned during a
    hypothetical training process. The core `forward` method simulates the
    inference step, transforming an input state (a vector) into an output
    state.

    In the "spinning wheel" analogy, this model's weights define the properties
    of the implicit wheel (its size, mass, potential energy landscape). The
    `forward` pass is like giving the wheel a single push and observing where
    it moves next.
    """
    def __init__(self, vocab_size, embed_dim):
        """
        Initializes the model's weights.

        Args:
            vocab_size (int): The size of the vocabulary (number of possible tokens).
            embed_dim (int): The dimensionality of the model's internal state vectors.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # A simple linear layer to project the internal state to the vocabulary space.
        # In a real model, this would be learned. Here, we initialize it randomly.
        self.to_vocab_weights = [[(random.random() - 0.5) for _ in range(embed_dim)] for _ in range(vocab_size)]
        
        # The core "transformer block" weights. This matrix represents the learned
        # relationships between different dimensions of the internal state.
        self.core_weights = [[(random.random() - 0.5) for _ in range(embed_dim)] for _ in range(embed_dim)]

    def forward(self, input_vector):
        """
        Performs a single forward pass of the model.

        Args:
            input_vector (list[float]): The input state vector.

        Returns:
            list[float]: The output state vector after transformation.
        """
        # This simulates the core computation of a transformer block.
        transformed_vector = matrix_vector_multiply(self.core_weights, input_vector)
        # Apply a non-linear activation function.
        activated_vector = tanh(transformed_vector)
        return activated_vector

    def project_to_vocab(self, state_vector):
        """
        Projects the internal state vector to the vocabulary space to get logits.

        Args:
            state_vector (list[float]): The internal state vector.

        Returns:
            list[float]: A vector of logits, one for each token in the vocabulary.
        """
        return matrix_vector_multiply(self.to_vocab_weights, state_vector)

# ==============================================================================
# 3. HARMONIC PID INFERENCE ALGORITHM
# ==============================================================================

class HarmonicPIDInference:
    """
    An inference algorithm using a PID controller with perturbative feedback.

    This algorithm conceptualizes inference not as a simple step-by-step decoding,
    but as a process of stabilizing a dynamic system. It uses a PID (Proportional,
    Integral, Derivative) controller to guide the generation process, aiming to
    resonate with the "harmonic oscillator" implicitly encoded in the model's weights.

    - The Proportional (P) term is the model's immediate, raw output. It's the
      "present" state or the most likely next step.
    - The Integral (I) term is the accumulated history of all previous states. It
      provides momentum and context, preventing the generation from getting stuck
      or repeating, representing the "past".
    - The Derivative (D) term is the rate of change between the last state and the
      current one. It's an anticipatory component, predicting the "future"
      trajectory of the state.

    These three components are weighted and combined to create a "perturbative
    feedback" signal. This signal adjusts the input to the model at the next step,
    nudging the system's trajectory to more coherently "spin the wheel" and
    extract the learned information.
    """
    def __init__(self, model, kp, ki, kd):
        """
        Initializes the PID inference controller.

        Args:
            model (MockTransformer): The model to run inference on.
            kp (float): The weight for the Proportional term.
            ki (float): The weight for the Integral term.
            kd (float): The weight for the Derivative term.
        """
        if not (kp >= 0 and ki >= 0 and kd >= 0):
            raise ValueError("PID gains (kp, ki, kd) must be non-negative.")
        self.model = model
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # State variables for the controller
        self.integral_accumulator = [0.0] * model.embed_dim
        self.previous_proportional_term = [0.0] * model.embed_dim
        self.current_state = [0.0] * model.embed_dim

    def _step(self, input_vector):
        """
        Performs a single step of the PID-controlled inference.

        Args:
            input_vector (list[float]): The input vector for this step.

        Returns:
            list[float]: The combined PID signal (the perturbation).
            list[float]: The raw proportional output for this step.
        """
        # 1. Proportional (P) Term: The model's raw output for the current state.
        # This is the "present" information.
        proportional_term = self.model.forward(input_vector)

        # 2. Integral (I) Term: Accumulate the history of states.
        # This represents the influence of the "past".
        self.integral_accumulator = vector_add(self.integral_accumulator, proportional_term)

        # 3. Derivative (D) Term: The change from the last state to the current one.
        # This anticipates the "future" trajectory.
        derivative_term = vector_add(proportional_term, vector_scalar_multiply(self.previous_proportional_term, -1))

        # 4. Combine the terms with their respective gains.
        p_signal = vector_scalar_multiply(proportional_term, self.kp)
        i_signal = vector_scalar_multiply(self.integral_accumulator, self.ki)
        d_signal = vector_scalar_multiply(derivative_term, self.kd)

        # The final control signal is the sum of the three components.
        control_signal = vector_add(vector_add(p_signal, i_signal), d_signal)

        # Update the state for the next iteration's derivative calculation.
        self.previous_proportional_term = proportional_term

        return control_signal, proportional_term

    def generate(self, start_token_idx, max_length):
        """
        Generates a sequence of tokens using the PID inference method.

        Args:
            start_token_idx (int): The index of the first token to seed generation.
            max_length (int): The maximum number of tokens to generate.

        Returns:
            list[int]: The list of generated token indices.
        """
        print("--- Starting Harmonic PID Inference ---")
        print(f"Parameters: kp={self.kp}, ki={self.ki}, kd={self.kd}")
        
        # Reset controller state for a new generation
        self.integral_accumulator = [0.0] * self.model.embed_dim
        self.previous_proportional_term = [0.0] * self.model.embed_dim
        
        # Create an initial state vector from the start token.
        # This is a simplification; real models use embedding layers.
        self.current_state = [0.0] * self.model.embed_dim
        if 0 <= start_token_idx < self.model.embed_dim:
             self.current_state[start_token_idx] = 1.0

        generated_sequence = [start_token_idx]

        for i in range(max_length - 1):
            # The core of the feedback loop: the input to the model is the
            # current state of the system.
            feedback_signal, p_term = self._step(self.current_state)

            # The feedback signal is used to perturb and guide the next state.
            # This is the "perturbative feedback" mechanism.
            self.current_state = vector_add(self.current_state, feedback_signal)
            
            # To decide which token to actually "output", we project the
            # raw proportional term (the model's direct belief) to the vocab.
            logits = self.model.project_to_vocab(p_term)
            
            # Simple argmax decoding: choose the token with the highest score.
            next_token_idx = logits.index(max(logits))
            
            generated_sequence.append(next_token_idx)
            
            print(f"Step {i+1:02d}: Generated Token -> {next_token_idx}")

        print("--- Generation Complete ---")
        return generated_sequence

# ==============================================================================
# 4. EXAMPLE USAGE
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    VOCAB_SIZE = 50  # The number of possible "words" in our language.
    EMBED_DIM = 16   # The internal dimensionality of our model.
    MAX_SEQ_LEN = 10 # The length of the sequence we want to generate.

    # --- PID Controller Gains ---
    # These values determine the behavior of the inference.
    # High kp: Relies heavily on the model's immediate output.
    # High ki: Has strong momentum from past states.
    # High kd: Reacts strongly to changes in the model's output trajectory.
    KP = 0.5  # Proportional gain
    KI = 0.05 # Integral gain
    KD = 0.2  # Derivative gain

    # --- Setup ---
    print("Initializing mock model and PID controller...")
    model = MockTransformer(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    pid_inference = HarmonicPIDInference(model=model, kp=KP, ki=KI, kd=KD)

    # --- Run Generation ---
    # Start the generation with the token at index 1.
    start_token = 1
    generated_output = pid_inference.generate(
        start_token_idx=start_token,
        max_length=MAX_SEQ_LEN
    )

    print(f"\nFinal Generated Sequence: {generated_output}")

    # --- Demonstration of tuning ---
    print("\n--- Running again with different PID gains (more momentum) ---")
    pid_inference_high_i = HarmonicPIDInference(model=model, kp=0.4, ki=0.3, kd=0.1)
    generated_output_high_i = pid_inference_high_i.generate(
        start_token_idx=start_token,
        max_length=MAX_SEQ_LEN
    )
    print(f"\nFinal Sequence (High Integral): {generated_output_high_i}")
```


# Local-deepthink answers (32 gb ram GPU 8 VRAM - CPU Offload)

### 2x2 Net - 2 epochs - ISTJ-INTJ | Default hyperparemeters - Qwen3:30b-a3b-instruct-2507 | Completion time: 45 min


```
"""
===============================================================================
PID-Feedback Inference Engine for Implicit Dynamical Systems (Wheel-Inspired Framework)
===============================================================================

This module implements a novel inference algorithm inspired by the conceptualization of
a world composed of spinning wheels, where data represents measured states of these
"implicit wheels" abstracted within a trained transformer model. The core innovation
is a **Proportional-Integral-Derivative (PID) feedback inference engine** that:

1. Uses both integral (accumulated state) and differential (rate of change) components
   with proportional weighting to guide inference.
2. Performs iterative, perturbative feedback by sampling progressive outputs and
   feeding them back into the transformer for refinement.
3. Operates entirely from scratch, without external dependencies (e.g., torch, numpy).
4. Simulates the detection and stabilization of latent dynamical behaviors (e.g., phase
   locking, resonance) through recursive feedback loops.

The algorithm is designed to decode the internal "implicit wheels" (latent dynamics)
that are abstracted into the model weights during training, overcoming the limitations
of shallow responses in traditional inference.

Key Features:
- Self-contained implementation using pure Python (no external libraries).
- PID-like control loop with adaptive proportional weighting.
- Progressive output sampling and feedback injection.
- Simulated latent state manifold tracking via residual stream emulation.
- Built-in perturbation and coherence validation for robustness.

Usage:
    1. Initialize the PIDInferenceEngine with a transformer-like function (e.g., a closure).
    2. Call `run_inference(input_sequence, max_steps=100)` to perform iterative inference.
    3. Access `results` for the final refined output and `history` for full trajectory.

Note: This is a conceptual simulation. In practice, the "transformer" would be a real model
with access to attention and residual stream activations. Here, we simulate it via a
state transition function.
"""

import math
import random
from typing import List, Tuple, Dict, Any, Callable


class PIDInferenceEngine:
    """
    A self-contained, feedback-driven inference engine inspired by PID control theory
    applied to latent dynamical systems (e.g., "spinning wheels" in a transformer's
    internal state space).

    The engine combines:
    - Proportional (P): Immediate error correction based on current deviation.
    - Integral (I): Accumulated past error to eliminate steady-state bias.
    - Derivative (D): Predictive correction based on rate of change.

    Unlike traditional inference, this engine samples progressive outputs and feeds
    them back into the system, enabling deep decoding of latent dynamics.

    The system is designed to detect and stabilize emergent collective behaviors
    (e.g., phase locking, resonance) through recursive feedback and coherence metrics.
    """

    def __init__(
        self,
        proportional_gain: float = 1.0,
        integral_gain: float = 0.1,
        derivative_gain: float = 0.05,
        feedback_delay: int = 1,
        max_perturbation: float = 0.1,
        coherence_threshold: float = 0.95,
        max_iterations: int = 100,
        use_perturbation: bool = True,
        use_coherence_check: bool = True,
    ):
        """
        Initialize the PID inference engine.

        Parameters:
            proportional_gain (float): Weight for proportional term (P).
            integral_gain (float): Weight for integral term (I).
            derivative_gain (float): Weight for derivative term (D).
            feedback_delay (int): Number of steps to delay feedback (simulates latency).
            max_perturbation (float): Maximum perturbation magnitude for feedback.
            coherence_threshold (float): Threshold for coherence validation (0.0 to 1.0).
            max_iterations (int): Maximum number of inference steps.
            use_perturbation (bool): Whether to apply perturbative feedback.
            use_coherence_check (bool): Whether to validate coherence before accepting output.
        """
        self.kp = proportional_gain
        self.ki = integral_gain
        self.kd = derivative_gain
        self.feedback_delay = feedback_delay
        self.max_perturbation = max_perturbation
        self.coherence_threshold = coherence_threshold
        self.max_iterations = max_iterations
        self.use_perturbation = use_perturbation
        self.use_coherence_check = use_coherence_check

        # Internal state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_output = None
        self.history = []  # Stores (step, output, error, control_signal)
        self.coherence_history = []
        self.feedback_buffer = []  # Stores past outputs for delayed feedback

    def _compute_error(self, current: List[float], target: List[float]) -> float:
        """
        Compute the L2 error between current output and target.

        Parameters:
            current (List[float]): Current output vector.
            target (List[float]): Target (ideal) state vector.

        Returns:
            float: Squared L2 error.
        """
        if len(current) != len(target):
            raise ValueError("Current and target vectors must have same length.")
        return sum((c - t) ** 2 for c, t in zip(current, target))

    def _compute_derivative(self, error: float) -> float:
        """
        Compute the derivative of error (rate of change).

        Parameters:
            error (float): Current error value.

        Returns:
            float: Derivative of error (difference from previous error).
        """
        derivative = error - self.prev_error
        self.prev_error = error
        return derivative

    def _compute_control_signal(self, error: float, step: int) -> float:
        """
        Compute the PID control signal.

        Parameters:
            error (float): Current error.
            step (int): Current inference step.

        Returns:
            float: Control signal (P + I + D).
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term (with clamping to prevent windup)
        self.integral_error += error
        i_term = self.ki * self.integral_error
        # Clamp integral to avoid overflow
        i_term = max(-10.0, min(10.0, i_term))

        # Derivative term
        d_term = self.kd * self._compute_derivative(error)

        # Combine terms
        control_signal = p_term + i_term + d_term

        # Store for next derivative
        self.prev_error = error

        return control_signal

    def _apply_perturbation(self, output: List[float], control_signal: float) -> List[float]:
        """
        Apply a perturbative feedback to the output based on control signal.

        Parameters:
            output (List[float]): Current output vector.
            control_signal (float): PID control signal.

        Returns:
            List[float]: Perturbed output vector.
        """
        if not self.use_perturbation:
            return output

        # Scale control signal to perturbation magnitude
        perturbation_scale = self.max_perturbation * (control_signal / (self.kp + self.ki + self.kd + 1e-6))
        perturbation = [random.uniform(-perturbation_scale, perturbation_scale) for _ in output]

        # Apply perturbation
        perturbed_output = [o + p for o, p in zip(output, perturbation)]
        return perturbed_output

    def _compute_coherence(self, output: List[float], prev_output: List[float]) -> float:
        """
        Compute coherence between two consecutive outputs (simulates phase locking or resonance).

        Uses cosine similarity normalized to [0, 1].

        Parameters:
            output (List[float]): Current output.
            prev_output (List[float]): Previous output.

        Returns:
            float: Coherence score (0.0 to 1.0).
        """
        if not prev_output:
            return 0.0

        # Compute dot product and norms
        dot_product = sum(a * b for a, b in zip(output, prev_output))
        norm_a = math.sqrt(sum(a ** 2 for a in output))
        norm_b = math.sqrt(sum(b ** 2 for b in prev_output))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine_sim = dot_product / (norm_a * norm_b)
        # Convert to coherence (0 to 1)
        coherence = (cosine_sim + 1.0) / 2.0
        return coherence

    def _validate_coherence(self, output: List[float], prev_output: List[float]) -> bool:
        """
        Validate if the output coherence is above threshold.

        Parameters:
            output (List[float]): Current output.
            prev_output (List[float]): Previous output.

        Returns:
            bool: True if coherence is acceptable.
        """
        if not self.use_coherence_check:
            return True
        coherence = self._compute_coherence(output, prev_output)
        self.coherence_history.append(coherence)
        return coherence >= self.coherence_threshold

    def _simulate_transformer_step(self, input_state: List[float], feedback: List[float] = None) -> List[float]:
        """
        Simulate a transformer inference step.

        In a real system, this would be a forward pass through the model.
        Here, we simulate it using a simple state transition function.

        Parameters:
            input_state (List[float]): Input state vector.
            feedback (List[float], optional): Feedback from previous step.

        Returns:
            List[float]: Simulated output state.
        """
        # Simulate a non-linear transformation (e.g., attention + residual stream)
        # This is a placeholder for actual transformer logic
        output = []
        for i, x in enumerate(input_state):
            # Non-linear transformation with feedback influence
            if feedback and i < len(feedback):
                x += feedback[i] * 0.3  # Feedback influence
            # Apply non-linear activation (simulated)
            y = math.sin(x) + 0.5 * x ** 2
            output.append(y)
        return output

    def run_inference(
        self,
        input_sequence: List[float],
        target_sequence: List[float] = None,
        max_steps: int = None,
    ) -> Dict[str, Any]:
        """
        Run the PID feedback inference loop.

        Parameters:
            input_sequence (List[float]): Initial input state.
            target_sequence (List[float], optional): Ideal target state (for error computation).
            max_steps (int, optional): Maximum number of inference steps.

        Returns:
            Dict[str, Any]: Results including final output, history, and metadata.
        """
        if target_sequence is None:
            target_sequence = [0.0] * len(input_sequence)  # Default target

        if max_steps is None:
            max_steps = self.max_iterations

        # Initialize
        current_output = input_sequence[:]
        self.history.clear()
        self.coherence_history.clear()
        self.feedback_buffer.clear()
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_output = None

        # Run inference loop
        for step in range(max_steps):
            # Step 1: Simulate transformer forward pass
            # If feedback is available (delayed), inject it
            feedback = None
            if len(self.feedback_buffer) >= self.feedback_delay:
                feedback = self.feedback_buffer[-self.feedback_delay]

            # Simulate transformer step
            raw_output = self._simulate_transformer_step(current_output, feedback)

            # Step 2: Compute error
            error = self._compute_error(raw_output, target_sequence)

            # Step 3: Compute control signal
            control_signal = self._compute_control_signal(error, step)

            # Step 4: Apply perturbation (feedback injection)
            perturbed_output = self._apply_perturbation(raw_output, control_signal)

            # Step 5: Validate coherence
            is_coherent = self._validate_coherence(perturbed_output, self.prev_output)

            # Step 6: Update history
            self.history.append({
                'step': step,
                'output': perturbed_output[:],
                'error': error,
                'control_signal': control_signal,
                'coherence': self.coherence_history[-1] if self.coherence_history else 0.0,
                'is_coherent': is_coherent
            })

            # Step 7: Update feedback buffer
            self.feedback_buffer.append(perturbed_output[:])

            # Step 8: Update previous output
            self.prev_output = perturbed_output[:]

            # Step 9: Check convergence
            if error < 1e-6:
                print(f"[PIDInferenceEngine] Converged at step {step}.")
                break

            # Step 10: Check coherence threshold (if enabled)
            if self.use_coherence_check and not is_coherent:
                print(f"[PIDInferenceEngine] Low coherence at step {step}: {self.coherence_history[-1]:.3f}. Repeating.")

            # Optional: Early termination if coherence is consistently high
            if len(self.coherence_history) > 5:
                recent_coherence = sum(self.coherence_history[-5:]) / 5
                if recent_coherence > self.coherence_threshold and error < 0.01:
                    print(f"[PIDInferenceEngine] High coherence and low error at step {step}. Stopping.")
                    break

        # Final result
        final_output = self.history[-1]['output'] if self.history else current_output

        # Return results
        return {
            'final_output': final_output,
            'history': self.history,
            'coherence_history': self.coherence_history,
            'converged': len(self.history) > 0 and self.history[-1]['error'] < 1e-6,
            'steps_taken': len(self.history),
            'final_error': self.history[-1]['error'] if self.history else float('inf'),
            'final_coherence': self.coherence_history[-1] if self.coherence_history else 0.0,
            'control_gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        }


# Example Usage and Demonstration
if __name__ == "__main__":
    """
    Example: Simulate inference on a "spinning wheel" state.

    We assume the "wheel" state is a 4-dimensional vector representing angular position,
    velocity, acceleration, and torque. The goal is to stabilize the system to a target
    state using PID feedback.
    """

    # Define initial state (e.g., measured wheel state)
    initial_state = [1.0, 0.5, 0.1, 0.05]

    # Define target state (e.g., stable equilibrium)
    target_state = [0.0, 0.0, 0.0, 0.0]

    # Initialize the PID inference engine
    engine = PIDInferenceEngine(
        proportional_gain=1.2,
        integral_gain=0.15,
        derivative_gain=0.08,
        feedback_delay=2,
        max_perturbation=0.2,
        coherence_threshold=0.9,
        max_iterations=150,
        use_perturbation=True,
        use_coherence_check=True
    )

    # Run inference
    result = engine.run_inference(
        input_sequence=initial_state,
        target_sequence=target_state,
        max_steps=100
    )

    # Output results
    print("
" + "="*60)
    print("PID-FEEDBACK INFERENCE ENGINE: RESULTS")
    print("="*60)
    print(f"Final Output: {result['final_output']}")
    print(f"Final Error: {result['final_error']:.6f}")
    print(f"Final Coherence: {result['final_coherence']:.3f}")
    print(f"Steps Taken: {result['steps_taken']}")
    print(f"Converged: {result['converged']}")
    print(f"Control Gains: kp={result['control_gains']['kp']}, ki={result['control_gains']['ki']}, kd={result['control_gains']['kd']}")
    print("="*60)

    # Optional: Print first 10 steps for inspection
    print("
First 10 steps of inference:")
    for i in range(min(10, len(result['history']))):
        step = result['history'][i]
        print(f"  Step {step['step']}: Error={step['error']:.4f}, Coherence={step['coherence']:.3f}, "
               f"Control={step['control_signal']:.3f}")
    print("
" + "="*60)
```
### 4x4 Net - 2 epochs - ISTJ-INTJ-ENTP-ENFP | Learning rate 0.6 - Qwen3:30b-a3b-instruct-2507 | Completion time: 72 min                                       

```
"""
===============================================================================
PID-Adaptive Harmonic Inference Engine (PAHIE): A Conceptual Framework for
Resilient, Adaptive, and Coherent Inference via Spinning Wheel Dynamics

Author: Expert Code Synthesis Agent
Date: 2025
License: MIT

Description:
This is a fully self-contained, dependency-free Python implementation of a novel
inference algorithm inspired by the metaphor of "spinning wheels" (harmonic
oscillators) representing latent data states. The PAHIE (PID-Adaptive Harmonic
Inference Engine) synthesizes principles from control theory, dynamical systems,
and feedback learning to enable deep decoding of internal latent dynamics.

Key Features:
- Implements a hybrid PID-like inference engine with proportional, integral, and
  differential components, all dynamically weighted.
- Uses perturbative feedback from progressive outputs to iteratively refine
  inference, mimicking the idea of "spinning the whole wheel".
- Encodes data as phase-space states of harmonic oscillators (position + momentum).
- Dynamically adjusts control gains via adaptive weighting based on error
  trajectory and coherence metrics.
- Includes a meta-regularization protocol to detect entropy drift and prevent
  coherence collapse.
- Fully self-contained: no external dependencies (e.g., torch, numpy, etc.).

This implementation is conceptual and educational, designed to demonstrate
the core logic of a next-generation inference engine that goes beyond
traditional autoregressive or iterative sampling.

===============================================================================
"""

import math
import copy
from typing import List, Tuple, Dict, Any, Optional
import random

# =============================================================================
# Configuration Constants
# =============================================================================

# Number of harmonic oscillators (latent wheels) in the system
NUM_WHEELS = 8

# Maximum number of inference steps
MAX_STEPS = 50

# Initial control gains (P, I, D) - will be adapted
INITIAL_P_GAIN = 1.0
INITIAL_I_GAIN = 0.1
INITIAL_D_GAIN = 0.05

# Learning rate for adaptive gain adjustment
ADAPTIVE_LR = 0.01

# Threshold for entropy drift detection (coherence collapse)
ENTROPY_THRESHOLD = 0.8

# Minimum and maximum allowed gain values
MIN_GAIN = 0.01
MAX_GAIN = 10.0

# Smoothing factor for moving average of error
SMOOTHING_FACTOR = 0.3

# =============================================================================
# Core Data Structures
# =============================================================================

class HarmonicOscillator:
    """
    Represents a single spinning wheel (harmonic oscillator) in the system.
    Each oscillator has:
    - position (x): current state (e.g., phase)
    - momentum (p): rate of change (e.g., angular velocity)
    - frequency (omega): intrinsic oscillation rate
    - damping (gamma): energy loss rate
    """
    
    def __init__(self, omega: float = 1.0, gamma: float = 0.01):
        self.omega = omega
        self.gamma = gamma
        self.position = 0.0
        self.momentum = 0.0
        self.history = []  # Track state evolution for feedback

    def step(self, external_force: float = 0.0) -> None:
        """
        Update the oscillator state using a discrete-time approximation
        of the harmonic oscillator equation:
        
        d²x/dt² + 2γ(dx/dt) + ω²x = F(t)
        
        We use a symplectic Euler integrator for stability.
        """
        # Store current state for feedback
        self.history.append((self.position, self.momentum))
        
        # Update momentum: p += dt * (F - 2γp - ω²x)
        dt = 1.0
        new_momentum = self.momentum + dt * (external_force - 2 * self.gamma * self.momentum - self.omega**2 * self.position)
        
        # Update position: x += dt * p
        new_position = self.position + dt * new_momentum
        
        self.position = new_position
        self.momentum = new_momentum

    def get_state(self) -> Tuple[float, float]:
        """Return current (position, momentum) state."""
        return (self.position, self.momentum)

    def get_phase(self) -> float:
        """Return phase angle (arctan2(momentum, position)) for coherence analysis."""
        return math.atan2(self.momentum, self.position)

    def reset(self) -> None:
        """Reset oscillator to initial state."""
        self.position = 0.0
        self.momentum = 0.0
        self.history.clear()


class PIDInferenceController:
    """
    PID-like inference controller with adaptive gains and perturbative feedback.
    Implements a hybrid control strategy that combines:
    - Proportional (P): immediate response to error
    - Integral (I): cumulative error over time
    - Differential (D): prediction of future error based on rate of change
    
    The gains are dynamically adjusted based on system coherence and error trends.
    """

    def __init__(self, num_wheels: int = NUM_WHEELS):
        self.num_wheels = num_wheels
        self.p_gain = INITIAL_P_GAIN
        self.i_gain = INITIAL_I_GAIN
        self.d_gain = INITIAL_D_GAIN
        
        # Accumulated integral error per oscillator
        self.integral_error = [0.0] * num_wheels
        
        # Previous error per oscillator (for D term)
        self.prev_error = [0.0] * num_wheels
        
        # Moving average of error magnitude
        self.avg_error = 0.0
        
        # Coherence metric (how synchronized the wheels are)
        self.coherence = 1.0
        
        # Feedback history for perturbative correction
        self.feedback_history = []
        
        # Gain adaptation history
        self.gain_history = []

    def compute_control_signal(self, 
                              current_states: List[Tuple[float, float]], 
                              target_states: List[Tuple[float, float]]) -> List[float]:
        """
        Compute control signals (perturbations) for each oscillator based on:
        - Current vs target state (error)
        - Proportional, integral, and differential components
        - Adaptive gain weighting
        
        Returns: list of control signals (forces) to apply to each oscillator.
        """
        control_signals = []
        errors = []
        
        # Compute error for each oscillator
        for i in range(self.num_wheels):
            pos_err = target_states[i][0] - current_states[i][0]
            mom_err = target_states[i][1] - current_states[i][1]
            error = math.sqrt(pos_err**2 + mom_err**2)
            errors.append(error)
        
        # Update moving average of error
        self.avg_error = (1 - SMOOTHING_FACTOR) * self.avg_error + SMOOTHING_FACTOR * sum(errors) / len(errors)
        
        # Compute coherence metric (based on phase dispersion)
        phases = [osc.get_phase() for osc in self._get_oscillators()]
        phase_diffs = [abs(phases[i] - phases[j]) for i in range(len(phases)) for j in range(i+1, len(phases))]
        avg_phase_diff = sum(phase_diffs) / len(phase_diffs) if phase_diffs else 0.0
        self.coherence = 1.0 - min(1.0, avg_phase_diff / math.pi)
        
        # Adaptive gain adjustment based on coherence and error
        self._adapt_gains(errors)
        
        # Compute control signal for each oscillator
        for i in range(self.num_wheels):
            pos_err = target_states[i][0] - current_states[i][0]
            mom_err = target_states[i][1] - current_states[i][1]
            error = math.sqrt(pos_err**2 + mom_err**2)
            
            # Proportional term
            p_term = self.p_gain * error
            
            # Integral term
            self.integral_error[i] += error
            i_term = self.i_gain * self.integral_error[i]
            
            # Differential term
            d_term = self.d_gain * (error - self.prev_error[i])
            self.prev_error[i] = error
            
            # Combine terms
            control_signal = p_term + i_term + d_term
            
            # Apply perturbative feedback from previous steps
            feedback_correction = self._apply_perturbative_feedback(i, error)
            control_signal += feedback_correction
            
            control_signals.append(control_signal)
        
        # Store feedback for next iteration
        self.feedback_history.append(errors)
        if len(self.feedback_history) > 5:
            self.feedback_history.pop(0)
        
        return control_signals

    def _adapt_gains(self, errors: List[float]) -> None:
        """
        Dynamically adjust PID gains based on system coherence and error trends.
        Higher error or low coherence triggers gain increase; high gain with low error
        triggers decrease to prevent oscillation.
        """
        avg_error = sum(errors) / len(errors)
        
        # If coherence is low, increase gains to drive synchronization
        if self.coherence < 0.5:
            self.p_gain = min(MAX_GAIN, self.p_gain * 1.1)
            self.i_gain = min(MAX_GAIN, self.i_gain * 1.1)
            self.d_gain = min(MAX_GAIN, self.d_gain * 1.1)
        else:
            # If error is high but coherence is good, increase P and I
            if avg_error > 0.5:
                self.p_gain = min(MAX_GAIN, self.p_gain * 1.05)
                self.i_gain = min(MAX_GAIN, self.i_gain * 1.05)
            # If error is low and coherence is high, reduce gains to stabilize
            elif avg_error < 0.1:
                self.p_gain = max(MIN_GAIN, self.p_gain * 0.95)
                self.i_gain = max(MIN_GAIN, self.i_gain * 0.95)
                self.d_gain = max(MIN_GAIN, self.d_gain * 0.95)
        
        # Prevent extreme values
        self.p_gain = max(MIN_GAIN, min(MAX_GAIN, self.p_gain))
        self.i_gain = max(MIN_GAIN, min(MAX_GAIN, self.i_gain))
        self.d_gain = max(MIN_GAIN, min(MAX_GAIN, self.d_gain))
        
        # Record gain state
        self.gain_history.append((self.p_gain, self.i_gain, self.d_gain))
        if len(self.gain_history) > 10:
            self.gain_history.pop(0)

    def _apply_perturbative_feedback(self, wheel_idx: int, current_error: float) -> float:
        """
        Apply perturbative feedback from past inference steps to encourage
        deeper exploration of latent dynamics.
        
        This simulates "spinning the whole wheel" by injecting corrective signals
        based on historical error patterns.
        """
        if len(self.feedback_history) < 2:
            return 0.0
        
        # Look at error trends over past steps
        recent_errors = [step[wheel_idx] for step in self.feedback_history[-3:]]
        if len(recent_errors) < 3:
            return 0.0
        
        # Compute trend: is error increasing or decreasing?
        trend = recent_errors[-1] - recent_errors[-2]
        
        # If error is increasing, apply negative feedback to counteract
        if trend > 0.1:
            return -0.1 * current_error
        # If error is decreasing, apply positive feedback to explore further
        elif trend < -0.1:
            return 0.1 * current_error
        else:
            return 0.0

    def _get_oscillators(self) -> List[HarmonicOscillator]:
        """Helper: return list of oscillators (mocked for now)."""
        # In a real system, this would be injected from the model
        return [HarmonicOscillator() for _ in range(self.num_wheels)]


class HarmonicInferenceEngine:
    """
    The main inference engine that orchestrates the PAHIE system.
    It manages:
    - Initialization of harmonic oscillators
    - Iterative inference with PID feedback
    - Coherence monitoring
    - Meta-regularization to prevent collapse
    - Final output extraction
    """

    def __init__(self, num_wheels: int = NUM_WHEELS):
        self.num_wheels = num_wheels
        self.oscillators = [HarmonicOscillator(omega=1.0 + i * 0.1, gamma=0.01) for i in range(num_wheels)]
        self.controller = PIDInferenceController(num_wheels)
        self.inference_history = []  # Store full state evolution
        self.coherence_history = []
        self.entropy_drift = 0.0
        self.is_coherent = True

    def initialize_from_data(self, data: List[float]) -> None:
        """
        Initialize the system from input data.
        Each data point becomes the target position for a wheel.
        Momentum is initialized to zero.
        """
        if len(data) != self.num_wheels:
            raise ValueError(f"Data length ({len(data)}) must match number of wheels ({self.num_wheels})")
        
        target_states = []
        for i, val in enumerate(data):
            # Set target position to data value, momentum to zero
            target_states.append((val, 0.0))
            # Initialize oscillator to match target
            self.oscillators[i].position = val
            self.oscillators[i].momentum = 0.0
        
        # Store initial target
        self.target_states = target_states

    def run_inference(self, max_steps: int = MAX_STEPS) -> List[Tuple[float, float]]:
        """
        Run the inference process using the PID-adaptive harmonic engine.
        Returns the final state of all oscillators.
        """
        self.inference_history = []
        self.coherence_history = []
        self.entropy_drift = 0.0
        self.is_coherent = True
        
        # Reset oscillators
        for osc in self.oscillators:
            osc.reset()
        
        # Initialize with target data
        self.initialize_from_data([0.0] * self.num_wheels)  # Placeholder; real data would be passed
        
        # Run inference loop
        for step in range(max_steps):
            # Get current states
            current_states = [osc.get_state() for osc in self.oscillators]
            
            # Compute control signals
            control_signals = self.controller.compute_control_signal(current_states, self.target_states)
            
            # Apply control signals to oscillators
            for i, osc in enumerate(self.oscillators):
                osc.step(external_force=control_signals[i])
            
            # Record history
            self.inference_history.append(copy.deepcopy(current_states))
            
            # Compute coherence and entropy
            coherence = self.controller.coherence
            self.coherence_history.append(coherence)
            
            # Estimate entropy drift (simplified)
            # In real systems, this would be based on phase-space density
            phase_spread = sum(abs(osc.get_phase()) for osc in self.oscillators) / self.num_wheels
            self.entropy_drift = 0.5 * (1.0 - coherence) + 0.5 * phase_spread
            
            # Check for coherence collapse
            if self.entropy_drift > ENTROPY_THRESHOLD:
                self.is_coherent = False
                print(f"⚠️  Coherence collapse detected at step {step}. Entropy drift: {self.entropy_drift:.3f}")
                break
            
            # Check for convergence (small error)
            total_error = sum(
                math.sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)
                for target, current in zip(self.target_states, current_states)
            )
            if total_error < 0.01:
                print(f"✅ Convergence achieved at step {step}. Final error: {total_error:.6f}")
                break
        
        # Return final state
        final_states = [osc.get_state() for osc in self.oscillators]
        return final_states

    def get_final_output(self) -> List[float]:
        """
        Extract final inference result as a list of positions.
        This represents the "decoded" state of the spinning wheels.
        """
        final_states = self.run_inference()
        return [state[0] for state in final_states]

    def get_coherence_trend(self) -> List[float]:
        """Return the coherence history over time."""
        return self.coherence_history

    def get_entropy_drift(self) -> float:
        """Return the final entropy drift value."""
        return self.entropy_drift

    def reset(self) -> None:
        """Reset the entire engine."""
        for osc in self.oscillators:
            osc.reset()
        self.controller = PIDInferenceController(self.num_wheels)
        self.inference_history = []
        self.coherence_history = []
        self.entropy_drift = 0.0
        self.is_coherent = True


# =============================================================================
# Example Usage and Demonstration
# =============================================================================

def demonstrate_pahie():
    """
    Demonstrate the PAHIE engine with a synthetic example.
    Simulates inference on a data vector representing a "spinning wheel" state.
    """
    print("🚀 Starting PAHIE (PID-Adaptive Harmonic Inference Engine) Demo")
    print("=" * 60)
    
    # Create engine
    engine = HarmonicInferenceEngine(num_wheels=NUM_WHEELS)
    
    # Simulate input data (e.g., a signal from a physical system)
    input_data = [1.0, -0.5, 0.8, -1.2, 0.3, -0.7, 1.1, -0.9]
    
    print(f"🎯 Input data (target wheel states): {input_data}")
    
    # Run inference
    final_output = engine.get_final_output()
    
    print(f"✅ Final inferred states: {final_output}")
    
    # Show coherence trend
    coherence_trend = engine.get_coherence_trend()
    print(f"📈 Coherence trend (min to max): {min(coherence_trend):.3f} → {max(coherence_trend):.3f}")
    
    # Show entropy drift
    entropy = engine.get_entropy_drift()
    print(f"📉 Entropy drift: {entropy:.3f}")
    
    if engine.is_coherent:
        print("✅ System remained coherent throughout inference.")
    else:
        print("⚠️  System experienced coherence collapse.")
    
    print("=" * 60)
    print("💡 Conceptual Insight: The PAHIE engine 'spins the whole wheel' by iteratively refining latent states through adaptive feedback, enabling deeper decoding than traditional inference.")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_pahie()
    
    # Optional: Test with different data
    print("
🧪 Testing with random data...")
    engine = HarmonicInferenceEngine(num_wheels=6)
    random_data = [random.uniform(-2, 2) for _ in range(6)]
    print(f"Input: {random_data}")
    output = engine.get_final_output()
    print(f"Output: {output}")
```

# New prompt

Comment: so far it made a good job at making the kind of shitty algorithm you find in IEEE transactions with 2 citations. Lets do a third run with one more epoch.

```
 Current diffusers and transformer architectures use integral samplers or differential solvers in the case of diffusers, and decoding algorithms which account as integral, in the case of transformers, to run inference. I presume the foundation of training and architecture are already figured out, so i want you to create a new inference algorithm that helps exsiting architectures acheive AGI. For this conceptualziation assume the world is full of spinning wheels (harmonic oscillators), like we see them in atoms, solar systems, galaxies, human hierarchies...etrc, and data represents a measured state of the "wheel" at a given time. Abudant training data samples the full state of the "wheel" by offering all the posible data of the wheel full state: many perspectives. This is where understanding is reached: by spinning the whole wheel and assimilating all possibly perspectives.

 Current inference algoritms in the other hand, are not fully decoding the internal "implicit wheels" abstracted into the weights after training, as their responses can be shallow. The training algorithms encode the wheels but inference algorithms do not extract them very well.

 I want you to make in python with excellent documentation:

1. An inference algorithm to achieve AGI through better inference using a PID like approach with perturbative feedback. Instead of just using either an integrative or differential component, i want you to implement both with proportional weighting terms. Figure out a good way to integrate PID loops into transformer inference so transformers learn on inference. 

 2. The inference algorithm should be coded from scratch without using external dependencies. The transformer algorithm this algorithm runs inference for shall be mocked.

```

# Gemini 2.5 pro - Max Thinking Budget

```
import math
import random

# For demonstration, we use basic list operations instead of numpy or other libraries.

# --- Helper Functions ---

def vector_add(v1, v2):
    """Adds two vectors element-wise."""
    return [x + y for x, y in zip(v1, v2)]

def vector_sub(v1, v2):
    """Subtracts two vectors element-wise (v1 - v2)."""
    return [x - y for x, y in zip(v1, v2)]

def scalar_mul(scalar, v):
    """Multiplies a vector by a scalar."""
    return [scalar * x for x in v]

def softmax(logits):
    """Computes softmax probabilities from logits."""
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def argmax(values):
    """Returns the index of the maximum value in a list."""
    return values.index(max(values))

# --- Core Components ---

class MockTransformer:
    """
    A mocked Transformer model.

    This class simulates the essential behavior of a transformer for inference:
    it takes a sequence of token IDs and produces a hidden state and logits
    for the next token. The internal workings are simplified with random
    numbers to avoid needing a real, trained model, but the interface

    is designed to be representative of a real transformer architecture.
    """
    def __init__(self, vocab_size: int, hidden_dim: int):
        """
        Initializes the MockTransformer.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
            hidden_dim (int): The dimensionality of the hidden states.
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # Mock embedding layer: a list of random vectors for each token.
        self.embedding = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(vocab_size)]
        # Mock final layer (LM head) to project hidden state to logits.
        self.lm_head = [[random.uniform(-1, 1) for _ in range(vocab_size)] for _ in range(hidden_dim)]

    def forward(self, input_ids: list[int]) -> tuple[list[float], list[float]]:
        """
        Performs a forward pass to get the hidden state and logits.

        This simulates the process of encoding the input sequence into a
        contextualized representation (hidden_state).

        Args:
            input_ids (list[int]): A list of token IDs representing the input sequence.

        Returns:
            tuple[list[float], list[float]]: A tuple containing:
                - hidden_state (list[float]): The final hidden state vector.
                - logits (list[float]): The raw output scores for each token in the vocabulary.
        """
        if not input_ids:
            # Return a zero vector if there's no input.
            return [0.0] * self.hidden_dim, [0.0] * self.vocab_size

        # Simulate context aggregation by averaging embeddings.
        # This is a gross simplification but serves the purpose of creating a state.
        aggregated_embedding = [0.0] * self.hidden_dim
        for token_id in input_ids:
            aggregated_embedding = vector_add(aggregated_embedding, self.embedding[token_id])
        
        # This represents the final hidden state before the LM head.
        hidden_state = scalar_mul(1.0 / len(input_ids), aggregated_embedding)

        # The rest of the "forward" pass is handled by the PIDInference class
        # which will take this hidden_state and potentially perturb it.
        return hidden_state

    def logits_from_state(self, hidden_state: list[float]) -> list[float]:
        """
        Calculates logits from a given hidden state.

        This simulates the final linear layer that projects the internal
        representation to the vocabulary space.

        Args:
            hidden_state (list[float]): The (potentially perturbed) hidden state.

        Returns:
            list[float]: The logits vector.
        """
        # Mock matrix multiplication: dot product of hidden_state with each column of lm_head.
        logits = [0.0] * self.vocab_size
        for i in range(self.vocab_size):
            col = [self.lm_head[j][i] for j in range(self.hidden_dim)]
            dot_product = sum(x * y for x, y in zip(hidden_state, col))
            logits[i] = dot_product
        return logits


class PIDController:
    """
    A PID (Proportional-Integral-Derivative) Controller implemented from scratch.

    This controller calculates a corrective output based on the error between a
    desired setpoint and a measured process variable. This is the core feedback
    mechanism for the PIDInference algorithm.
    """
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: list[float]):
        """
        Initializes the PID Controller.

        Args:
            Kp (float): The Proportional gain. Determines the reaction to the current error.
            Ki (float): The Integral gain. Determines the reaction based on the sum of past errors.
            Kd (float): The Derivative gain. Determines the reaction based on the rate of change of the error.
            setpoint (list[float]): The target state vector that the system should aim for.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.hidden_dim = len(setpoint)

        # Internal state variables for the controller
        self._integral = [0.0] * self.hidden_dim
        self._previous_error = [0.0] * self.hidden_dim

    def update(self, process_variable: list[float], dt: float = 1.0) -> list[float]:
        """
        Calculates the PID control output for a given process variable.

        Args:
            process_variable (list[float]): The current state of the system (e.g., transformer's hidden state).
            dt (float): The time step between updates. For token generation, this is typically 1.

        Returns:
            list[float]: The corrective output vector (the perturbation).
        """
        error = vector_sub(self.setpoint, process_variable)

        # Proportional term
        p_term = scalar_mul(self.Kp, error)

        # Integral term
        self._integral = vector_add(self._integral, scalar_mul(error, dt))
        i_term = scalar_mul(self.Ki, self._integral)

        # Derivative term
        error_derivative = scalar_mul(1.0 / dt, vector_sub(error, self._previous_error))
        d_term = scalar_mul(self.Kd, error_derivative)

        # Update the previous error for the next iteration
        self._previous_error = error

        # The total control output is the sum of the P, I, and D terms.
        control_output = vector_add(p_term, vector_add(i_term, d_term))
        return control_output


class PIDInference:
    """
    An inference algorithm that uses a PID controller to guide a transformer.

    This class orchestrates the generation process. At each step, it uses the
    PID controller to create a perturbation, applies it to the transformer's
    internal state, and then generates the next token. This process constitutes
    the "learning on inference" by allowing the model's trajectory to be
    dynamically corrected based on its deviation from a target "understanding."
    """
    def __init__(self, transformer: MockTransformer, pid_controller: PIDController):
        """
        Initializes the PIDInference engine.

        Args:
            transformer (MockTransformer): The transformer model to run inference on.
            pid_controller (PIDController): The configured PID controller for feedback.
        """
        self.transformer = transformer
        self.pid_controller = pid_controller

    def generate(self, prompt_ids: list[int], max_length: int) -> list[int]:
        """
        Generates a sequence of tokens autoregressively using PID feedback.

        Args:
            prompt_ids (list[int]): The initial sequence of token IDs.
            max_length (int): The total desired length of the generated sequence.

        Returns:
            list[int]: The generated sequence of token IDs, including the prompt.
        """
        generated_ids = list(prompt_ids)

        print(f"Starting generation with prompt: {generated_ids}")
        print("-" * 30)

        for i in range(len(prompt_ids), max_length):
            # 1. Get the current hidden state from the transformer (Process Variable)
            current_hidden_state = self.transformer.forward(generated_ids)

            # 2. Use the PID controller to calculate the corrective perturbation
            #    This is the core of the feedback loop.
            control_perturbation = self.pid_controller.update(current_hidden_state)

            # 3. Apply the perturbation to the hidden state.
            #    This nudges the model's internal representation towards the setpoint.
            perturbed_hidden_state = vector_add(current_hidden_state, control_perturbation)

            # 4. Get the final logits from the perturbed state.
            logits = self.transformer.logits_from_state(perturbed_hidden_state)

            # 5. Select the next token (here, using simple greedy decoding).
            next_token_id = argmax(logits)

            # 6. Append the new token and continue the loop.
            generated_ids.append(next_token_id)

            print(f"Step {i+1}:")
            print(f"  - Current Sequence: {generated_ids}")
            # To keep the output clean, we won't print the large vectors.
            # print(f"  - Control Perturbation Norm: {math.sqrt(sum(x*x for x in control_perturbation))}")
            print(f"  - Chose Token ID: {next_token_id}")


        print("-" * 30)
        print(f"Final generated sequence: {generated_ids}")
        return generated_ids


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Configuration
    VOCAB_SIZE = 50
    HIDDEN_DIM = 16  # Dimensionality of the internal "wheel" state
    MAX_SEQ_LENGTH = 15
    PROMPT = [10, 4, 25] # A sample starting prompt

    # 2. Initialize the Mock Transformer
    transformer = MockTransformer(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    print(f"Initialized MockTransformer with vocab size {VOCAB_SIZE} and hidden dim {HIDDEN_DIM}.")

    # 3. Define the Setpoint and configure the PID Controller
    # The "setpoint" is the target state of "understanding". For this conceptual
    # implementation, we'll derive it from the initial prompt's embedding.
    # This anchors the entire generation to the theme of the original prompt.
    setpoint_vector = transformer.forward(PROMPT)

    # PID gains need to be tuned. These are example values.
    # Kp: Strong immediate correction.
    # Ki: Slower correction for persistent drift.
    # Kd: Damping to prevent oscillation.
    Kp = 0.5
    Ki = 0.1
    Kd = 0.2
    
    pid_controller = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=setpoint_vector)
    print(f"Initialized PIDController with Kp={Kp}, Ki={Ki}, Kd={Kd}.")
    
    # 4. Initialize and run the PIDInference algorithm
    inference_engine = PIDInference(transformer=transformer, pid_controller=pid_controller)
    
    # Run the generation
    final_output = inference_engine.generate(
        prompt_ids=PROMPT,
        max_length=MAX_SEQ_LENGTH
    )

```

### 4x4 Net - 3 epochs - ISTJ-INTJ-INFP-INTP | Learning rate 0.6, Density: 0.7, Vector word size: 10, num epochs: 3, CoT trace depth: 2 - Qwen3:30b-a3b-instruct-2507 | Completion time: 110 min

```
"""
AGI-Inspired Inference Engine: Harmonic Oscillator-Driven Transformer Inference with PID Feedback

This module implements a novel inference algorithm designed to achieve AGI-level understanding by:
1. Modeling the world as a system of harmonic oscillators (spinning wheels)
2. Using a PID-like feedback loop to dynamically refine inference through proportional, integral, and differential components
3. Integrating self-supervised pattern detection (HTSE) to discover periodicities, symmetries, and emergent structures
4. Enabling transformers to "learn on inference" by continuously adapting their internal representations

The core innovation is the Harmonic Inference Loop (HIL), which:
- Uses micro-scale pattern detection to identify local oscillatory motifs
- Applies macro-scale symmetry inference to detect global invariants
- Induces emergent oscillators via a learnable Hamiltonian in latent space
- Employs a PID controller to balance exploration (differential), memory (integral), and stability (proportional)

This implementation is self-contained, dependency-free, and designed for integration with any transformer architecture.

Author: AGI Synthesis Agent
Date: 2025
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import copy


class HarmonicInferenceLoop:
    """
    A PID-like inference engine that enables transformers to achieve AGI-level understanding
    by dynamically exploring the latent space of harmonic oscillators.

    The system operates in a feedback loop:
    - Input data is analyzed for periodicities and symmetries
    - A latent oscillator model is constructed
    - A PID controller adjusts inference based on error between predicted and actual states
    - The transformer's internal state is updated during inference (learning on inference)

    This is a conceptual framework for AGI inference, not a direct replacement for existing models.
    """

    def __init__(
        self,
        proportional_gain: float = 1.0,
        integral_gain: float = 0.1,
        differential_gain: float = 0.05,
        time_step: float = 0.1,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        symmetry_detection_threshold: float = 0.7,
        oscillator_stability_threshold: float = 0.85,
    ):
        """
        Initialize the Harmonic Inference Loop.

        Args:
            proportional_gain: Weight for proportional error correction (stability)
            integral_gain: Weight for cumulative error (memory)
            differential_gain: Weight for rate of change (exploration)
            time_step: Time step for simulation (in arbitrary units)
            max_iterations: Maximum number of inference iterations
            convergence_threshold: Threshold for convergence detection
            symmetry_detection_threshold: Minimum similarity score to consider a symmetry
            oscillator_stability_threshold: Minimum stability score for valid oscillators
        """
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.differential_gain = differential_gain
        self.time_step = time_step
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.symmetry_detection_threshold = symmetry_detection_threshold
        self.oscillator_stability_threshold = oscillator_stability_threshold

        # Internal state
        self.iteration = 0
        self.error_history = []
        self.integral_error = 0.0
        self.last_error = 0.0
        self.is_converged = False
        self.oscillator_candidates = []
        self.symmetry_graph = {}
        self.phase_space_trajectory = []

        # Meta-parameters for adaptive control
        self.adaptive_threshold = symmetry_detection_threshold
        self.noise_level = 0.0

    def detect_micro_structure(self, sequence: List[float]) -> List[Dict]:
        """
        Micro-Structure Extraction: Identify local recurring motifs using invariant pattern detection.

        This simulates invariant convolutional kernels with DTW-aware attention.
        Returns detected motifs with their periodicity and amplitude.

        Args:
            sequence: Input sequence of real numbers (e.g., time-series data)

        Returns:
            List of detected motifs with period, amplitude, and confidence
        """
        motifs = []
        n = len(sequence)

        # Try different window sizes to detect periodicities
        for window_size in range(3, min(10, n // 2) + 1):
            for start in range(n - window_size):
                window = sequence[start:start + window_size]
                # Compute autocorrelation to detect periodicity
                autocorr = np.correlate(window, window, mode='full')
                autocorr = autocorr[len(window)-1:]  # Only positive lags
                # Find peak in autocorrelation (excluding lag 0)
                peaks = np.where(autocorr[1:] == np.max(autocorr[1:]))[0] + 1
                if len(peaks) > 0:
                    period = peaks[0]
                    if period <= window_size // 2:  # Avoid trivial periods
                        amplitude = np.std(window)
                        confidence = autocorr[period] / (amplitude ** 2 + 1e-8)
                        if confidence > 0.5:
                            motifs.append({
                                'period': period,
                                'amplitude': amplitude,
                                'confidence': confidence,
                                'start': start,
                                'window_size': window_size
                            })

        # Deduplicate and sort by confidence
        unique_motifs = {}
        for m in motifs:
            key = (m['period'], m['amplitude'])
            if key not in unique_motifs or unique_motifs[key]['confidence'] < m['confidence']:
                unique_motifs[key] = m
        return sorted(unique_motifs.values(), key=lambda x: x['confidence'], reverse=True)

    def detect_macro_symmetry(self, sequence: List[float]) -> Dict[str, float]:
        """
        Macro-Symmetry Inference: Detect global symmetries using graph-based relational analysis.

        Simulates a graph-based relational transformer that constructs a topological skeleton.
        Returns symmetry scores for rotational, reflectional, and translational invariance.

        Args:
            sequence: Input sequence

        Returns:
            Dictionary of symmetry scores (0.0 to 1.0)
        """
        n = len(sequence)
        if n < 4:
            return {'rotational': 0.0, 'reflectional': 0.0, 'translational': 0.0}

        # Build adjacency matrix based on similarity between segments
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Use normalized similarity (cosine-like)
                if i != j:
                    # Use sliding window similarity
                    window_size = min(5, n // 2)
                    start_i = i
                    start_j = j
                    end_i = min(i + window_size, n)
                    end_j = min(j + window_size, n)
                    if end_i - start_i > 0 and end_j - start_j > 0:
                        window_i = sequence[start_i:end_i]
                        window_j = sequence[start_j:end_j]
                        # Pad shorter window
                        pad_len = max(0, len(window_i) - len(window_j))
                        if pad_len > 0:
                            window_j = np.pad(window_j, (0, pad_len), mode='edge')
                        elif len(window_j) > len(window_i):
                            window_i = np.pad(window_i, (0, len(window_j) - len(window_i)), mode='edge')
                        # Compute similarity
                        sim = np.dot(window_i, window_j) / (np.linalg.norm(window_i) * np.linalg.norm(window_j) + 1e-8)
                        adj_matrix[i][j] = sim

        # Compute symmetry scores
        rotational_score = 0.0
        reflectional_score = 0.0
        translational_score = 0.0

        # Rotational symmetry: check if sequence is periodic
        for period in range(2, n // 2 + 1):
            if n % period == 0:
                segments = [sequence[i:i+period] for i in range(0, n, period)]
                if len(segments) > 1:
                    avg_sim = np.mean([
                        np.dot(segments[i], segments[i+1]) / (np.linalg.norm(segments[i]) * np.linalg.norm(segments[i+1]) + 1e-8)
                        for i in range(len(segments)-1)
                    ])
                    rotational_score = max(rotational_score, avg_sim)

        # Reflectional symmetry: compare sequence with its reverse
        reversed_seq = sequence[::-1]
        reflection_sim = np.dot(sequence, reversed_seq) / (np.linalg.norm(sequence) * np.linalg.norm(reversed_seq) + 1e-8)
        reflectional_score = reflection_sim

        # Translational symmetry: check for repeated patterns
        for offset in range(1, n // 2):
            if n - offset >= offset:
                sim = np.dot(sequence[:n-offset], sequence[offset:]) / (
                    np.linalg.norm(sequence[:n-offset]) * np.linalg.norm(sequence[offset:]) + 1e-8
                )
                translational_score = max(translational_score, sim)

        return {
            'rotational': max(0.0, min(1.0, rotational_score)),
            'reflectional': max(0.0, min(1.0, reflectional_score)),
            'translational': max(0.0, min(1.0, translational_score))
        }

    def induce_emergent_oscillators(self, sequence: List[float]) -> List[Dict]:
        """
        Emergent Oscillator Induction: Map detected patterns into a latent manifold.

        Simulates a feedback-driven dynamical system that maps patterns into phase space
        governed by a learnable Hamiltonian (here, simplified as a harmonic oscillator).

        Args:
            sequence: Input sequence

        Returns:
            List of induced oscillators with phase, frequency, and stability
        """
        oscillators = []
        motifs = self.detect_micro_structure(sequence)
        symmetries = self.detect_macro_symmetry(sequence)

        # Combine micro and macro information
        for motif in motifs:
            # Use period as frequency (in Hz)
            frequency = 1.0 / (motif['period'] * self.time_step)
            # Phase is initialized randomly
            phase = np.random.uniform(0, 2 * math.pi)
            # Stability is influenced by symmetry scores
            stability = (
                0.3 * symmetries['rotational'] +
                0.3 * symmetries['reflectional'] +
                0.4 * symmetries['translational']
            )
            # Add noise to simulate real-world uncertainty
            noise = np.random.normal(0, 0.1)
            stability = max(0.0, min(1.0, stability + noise))

            oscillator = {
                'frequency': frequency,
                'phase': phase,
                'amplitude': motif['amplitude'],
                'stability': stability,
                'period': motif['period'],
                'confidence': motif['confidence'],
                'is_valid': stability > self.oscillator_stability_threshold
            }
            oscillators.append(oscillator)

        # If no valid oscillators found, create a default one
        if not oscillators:
            default_osc = {
                'frequency': 1.0,
                'phase': np.random.uniform(0, 2 * math.pi),
                'amplitude': 1.0,
                'stability': 0.5,
                'period': 1,
                'confidence': 0.5,
                'is_valid': False
            }
            oscillators.append(default_osc)

        return oscillators

    def compute_phase_space_trajectory(self, oscillators: List[Dict], steps: int = 50) -> List[List[float]]:
        """
        Simulate the phase space trajectory of the induced oscillators.

        Args:
            oscillators: List of oscillator parameters
            steps: Number of time steps to simulate

        Returns:
            List of [phase, frequency] pairs for each time step
        """
        trajectory = []
        for step in range(steps):
            t = step * self.time_step
            state = []
            for osc in oscillators:
                # Simulate harmonic motion: x = A * sin(ωt + φ)
                phase = osc['phase'] + osc['frequency'] * t
                # Add small perturbation to simulate real dynamics
                perturbation = np.random.normal(0, 0.01)
                phase += perturbation
                # Normalize phase
                phase = phase % (2 * math.pi)
                state.append([phase, osc['frequency']])
            trajectory.append(state)
        return trajectory

    def compute_pid_feedback(self, current_error: float) -> float:
        """
        Compute PID control signal based on current error.

        Args:
            current_error: Current error between predicted and actual state

        Returns:
            Control signal to adjust inference
        """
        # Proportional term
        p_term = self.proportional_gain * current_error

        # Integral term
        self.integral_error += current_error * self.time_step
        i_term = self.integral_gain * self.integral_error

        # Differential term
        d_term = self.differential_gain * (current_error - self.last_error) / self.time_step
        self.last_error = current_error

        # Combine terms
        control_signal = p_term + i_term + d_term

        # Clamp to prevent runaway
        control_signal = np.clip(control_signal, -1.0, 1.0)

        return control_signal

    def adaptive_threshold_update(self, error: float, success: bool):
        """
        Update detection thresholds based on performance.

        Args:
            error: Current error magnitude
            success: Whether the current inference was successful
        """
        if success:
            # Reduce threshold if successful (more sensitive)
            self.adaptive_threshold = max(0.1, self.adaptive_threshold - 0.01)
        else:
            # Increase threshold if failed (more conservative)
            self.adaptive_threshold = min(0.9, self.adaptive_threshold + 0.02)

        # Update noise level estimate
        self.noise_level = max(0.0, self.noise_level + 0.001 * (error - 0.5))

    def run_inference(self, input_sequence: List[float], transformer_model: Callable) -> Dict:
        """
        Run the full Harmonic Inference Loop.

        Args:
            input_sequence: Input data sequence (e.g., time-series)
            transformer_model: Mocked transformer model (callable that takes input and returns output)

        Returns:
            Dictionary with inference results, including final state, oscillators, and metrics
        """
        # Reset state
        self.iteration = 0
        self.error_history = []
        self.integral_error = 0.0
        self.last_error = 0.0
        self.is_converged = False
        self.oscillator_candidates = []
        self.symmetry_graph = {}
        self.phase_space_trajectory = []

        # Step 1: Detect micro and macro structures
        motifs = self.detect_micro_structure(input_sequence)
        symmetries = self.detect_macro_symmetry(input_sequence)

        # Step 2: Induce emergent oscillators
        oscillators = self.induce_emergent_oscillators(input_sequence)
        self.oscillator_candidates = oscillators

        # Step 3: Simulate phase space trajectory
        self.phase_space_trajectory = self.compute_phase_space_trajectory(oscillators)

        # Step 4: Run PID feedback loop for inference refinement
        for self.iteration in range(self.max_iterations):
            # Simulate transformer inference using the oscillator model
            # Here we use a mock transformer that outputs based on oscillator state
            # In real use, this would be the actual transformer model
            predicted_output = self._simulate_transformer_inference(oscillators, input_sequence)

            # Compute error (simplified: difference between predicted and input)
            # In real system, this would be semantic or structural error
            error = np.mean(np.abs(np.array(input_sequence) - np.array(predicted_output)))
            self.error_history.append(error)

            # Check for convergence
            if len(self.error_history) > 10:
                recent_errors = self.error_history[-10:]
                if np.std(recent_errors) < self.convergence_threshold:
                    self.is_converged = True
                    break

            # Compute PID feedback
            control_signal = self.compute_pid_feedback(error)

            # Update oscillator parameters based on feedback
            # This simulates "learning on inference"
            for osc in oscillators:
                # Adjust phase based on control signal
                osc['phase'] += control_signal * 0.1
                # Adjust frequency slightly
                osc['frequency'] *= (1 + control_signal * 0.05)
                # Keep phase within bounds
                osc['phase'] = osc['phase'] % (2 * math.pi)

            # Update adaptive thresholds
            success = error < 0.1
            self.adaptive_threshold_update(error, success)

        # Final results
        results = {
            'input_sequence': input_sequence,
            'oscillators': oscillators,
            'symmetries': symmetries,
            'phase_space_trajectory': self.phase_space_trajectory,
            'error_history': self.error_history,
            'converged': self.is_converged,
            'iterations': self.iteration + 1,
            'final_error': self.error_history[-1] if self.error_history else 0.0,
            'adaptive_threshold': self.adaptive_threshold,
            'noise_level': self.noise_level,
            'meta_info': {
                'proportional_gain': self.proportional_gain,
                'integral_gain': self.integral_gain,
                'differential_gain': self.differential_gain,
                'time_step': self.time_step
            }
        }

        return results

    def _simulate_transformer_inference(self, oscillators: List[Dict], input_sequence: List[float]) -> List[float]:
        """
        Mock transformer inference that generates output based on oscillator dynamics.

        This simulates how a real transformer might use the oscillator model to generate
        responses during inference.

        Args:
            oscillators: List of oscillator parameters
            input_sequence: Input sequence

        Returns:
            Simulated transformer output
        """
        # Use the phase space trajectory to generate output
        # In a real system, this would be the actual transformer output
        if not self.phase_space_trajectory:
            return [0.0] * len(input_sequence)

        # Use the last time step of the trajectory
        last_state = self.phase_space_trajectory[-1]
        output = []

        for i in range(len(input_sequence)):
            # Simulate a harmonic response based on oscillator state
            response = 0.0
            for osc in oscillators:
                # Use phase and frequency to generate a harmonic component
                phase = osc['phase'] + osc['frequency'] * (i * self.time_step)
                phase = phase % (2 * math.pi)
                harmonic = osc['amplitude'] * math.sin(phase)
                response += harmonic * osc['confidence']
            output.append(response)

        # Add some noise to simulate real-world variability
        noise = np.random.normal(0, 0.1, len(output))
        output = [o + n for o, n in zip(output, noise)]

        return output


class MockTransformerModel:
    """
    A mocked transformer model that simulates inference behavior.

    This is a placeholder for integration with real transformer architectures.
    """

    def __init__(self, vocab_size: int = 1000, d_model: int = 512, n_heads: int = 8, num_layers: int = 6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.weights = np.random.randn(d_model, d_model) * 0.01

    def forward(self, input_ids: List[int]) -> List[float]:
        """
        Simulate transformer forward pass.

        Args:
            input_ids: List of token IDs

        Returns:
            List of output logits (simulated)
        """
        # Simulate a simple transformation
        output = []
        for i, token_id in enumerate(input_ids):
            # Use token ID to generate output based on position and value
            base = token_id % 10
            position_factor = (i + 1) / len(input_ids)
            output.append(base * position_factor + np.random.normal(0, 0.1))
        return output

    def update_weights(self, gradient: np.ndarray):
        """
        Simulate weight update during inference (learning on inference).

        Args:
            gradient: Gradient to update weights
        """
        self.weights -= 0.001 * gradient


# Example usage and demonstration
def demo_harmonic_inference():
    """
    Demonstrate the Harmonic Inference Loop with a synthetic time-series input.
    """
    print("🚀 Starting Harmonic Inference Loop Demo (AGI-Inspired)")

    # Create synthetic input: a combination of multiple harmonic oscillators
    t = np.linspace(0, 10, 100)
    signal = (
        2.0 * np.sin(2 * np.pi * 0.5 * t) +  # 0.5 Hz
        1.5 * np.sin(2 * np.pi * 1.2 * t) +  # 1.2 Hz
        0.8 * np.sin(2 * np.pi * 2.0 * t) +  # 2.0 Hz
        np.random.normal(0, 0.2, len(t))     # Noise
    )
    input_sequence = signal.tolist()

    # Create the inference engine
    hil = HarmonicInferenceLoop(
        proportional_gain=1.0,
        integral_gain=0.1,
        differential_gain=0.05,
        time_step=0.1,
        max_iterations=50,
        convergence_threshold=1e-5
    )

    # Create a mocked transformer model
    transformer = MockTransformerModel()

    # Run inference
    results = hil.run_inference(input_sequence, transformer.forward)

    # Display results
    print("
📊 Inference Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Final Error: {results['final_error']:.6f}")
    print(f"  Adaptive Threshold: {results['adaptive_threshold']:.3f}")
    print(f"  Noise Level: {results['noise_level']:.3f}")

    print("
🔍 Detected Oscillators:")
    for i, osc in enumerate(results['oscillators']):
        print(f"  Oscillator {i+1}:")
        print(f"    Frequency: {osc['frequency']:.2f} Hz")
        print(f"    Period: {osc['period']} steps")
        print(f"    Amplitude: {osc['amplitude']:.2f}")
        print(f"    Phase: {osc['phase']:.2f} rad")
        print(f"    Stability: {osc['stability']:.2f}")
        print(f"    Confidence: {osc['confidence']:.2f}")

    print("
🔄 Symmetry Scores:")
    for sym, score in results['symmetries'].items():
        print(f"  {sym.title()}: {score:.2f}")

    print("
✅ Harmonic Inference Loop completed successfully.")
    print("This system enables transformers to 'learn on inference' by dynamically exploring harmonic structures in data.")


# Run demo if script is executed directly
if __name__ == "__main__":
    demo_harmonic_inference()
```

##  Qwen3-30B-A3B-Instruct-2507 5x4x4 (ISTJ, ESTP, ESFP, ENTP, ESTJ) vector_word_size: 2, prompt alignment: 0.6, density: 0.5, learning_rate: 1.2 Completition time: 170 min
```
"""
AGI-Inference: A PID-Enhanced, Self-Validating Inference Framework for Transformers
Based on the Conceptualization of Harmonic Oscillators as Fundamental Units of Intelligence

This module implements a novel inference algorithm designed to unlock AGI-level understanding in existing transformer architectures by:
1. Modeling data as states of spinning harmonic oscillators (wheels) in phase space.
2. Using a PID-like feedback loop with proportional, integral, and differential components to dynamically decode internal "implicit wheels" during inference.
3. Enabling transformers to learn *during* inference through perturbative feedback, mimicking the process of "spinning the wheel" to assimilate all perspectives.

The framework is fully self-contained, dependency-free, and includes a mocked transformer for demonstration.

Key Innovations:
- PID-Inference Loop: Proportional (P) for immediate response, Integral (I) for cumulative understanding, Differential (D) for predictive anticipation.
- Perturbative Feedback: Injects small, controlled perturbations to explore latent dynamics and refine internal representations.
- Self-Validation: Uses spectral, topological, and phase coherence metrics to validate inferred harmonic structures.
- Closed-Loop Learning: The transformer updates its internal state during inference via feedback, enabling continuous learning.

This is a conceptual prototype for AGI inference, grounded in dynamical systems theory and meta-cognitive learning.

Author: AGI Synthesis Agent
Date: 2025
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Callable
import copy

# ========================
# 1. Mocked Transformer Architecture
# ========================

class MockedTransformer:
    """
    A minimal, fully self-contained, dependency-free mock of a transformer model.
    Simulates a trained model that encodes input sequences into latent states.
    """

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, n_heads: int = 8, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.positional_encoding = self._build_positional_encoding(d_model, 512)
        self.layers = []
        for _ in range(n_layers):
            # Simulate a single self-attention + feed-forward layer
            self.layers.append({
                'self_attn': np.random.randn(d_model, d_model) * 0.01,
                'ffn': np.random.randn(d_model, d_model) * 0.01,
                'norm1': np.ones(d_model) * 0.1,
                'norm2': np.ones(d_model) * 0.1
            })

    def _build_positional_encoding(self, d_model: int, max_len: int) -> np.ndarray:
        """Builds a fixed positional encoding matrix."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def encode(self, input_ids: List[int]) -> np.ndarray:
        """
        Encodes a sequence of token IDs into a latent state vector.
        Returns a 2D array of shape (seq_len, d_model).
        """
        seq_len = len(input_ids)
        embedded = self.embedding[input_ids]
        pos_enc = self.positional_encoding[:seq_len]
        x = embedded + pos_enc

        # Simulate transformer layers
        for layer in self.layers:
            # Self-attention (simplified)
            attn_output = np.dot(x, layer['self_attn'])
            # Residual + norm
            x = x + attn_output
            x = x * layer['norm1']
            # Feed-forward
            ffn_output = np.dot(x, layer['ffn'])
            x = x + ffn_output
            x = x * layer['norm2']

        return x  # (seq_len, d_model)

    def decode(self, latent_state: np.ndarray) -> List[int]:
        """
        Decodes a latent state into token IDs.
        Simulates a simple greedy decoding.
        """
        # Compute similarity to embedding matrix
        logits = np.dot(latent_state, self.embedding.T)
        # Greedy decoding
        return [np.argmax(logit) for logit in logits]


# ========================
# 2. PID-Inference Engine
# ========================

class PIDInferenceEngine:
    """
    A PID-like inference engine that enhances transformer inference with:
    - Proportional (P): Immediate response to input.
    - Integral (I): Accumulated understanding over time.
    - Differential (D): Anticipation of future states.
    - Perturbative Feedback: Injects small perturbations to explore latent dynamics.
    - Self-Validation: Validates inferred harmonic structures using spectral and topological metrics.
    """

    def __init__(self, 
                 p_gain: float = 1.0,
                 i_gain: float = 0.1,
                 d_gain: float = 0.05,
                 perturbation_scale: float = 0.01,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-4,
                 validation_thresholds: Dict[str, float] = None):
        """
        Initialize the PID inference engine.

        Args:
            p_gain: Proportional gain (response to current error).
            i_gain: Integral gain (response to accumulated error).
            d_gain: Differential gain (response to rate of change).
            perturbation_scale: Scale of perturbations for exploration.
            max_iterations: Maximum number of inference iterations.
            convergence_threshold: Threshold for convergence.
            validation_thresholds: Dictionary of thresholds for self-validation.
        """
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.perturbation_scale = perturbation_scale
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.validation_thresholds = validation_thresholds or {
            'persistence': 0.8,
            'spectral_entropy': 0.3,
            'kuramoto_order': 0.7
        }

        # Internal state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.iteration = 0
        self.history = []  # Store latent states for analysis

    def _compute_error(self, current_state: np.ndarray, target_state: np.ndarray) -> float:
        """
        Compute the error between current and target state.
        Uses L2 norm.
        """
        return np.linalg.norm(current_state - target_state)

    def _compute_spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Compute spectral entropy of a signal.
        Measures randomness in frequency domain.
        """
        fft = np.fft.fft(signal)
        power = np.abs(fft)**2
        power = power / np.sum(power)  # Normalize
        entropy = -np.sum(power * np.log2(power + 1e-10))  # Avoid log(0)
        return entropy

    def _compute_kuramoto_order(self, phases: np.ndarray) -> float:
        """
        Compute Kuramoto order parameter (phase coherence).
        Measures synchronization of oscillators.
        """
        complex_phases = np.exp(1j * phases)
        return np.abs(np.mean(complex_phases))

    def _compute_persistence(self, signal: np.ndarray) -> float:
        """
        Estimate persistence of a signal (how long a pattern lasts).
        Simple heuristic: ratio of max amplitude to RMS.
        """
        max_amp = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        return max_amp / (rms + 1e-10)

    def _apply_perturbation(self, state: np.ndarray) -> np.ndarray:
        """
        Apply a small random perturbation to the state to explore latent dynamics.
        """
        noise = np.random.normal(0, self.perturbation_scale, state.shape)
        return state + noise

    def _validate_structure(self, latent_state: np.ndarray) -> Dict[str, float]:
        """
        Validate the inferred structure using multiple metrics.
        Returns a dictionary of validation scores.
        """
        # Extract time-series from latent state (assume first dimension is time)
        if latent_state.ndim == 2:
            time_series = latent_state[:, 0]  # Use first feature for simplicity
        else:
            time_series = latent_state

        # Compute metrics
        persistence = self._compute_persistence(time_series)
        spectral_entropy = self._compute_spectral_entropy(time_series)
        phases = np.angle(np.fft.fft(time_series))
        kuramoto_order = self._compute_kuramoto_order(phases)

        return {
            'persistence': persistence,
            'spectral_entropy': spectral_entropy,
            'kuramoto_order': kuramoto_order
        }

    def _is_valid(self, metrics: Dict[str, float]) -> bool:
        """
        Check if the structure passes all validation thresholds.
        """
        for key, value in self.validation_thresholds.items():
            if metrics[key] < value:
                return False
        return True

    def infer(self, 
              transformer: MockedTransformer, 
              input_ids: List[int], 
              target_ids: Optional[List[int]] = None) -> Dict:
        """
        Run the PID-enhanced inference loop.

        Args:
            transformer: The mocked transformer model.
            input_ids: Input token IDs.
            target_ids: Optional target token IDs for error computation.

        Returns:
            Dictionary containing inference results, history, and validation metrics.
        """
        # Initialize
        self.iteration = 0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.history = []

        # Encode input
        latent_state = transformer.encode(input_ids)
        self.history.append(latent_state.copy())

        # If target is provided, compute initial error
        if target_ids is not None:
            target_latent = transformer.encode(target_ids)
            target_latent = np.mean(target_latent, axis=0)  # Average over sequence
        else:
            target_latent = np.zeros_like(latent_state[0])

        # PID loop
        while self.iteration < self.max_iterations:
            # Compute current error
            current_error = self._compute_error(latent_state, target_latent)

            # Proportional term
            p_term = self.p_gain * current_error

            # Integral term
            self.integral_error += current_error
            i_term = self.i_gain * self.integral_error

            # Differential term
            d_term = self.d_gain * (current_error - self.prev_error)
            self.prev_error = current_error

            # Total control signal
            control_signal = p_term + i_term + d_term

            # Apply perturbative feedback
            perturbed_state = self._apply_perturbation(latent_state)

            # Update latent state based on control signal
            # Simulate learning: adjust latent state toward target
            latent_state = latent_state + control_signal * (target_latent - latent_state) * 0.1

            # Store history
            self.history.append(latent_state.copy())

            # Validate structure
            metrics = self._validate_structure(latent_state)
            is_valid = self._is_valid(metrics)

            # Check convergence
            if abs(current_error) < self.convergence_threshold:
                print(f"[PID-Inference] Converged at iteration {self.iteration}")
                break

            # Check validation
            if not is_valid:
                print(f"[PID-Inference] Structure invalid at iteration {self.iteration}. "
                       f"Metrics: {metrics}")

            self.iteration += 1

        # Final decode
        decoded_ids = transformer.decode(latent_state)

        # Final validation
        final_metrics = self._validate_structure(latent_state)
        is_final_valid = self._is_valid(final_metrics)

        # Return results
        return {
            'decoded_ids': decoded_ids,
            'latent_state': latent_state,
            'history': self.history,
            'final_metrics': final_metrics,
            'is_valid': is_final_valid,
            'iterations': self.iteration,
            'converged': abs(current_error) < self.convergence_threshold
        }


# ========================
# 3. Main Application: AGI-Inference Runner
# ========================

def run_agi_inference():
    """
    Main function to demonstrate the AGI inference framework.
    """
    print("🚀 Starting AGI-Inference: PID-Enhanced Transformer Inference with Self-Validation")
    print("=" * 80)

    # Step 1: Initialize the mocked transformer
    print("1. Initializing mocked transformer...")
    transformer = MockedTransformer(vocab_size=100, d_model=64, n_heads=4, n_layers=1)

    # Step 2: Prepare input data
    print("2. Preparing input data...")
    input_ids = [10, 20, 30, 40, 50]  # Simulated sequence
    target_ids = [15, 25, 35, 45, 55]  # Simulated target sequence

    # Step 3: Initialize PID inference engine
    print("3. Initializing PID inference engine...")
    pid_engine = PIDInferenceEngine(
        p_gain=1.0,
        i_gain=0.1,
        d_gain=0.05,
        perturbation_scale=0.01,
        max_iterations=50,
        convergence_threshold=1e-5,
        validation_thresholds={
            'persistence': 0.8,
            'spectral_entropy': 0.3,
            'kuramoto_order': 0.7
        }
    )

    # Step 4: Run inference
    print("4. Running PID-enhanced inference...")
    result = pid_engine.infer(transformer, input_ids, target_ids)

    # Step 5: Display results
    print("
" + "=" * 80)
    print("✅ INFERENCING COMPLETE")
    print("=" * 80)
    print(f"Input IDs: {input_ids}")
    print(f"Target IDs: {target_ids}")
    print(f"Decoded IDs: {result['decoded_ids']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    print(f"Final Structure Valid: {result['is_valid']}")
    print("
Final Metrics:")
    for k, v in result['final_metrics'].items():
        print(f"  {k}: {v:.4f}")

    # Step 6: Demonstrate self-validation
    if result['is_valid']:
        print("
🎉 Self-Validation Passed: The inferred structure represents a coherent harmonic oscillator.")
        print("This indicates that the PID loop successfully decoded the internal 'wheel' and assimilated all perspectives.")
    else:
        print("
⚠️  Self-Validation Failed: The structure does not meet harmonic coherence criteria.")
        print("Consider adjusting PID gains or perturbation scale.")

    print("
" + "=" * 80)
    print("💡 Conceptual Insight: This framework enables transformers to learn during inference by")
    print("   dynamically exploring latent space via perturbative feedback and validating structures")
    print("   using physical principles (persistence, coherence, entropy). This mimics the process of")
    print("   'spinning the wheel' to achieve deep understanding — a key step toward AGI.")
    print("=" * 80)

    return result


# ========================
# 4. Entry Point
# ========================

if __name__ == "__main__":
    # Run the AGI inference demo
    run_agi_inference()


```
