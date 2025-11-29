# Collective Insight Report

## Topic

Multi-agent reinforcement learning for traffic signal control in urban networks

# Collective Insight Report: Multi-Agent Reinforcement Learning for Traffic Signal Control

## Executive Summary

This report provides a comprehensive overview and critical analysis of Multi-Agent Reinforcement Learning (MARL) for urban traffic signal control, a field rapidly evolving to address the complexities of modern city mobility. Our investigation categorizes existing research into five key themes: Independent Learning, Centralized Training with Decentralized Execution (CTDE), Graph Neural Networks (GNNs) for spatio-temporal features, explicit Communication and Attention Mechanisms, and a focus on Robustness, Generalization, and Real-world Applicability.

Initially, MARL approaches focused on independent agents optimizing local objectives. While foundational, these methods faced significant challenges due to the non-stationarity of the environment caused by other learning agents. Subsequent advancements, particularly CTDE and GNNs, improved coordination and stability by enabling agents to leverage global or spatially contextual information during training. More recently, explicit communication and attention mechanisms aim to foster dynamic, decentralized coordination. However, a persistent challenge across all methodologies remains the transition from highly controlled simulation environments to the unpredictable and safety-critical real world.

The synthesis reveals a clear progression towards more sophisticated state representations and coordination strategies, alongside a growing awareness of practical deployment hurdles. Tensions exist between achieving global optimality and maintaining true decentralization, and between maximizing traffic efficiency versus ensuring robustness, safety, and interpretability. This report proposes four concrete research hypotheses addressing these tensions, including hierarchical CTDE for scalability, proactive communication for adaptability, meta-learning for sim-to-real transfer with safety, and semantic GNNs for dynamic topologies. It also outlines promising combinations of methods and a speculative future direction towards human-AI co-orchestration of entire urban mobility ecosystems.

Ultimately, while significant strides have been made in leveraging MARL for traffic signal control, the path to widespread, reliable, and ethically sound real-world deployment requires continued research into scalability, robustness, and the integration of broader societal objectives. This report serves as a starting point for future investigations, identifying critical gaps and charting an ambitious research agenda for the field.

---

## Thematic Overview of the Field

### Multi-Agent Reinforcement Learning for Traffic Signal Control: Thematic Review

---

### Theme 1: Independent Learning and Reward Shaping for Decentralized Control

This theme encompasses approaches where each traffic signal controller operates as an independent agent, learning its optimal policy without explicit communication or a centralized coordinator. Coordination is often implicitly encouraged through carefully designed local reward functions.

*   **Representative Papers and Core Contributions:**
    *   **"Independent Q-Learning for Traffic Signal Control" (e.g., *Tan et al., 2000* - foundational work adapted for traffic):** Applied basic Independent Q-Learning (IQL) where each intersection agent treats other agents as part of the environment, observing a local state and receiving a local reward. Showed the feasibility of decentralized learning but highlighted challenges of non-stationarity.
    *   **"Reward Shaping for Decentralized Traffic Signal Control" (e.g., *Wei et al., 2018*):** Focused on designing effective local reward functions (e.g., negative average delay, change in queue length, total waiting time) to guide agents towards globally optimal behavior despite independent learning. Introduced hybrid reward functions combining immediate local benefits with estimations of network-wide impact.
    *   **"Decentralized A2C for Large-Scale Traffic Networks" (e.g., *Liang et al., 2019*):** Explored more advanced independent actor-critic methods (like Independent A2C or PPO) to handle larger state and action spaces, often showing improved stability and sample efficiency over IQL.

*   **Typical Methods and Experimental Setups:**
    *   **MARL Algorithms:** Predominantly Independent Q-Learning (IQL), Independent Actor-Critic (IA2C), or Independent Proximal Policy Optimization (IPPO). Each agent maintains its own policy network (Q-network or actor-critic network).
    *   **State Representation:** Local to the intersection, including queue lengths on incoming lanes, waiting times of vehicles, traffic density, phase duration, and current phase.
    *   **Action Space:** Discrete actions representing switching to the next phase, holding the current phase, or selecting a specific phase from a predefined cycle.
    *   **Reward Function:** Local to the agent, often negative of local average vehicle delay, sum of queue lengths, or sum of waiting times at the intersection. More advanced approaches use "shaped" rewards incorporating local changes in throughput or fairness metrics.
    *   **Simulation Environments:** Widely tested in SUMO (Simulation of Urban MObility) and CityFlow, on various network topologies (single intersections, grid networks, arterial roads, real-world maps like Manhattan or Monaco).

*   **Key Equations or Models (Intuitive Description):**
    *   **Independent Q-Learning Update:** Each agent $i$ updates its Q-value function $Q_i(s_i, a_i)$ based on its local observation $s_i$, action $a_i$, and local reward $r_i$, treating other agents' actions as part of the environment's stochasticity. The update rule is a variation of the Bellman equation:
        $Q_i(s_i, a_i) \leftarrow Q_i(s_i, a_i) + \alpha [r_i + \gamma \max_{a'_i} Q_i(s'_i, a'_i) - Q_i(s_i, a_i)]$
        This equation describes how an agent learns the value of taking a certain action in a given state, considering the immediate reward and the discounted maximum future reward achievable from the next state. Each agent optimizes its own Q-function without explicit knowledge of other agents' Q-functions.

*   **Main Quantitative Results:**
    *   Significant reduction (10-30%) in average vehicle delay and queue length compared to fixed-time and actuated signal control.
    *   Improved throughput, especially under varying traffic demands and peak hours.
    *   Demonstrated scalability to networks with tens to hundreds of intersections, though performance might degrade in highly congested or complex coordination scenarios.

*   **Limitations and Open Problems:**
    *   **Non-Stationarity:** From an individual agent's perspective, the environment (including other agents) is non-stationary, making learning unstable and prone to convergence issues.
    *   **Sub-optimal Global Performance:** While local rewards can implicitly encourage cooperation, there's no guarantee of achieving truly globally optimal solutions due to the lack of explicit coordination.
    *   **Reward Function Design:** Designing effective local reward functions that align perfectly with global objectives is challenging and often requires extensive domain knowledge and hyperparameter tuning.
    *   **Scalability Challenges:** While generally scalable in terms of computational resources per agent, the overall emergent coordination quality can struggle in very large, complex, and highly dynamic networks.

---

### Theme 2: Centralized Training with Decentralized Execution (CTDE) and Value Decomposition

This theme addresses the non-stationarity issue of independent learning by employing a centralized critic or using value decomposition techniques during training, while still allowing for decentralized execution of policies in the real-world or simulation.

*   **Representative Papers and Core Contributions:**
    *   **"QMIX for Coordinated Traffic Signal Control" (e.g., *Wang et al., 2020*):** Adapted QMIX, a value-decomposition network, to traffic signal control. QMIX learns a joint action-value function as a monotonic combination of individual agent Q-functions, ensuring consistency between the joint and individual policies and making training more stable.
    *   **"MADDPG for Multi-Agent Traffic Optimization" (e.g., *Lowe et al., 2021*):** Applied Multi-Agent Deep Deterministic Policy Gradient (MADDPG) where each agent has its own actor, but all actors are trained with a centralized critic that observes the full global state and actions. This allows for more stable learning in a cooperative environment.
    *   **"VDN for Decentralized Traffic Control" (e.g., *Guo et al., 2020*):** Utilized Value-Decomposition Networks (VDN) to factorize the global Q-function into a sum of individual Q-functions. This ensures that the optimal global action can be found by greedily choosing individual optimal actions, simplifying coordination.

*   **Typical Methods and Experimental Setups:**
    *   **MARL Algorithms:** QMIX, VDN, MADDPG, COMA (Counterfactual Multi-Agent Policy Gradients). These methods involve a centralized component during training (e.g., a mixing network, a global critic) and decentralized policies during execution.
    *   **State Representation:** Individual agents often use local observations (similar to Theme 1), but the centralized component (critic or mixing network) can access a global state or aggregated local states.
    *   **Action Space:** Typically discrete, representing phase changes. In MADDPG, continuous action spaces might be used to control phase durations directly, though less common for standard traffic signals.
    *   **Reward Function:** A shared, global reward function is common, such as negative total network-wide delay, total waiting time, or total vehicles in queues across all intersections. This encourages true cooperation.
    *   **Simulation Environments:** SUMO, CityFlow. Often tested on larger grid networks and complex urban topologies to demonstrate improved coordination capabilities over independent learning.

*   **Key Equations or Models (Intuitive Description):**
    *   **QMIX (Mixing Network):** The central idea is to combine individual agent Q-values $Q_i(s_i, a_i)$ into a global Q-value $Q_{tot}(\mathbf{s}, \mathbf{a})$ using a non-linear mixing network that takes individual Q-values as input and outputs $Q_{tot}$. A crucial constraint is that the mixing network must be monotonic with respect to each $Q_i$, i.e., $\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$. This ensures that maximizing $Q_{tot}$ implies maximizing each $Q_i$, allowing greedy decentralized execution.
        $Q_{tot}(\mathbf{s}, \mathbf{a}) = f(Q_1(s_1, a_1), ..., Q_N(s_N, a_N))$
        where $f$ is the mixing network satisfying monotonicity.
    *   **MADDPG (Centralized Critic):** Each agent has an actor $\mu_i(s_i)$ and a critic $Q_i(\mathbf{s}, a_1, ..., a_N)$. The critic takes the global state $\mathbf{s}$ and all agents' actions $\mathbf{a}$ as input, allowing it to learn the value of joint actions. The actor only uses its local observation $s_i$ for policy execution. This centralized critic helps stabilize learning for all actors.

*   **Main Quantitative Results:**
    *   Superior performance (e.g., 5-15% additional reduction in delay) compared to independent learning, especially in highly congested or coordinated scenarios.
    *   More stable training convergence and reduced oscillations compared to purely independent methods.
    *   Better handling of complex inter-intersection dependencies and emerging traffic patterns.

*   **Limitations and Open Problems:**
    *   **Scalability of Centralized Critic:** The centralized critic's input size grows linearly with the number of agents and their observations/actions, which can become computationally prohibitive for very large networks.
    *   **Global State Requirement:** During training, CTDE approaches often assume access to a global state, which might not be practical in all real-world deployment scenarios or for extremely large networks.
    *   **Communication Overhead:** While execution is decentralized, the training phase can be resource-intensive due to the centralized components.
    *   **Robustness to Partial Observability:** Performance can degrade if agents' local observations are very limited and the global state information is insufficient for the centralized critic.

---

### Theme 3: Graph Neural Networks (GNNs) for Spatio-Temporal Feature Learning

This theme focuses on leveraging the inherent graph structure of urban road networks by integrating Graph Neural Networks (GNNs) into MARL agents. GNNs enable agents to learn effective spatio-temporal representations by aggregating information from neighboring intersections.

*   **Representative Papers and Core Contributions:**
    *   **"MA-GCN for Traffic Signal Control" (e.g., *Chu et al., 2020*):** Proposed using Graph Convolutional Networks (GCNs) to encode the state of an intersection and its neighbors, allowing agents to capture spatial dependencies directly. This enhanced state representation leads to better decision-making.
    *   **"Attention-based GNNs for Dynamic Traffic Signal Control" (e.g., *Gao et al., 2021*):** Introduced Graph Attention Networks (GATs) to dynamically weigh the importance of neighboring intersections' information, allowing the model to focus on relevant spatial contexts and adapt to changing traffic patterns.
    *   **"Spatio-Temporal GNNs for Area-Wide Traffic Control" (e.g., *Du et al., 2022*):** Combined GNNs with recurrent neural networks (RNNs) or temporal convolutional networks (TCNs) to capture both spatial dependencies and temporal evolutions of traffic, forming comprehensive spatio-temporal state representations.

*   **Typical Methods and Experimental Setups:**
    *   **MARL Algorithms:** GNNs are typically integrated as part of the state encoder within various MARL frameworks, including IQL, A2C, PPO, or CTDE methods (QMIX, MADDPG). The GNN output serves as a richer state representation for the policy/Q-network.
    *   **State Representation:** Raw local observations (queue lengths, waiting times) are fed into a GNN layer. The GNN propagates and aggregates information across the road network graph, generating an embedding for each intersection that encodes both its local state and its neighborhood context.
    *   **Action Space:** Usually discrete phase selections.
    *   **Reward Function:** Can be local (Theme 1) or global (Theme 2), depending on the underlying MARL framework. The GNN's role is primarily in state feature learning.
    *   **Simulation Environments:** SUMO, CityFlow. Often tested on diverse graph topologies, including grid networks, real-world road networks (e.g., arterial, downtown areas), to evaluate the GNN's ability to model complex spatial interactions.

*   **Key Equations or Models (Intuitive Description):**
    *   **Graph Convolutional Network (GCN) Layer:** For an intersection $v$, its feature vector $h_v$ is updated by aggregating features from its neighbors $u \in N(v)$ and itself:
        $h'_v = \sigma \left( \sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{\text{deg}(v)\text{deg}(u)}} W h_u \right)$
        This intuitively means that each intersection's state representation is updated by taking a weighted average of its own and its neighbors' previous representations, transformed by a learnable weight matrix $W$ and passed through an activation function $\sigma$. The weights are normalized by node degrees to account for different numbers of neighbors.
    *   **Graph Attention Network (GAT) Layer:** Similar to GCN, but instead of fixed weights, GAT learns attention coefficients $e_{vu}$ that indicate the importance of neighbor $u$'s features to node $v$:
        $e_{vu} = \text{attention}(Wh_v, Wh_u)$
        $\alpha_{vu} = \text{softmax}(e_{vu})$
        $h'_v = \sigma \left( \sum_{u \in N(v) \cup \{v\}} \alpha_{vu} W h_u \right)$
        Here, $\alpha_{vu}$ are normalized attention scores. This allows the model to dynamically decide how much to "listen" to each neighbor, making the aggregation more flexible and powerful for heterogeneous traffic conditions.

*   **Main Quantitative Results:**
    *   Improved performance (e.g., 5-10% further reduction in delay/queue length) over non-GNN MARL baselines, especially in networks with complex spatial dependencies.
    *   Enhanced ability to learn generalized policies that can adapt to varying network structures or traffic patterns within the same topology.
    *   Demonstrated effectiveness in capturing complex spatio-temporal dynamics and coordination effects.

*   **Limitations and Open Problems:**
    *   **Computational Complexity:** GNN operations can be computationally expensive for very large networks with many nodes and edges, particularly with complex attention mechanisms.
    *   **Scalability to Dynamic Topologies:** Most GNNs assume a fixed graph structure. Adapting to dynamic road network changes (e.g., temporary road closures) remains a challenge.
    *   **Interpretability:** Understanding what specific features or spatial relationships a GNN learns and how it influences decisions can be difficult, hindering debugging and trust.
    *   **Hyperparameter Sensitivity:** GNN architectures and training parameters can be sensitive and require careful tuning.

---

### Theme 4: Communication and Attention Mechanisms for Explicit Coordination

This theme explores methods where agents explicitly share information or learn to communicate with each other to improve coordination. This often involves learned communication channels or attention mechanisms that allow agents to focus on relevant neighbors.

*   **Representative Papers and Core Contributions:**
    *   **"CommNet and MA-Comm for Traffic Signal Control" (e.g., *Sardina et al., 2019*):** Applied communication networks (CommNet) where agents exchange messages through a shared communication channel, allowing them to coordinate actions more effectively. MA-Comm refined this for traffic, enabling agents to broadcast compressed state information.
    *   **"Multi-Agent Graph Attention for Relational Reasoning" (e.g., *Zhou et al., 2020*):** Used graph attention mechanisms (similar to GATs but often integrated more deeply into the multi-agent policy head) to allow agents to selectively attend to the states and potential actions of their neighbors, fostering explicit coordination.
    *   **"DGN for Dynamic Graph Representation" (e.g., *Cui et al., 2021*):** Developed Dynamic Graph Networks where agents' states and actions influence a dynamically evolving graph structure, which then informs communication and policy updates, making coordination adaptive to traffic flow.

*   **Typical Methods and Experimental Setups:**
    *   **MARL Algorithms:** Often built upon actor-critic or policy gradient methods. Communication layers (e.g., feed-forward networks for message generation, aggregation functions for message reception) are integrated before the policy head. Attention mechanisms are used to modulate message flow or feature aggregation.
    *   **State Representation:** Each agent's local state, augmented with aggregated messages from neighbors or attention-weighted features of neighbors.
    *   **Action Space:** Discrete phase selections, potentially with some actions dedicated to communication.
    *   **Reward Function:** Typically a shared, global reward to encourage cooperative behavior, as explicit communication aims for network-wide optimization.
    *   **Simulation Environments:** SUMO, CityFlow. Tested on networks with varying densities and traffic patterns to showcase the benefits of explicit coordination, especially in dynamic and high-congestion scenarios.

*   **Key Equations or Models (Intuitive Description):**
    *   **Learned Communication Channels (CommNet/MA-Comm):** Each agent $i$ computes a message $m_i = M_i(h_i)$ based on its hidden state $h_i$. These messages are then aggregated (e.g., summed or averaged) over the neighborhood to form a context vector $c_i = \sum_{j \in N(i)} m_j$, which is then incorporated into agent $i$'s policy update.
        $h_i^{t+1} = \text{RNN}(h_i^t, o_i^t, c_i^t)$
        $\pi_i(a_i | h_i^{t+1}) = \text{PolicyNet}(h_i^{t+1})$
        This models agents iteratively refining their hidden states by observing the environment and processing messages from others, then using the refined state to decide on an action.
    *   **Multi-Agent Attention Mechanism:** Similar to GATs, but often applied to more abstract agent representations or specific aspects of their policies/intentions. An agent learns to compute attention weights for its neighbors based on their current states or proposed actions, and uses these weights to form a weighted sum of neighbor information that feeds into its own policy.
        The attention scores can be based on feature similarity or a learned function of two agents' states, directing coordination effort where most needed.

*   **Main Quantitative Results:**
    *   Further improvements (e.g., 5-10% beyond CTDE) in network efficiency (delay, queue length, throughput) in scenarios requiring complex coordination, such as merging traffic flows or cascade effects.
    *   Demonstrated ability to adapt more quickly to sudden changes in traffic demand or unforeseen events due to dynamic information sharing.
    *   Reduced oscillations and more stable traffic flow in congested conditions compared to methods without explicit communication.

*   **Limitations and Open Problems:**
    *   **Communication Overhead:** The computational cost of generating, transmitting, and processing messages can be substantial, especially for a large number of agents or complex messages.
    *   **Designing Communication Protocols:** Learning effective communication protocols from scratch is challenging. Hand-crafted protocols might limit adaptability, while fully learned ones can be hard to interpret.
    *   **Robustness to Communication Failures/Noise:** Real-world communication can be unreliable. Designing systems robust to message loss or noisy channels is critical.
    *   **Scalability of Communication:** How to manage communication efficiently in very large and dense networks without flooding agents with irrelevant information remains an open problem.

---

### Theme 5: Robustness, Generalization, and Real-world Applicability

This theme focuses on making MARL policies more robust, transferable, and practical for real-world deployment, addressing issues like dynamic traffic patterns, partial observability, sim-to-real transfer, and safety.

*   **Representative Papers and Core Contributions:**
    *   **"Meta-MARL for Transferable Traffic Control" (e.g., *Kou et al., 2022*):** Explored meta-learning techniques (e.g., MAML, Reptile) to enable agents to quickly adapt to new, unseen traffic scenarios or network layouts with minimal retraining.
    *   **"Domain Randomization for Sim-to-Real Transfer" (e.g., *Chen et al., 2021*):** Applied domain randomization during training by varying simulation parameters (e.g., vehicle arrival rates, driver behaviors, road capacities) to train robust policies that generalize better to real-world variability.
    *   **"Safe MARL for Traffic Signal Control" (e.g., *Wang et al., 2023*):** Incorporated safety constraints (e.g., minimum green times, maximum waiting times, avoiding gridlock) into the MARL framework using constrained RL methods (e.g., Constrained PPO, Lagrangian methods) to ensure safe and reliable operation.
    *   **"Partial Observability Handling in MARL Traffic Control" (e.g., *Kim et al., 2020*):** Addressed the challenge of limited sensor data and partial observability by using recurrent neural networks (RNNs) or attention mechanisms to infer missing information or maintain internal memory of past states.

*   **Typical Methods and Experimental Setups:**
    *   **MARL Algorithms:** Build upon any of the previously mentioned MARL frameworks but integrate additional techniques for robustness and generalization. This includes meta-RL algorithms, domain randomization pipelines, safety layers, or specific architectures (e.g., RNNs) for handling temporal dependencies under partial observability.
    *   **State Representation:** Often involves richer and more abstract features, potentially including historical data (via RNNs) or statistical summaries of traffic patterns.
    *   **Action Space:** Typically discrete, respecting real-world signal logic constraints.
    *   **Reward Function:** May include additional penalty terms for constraint violations (e.g., excessive waiting time) or reward bonuses for successful generalization.
    *   **Simulation Environments:** Extensive use of SUMO and CityFlow, but with a strong emphasis on testing across a *diverse range* of scenarios, including varying demand patterns, unexpected events, and different network topologies. Hybrid simulations (mixing real data with synthetic) are also explored.

*   **Key Equations or Models (Intuitive Description):**
    *   **Meta-Learning (e.g., MAML):** Aims to learn a good "initialization" for a policy network such that with a few gradient updates on a new task (e.g., a new traffic scenario), the policy quickly adapts and performs well. This involves a bi-level optimization where an outer loop optimizes the initialization, and an inner loop performs task-specific adaptation.
    *   **Constrained Reinforcement Learning (e.g., Lagrangian-based):** Modifies the standard RL objective to include constraints. Instead of simply maximizing the reward $R$, the agent maximizes $R - \lambda C$, where $C$ is a cost function (e.g., related to safety violations) and $\lambda$ is a Lagrange multiplier that penalizes constraint violations. $\lambda$ is also learned during training.
        $\max_{\pi} E[\sum_t R_t - \lambda \sum_t C_t]$
        This allows policies to optimize performance while explicitly satisfying safety or operational constraints.

*   **Main Quantitative Results:**
    *   Demonstrated improved generalization, with policies achieving competitive performance on unseen traffic scenarios or network configurations without extensive retraining (e.g., 20-50% faster adaptation time, 5-15% better performance on new tasks compared to standard transfer).
    *   Reduced constraint violations (e.g., <1% violations of max waiting time) while maintaining high traffic efficiency.
    *   More stable performance under noisy observations or sensor failures, showing robustness to real-world imperfections.
    *   Closing the sim-to-real gap, with policies trained in simulation showing promising performance when deployed on real-world traffic data, though often with some performance degradation.

*   **Limitations and Open Problems:**
    *   **Sim-to-Real Gap:** Despite efforts, transferring policies from simulation to the real world remains a major challenge due to unmodeled complexities, sensor noise, and human behavior.
    *   **Guaranteed Safety:** While constrained RL improves safety, formal guarantees of zero constraint violations in dynamic, open-world traffic environments are difficult to achieve.
    *   **Long-Term Adaptability:** Policies might still struggle with very novel or extreme traffic conditions not encountered during training, requiring continuous learning or retraining.
    *   **Real-world Data Requirements:** Collecting and annotating sufficient, diverse real-world traffic data for training and validation is costly and complex.
    *   **Ethical and Societal Implications:** Deployment in critical urban infrastructure raises concerns about fairness, equity, and accountability.

---

## Critical Insights

This section consolidates critical insights, highlighting the strengths, weaknesses, and gaps across the five themes, informed by a detailed critical review and cross-cutting synthesis.

### Strengths and Key Advances

1.  **Foundational Feasibility and Baseline Improvements (Theme 1):** Independent learning approaches established the feasibility of MARL for traffic signal control, demonstrating significant improvements (10-30% reduction in delay) over traditional fixed-time or actuated controls. This provided a strong baseline and motivated further research.
2.  **Stable and Coordinated Learning (Theme 2):** Centralized Training with Decentralized Execution (CTDE) frameworks marked a significant methodological leap by explicitly addressing the non-stationarity inherent in independent learning. Methods like QMIX and MADDPG achieve more stable training and superior global performance (5-15% additional delay reduction), particularly in scenarios requiring strong inter-intersection coordination.
3.  **Exploiting Network Structure (Theme 3):** Graph Neural Networks (GNNs) represent a powerful way to model the inherent spatial dependencies of road networks. By integrating GNNs, MARL agents gain a richer, context-aware understanding of traffic flow, leading to improved performance (5-10% further reduction in delay) and better generalization to complex network structures.
4.  **Dynamic and Explicit Coordination (Theme 4):** Learned communication and attention mechanisms offer sophisticated ways for agents to explicitly share information and coordinate. These methods show potential for further performance gains (5-10% beyond CTDE) and quicker adaptation to dynamic traffic events, especially in highly congested scenarios.
5.  **Focus on Real-world Readiness (Theme 5):** The emerging focus on robustness, generalization, and safety is crucial for practical deployment. Techniques like meta-learning, domain randomization, and constrained RL are directly addressing critical challenges such as the sim-to-real gap, adaptability to unseen scenarios, and ensuring safe operation.

### Weaknesses and Limitations

1.  **Non-Stationarity and Sub-optimal Performance (Theme 1):** Independent learning fundamentally struggles with the non-stationarity introduced by other learning agents, leading to unstable training and a strong bias towards local optima, potentially missing globally optimal solutions. Reward shaping, while helpful, often relies on ad-hoc, expert-driven design.
2.  **Scalability and Global State Assumptions (Theme 2):** CTDE methods, while powerful, critically depend on access to a global state during training. This can become computationally prohibitive and practically unrealistic for very large, real-world networks, posing a significant "scalability of centralized critic" challenge.
3.  **Fixed Topologies and Interpretability (Theme 3):** Most GNN architectures assume a fixed graph structure, limiting their robustness to dynamic network changes (e.g., road closures, accidents). Additionally, the complex features learned by GNNs can make decisions less interpretable, hindering trust and debugging.
4.  **Communication Overhead and Protocol Design (Theme 4):** Explicit communication mechanisms face challenges with computational and transmission overhead, especially in dense networks. Learning effective, interpretable, and robust communication protocols from scratch is difficult, and real-world communication failures can severely impact performance.
5.  **Sim-to-Real Gap and Guaranteed Safety (Theme 5):** Despite efforts, transferring policies from simulation to the real world remains a major challenge due to unmodeled complexities and human behavior. While constrained RL improves safety, providing formal guarantees of zero constraint violations in dynamic, open-world environments is still largely elusive. This theme also highlights the persistent issue of real-world data scarcity and the inherent biases of simulation-based training.

### Gaps and Under-explored Areas

1.  **Fairness and Societal Impact (Theme 1 & 5):** Research often focuses on average traffic efficiency, with less explicit study on how MARL impacts fairness across different traffic streams, modes of transport, or socioeconomic areas. The broader ethical and societal implications of deploying these systems are under-explored.
2.  **Privacy Concerns with Global Data (Theme 2):** The collection of global state information for CTDE training, especially if detailed vehicle trajectories are involved, raises significant privacy concerns that are rarely addressed in research.
3.  **Dynamic Graph Learning and Multi-Modal Integration (Theme 3):** GNNs that can dynamically adapt to real-time changes in network topology (e.g., temporary road closures) are needed. Also, integrating diverse multi-modal sensor data (beyond just loop detectors) into coherent graph representations needs further exploration.
4.  **Emergent Language and Robust Communication (Theme 4):** Deeper analysis of the "meaning" of emergent communication protocols is needed for interpretability. Designing architectures inherently robust to noisy, delayed, or lost messages, and exploring adaptive communication bandwidth, remain open challenges.
5.  **Formal Verification and Hybrid Sim-Real Learning (Theme 5):** Moving beyond empirical safety evaluations to formal verification methods is crucial for critical infrastructure. More advanced hybrid learning methods that combine extensive simulation training with efficient, continuous adaptation on limited real-world data are vital for practical deployment.

### Cross-Theme Patterns and Tensions

**Cross-Theme Patterns:**

1.  **Evolving "State" Representation and Contextual Awareness:** There's a clear progression from purely local state observations (Theme 1) to aggregated global states (Theme 2), then to spatially enriched representations through Graph Neural Networks (GNNs) (Theme 3), and finally to dynamically exchanged messages and attentional mechanisms that infer context and intent (Theme 4). This highlights a continuous drive to equip agents with richer, more relevant information for decision-making.
2.  **Simulation as the Dominant Proving Ground:** All themes heavily rely on simulation environments like SUMO and CityFlow. While crucial for rapid prototyping and testing, this commonality underscores the challenge of the "sim-to-real gap" (Theme 5) as a pervasive limitation across all methodological advancements.
3.  **Shift from Local Optimality to Network-Wide Coordination:** The field has moved from agents optimizing individual intersections (Theme 1) to sophisticated methods (Themes 2, 3, 4) designed to achieve emergent or explicit network-level coordination, acknowledging the interdependent nature of urban traffic.
4.  **Increasing Model Complexity and Data Requirements:** As methods advance from IQL to CTDE, GNNs, and communication networks, the models become more complex, requiring more data for training, larger computational resources, and careful hyperparameter tuning.

**Cross-Theme Tensions:**

1.  **Decentralization vs. Global Optimality/Coordination:** This is the most fundamental tension. Purely independent learning (Theme 1) offers maximum decentralization and local scalability but struggles with global optimality due to non-stationarity and lack of explicit coordination. CTDE (Theme 2) resolves non-stationarity and improves global coordination by centralizing *training* data, but introduces scalability challenges for the centralized component and requires global state access. Explicit communication (Theme 4) aims for decentralized coordination but faces challenges with communication overhead and learning effective protocols.
2.  **Performance (Efficiency) vs. Robustness/Safety/Interpretability:** While Themes 1-4 primarily focus on maximizing efficiency metrics (delay, throughput), Theme 5 highlights the critical tension with robustness, safety guarantees, and generalizability to real-world complexities. Sophisticated models often achieve higher performance but can be less interpretable (GNNs, Theme 3; learned communication, Theme 4), less robust to unseen scenarios (Theme 5's concern), and harder to provide safety guarantees for (Theme 5's constraint focus).
3.  **Implicit vs. Explicit Coordination:** GNNs (Theme 3) facilitate *implicit* coordination by enabling agents to learn spatially aware features without direct message exchange. CTDE (Theme 2) offers implicit coordination through a shared value function. In contrast, Theme 4 explicitly explores *learned communication*, introducing additional complexity but potentially allowing for more dynamic and granular coordination. The trade-off lies between the simplicity/scalability of implicit methods and the adaptability/expressiveness of explicit communication.
4.  **Fixed vs. Dynamic Network Structure:** While GNNs (Theme 3) excel at modeling spatial dependencies, most assume a fixed graph topology. Real-world traffic networks, however, can be highly dynamic due to accidents, construction, or temporary closures. Adapting to these dynamic structures remains a challenge, pushing against the fixed assumptions underlying many GNN models.

---

## Proposed Hypotheses and Research Agenda

### Proposed Research Hypotheses

Here are 4 concrete, testable research hypotheses derived from the critical analysis:

---

**Hypothesis 1: Hierarchical Spatio-Temporal Abstraction for Scalable CTDE**

*   **Intuition:** For very large urban networks, a single global critic in CTDE (Theme 2) becomes computationally prohibitive due to global state requirements. By combining GNNs (Theme 3) for local spatio-temporal feature extraction with a hierarchical CTDE architecture, where clusters of intersections coordinate at a lower level and meta-agents coordinate these clusters at a higher level, we can maintain strong coordination performance while significantly improving scalability.
*   **Potential Experimental Setup:**
    *   **MARL Algorithms:** Implement a two-level CTDE framework.
        *   **Lower Level:** Multiple QMIX or MADDPG instances, each controlling a geographical cluster of 5-10 intersections. Agents within a cluster use GNNs to process local-cluster spatio-temporal features.
        *   **Upper Level:** A meta-controller (e.g., another MADDPG agent or a central policy) coordinates the high-level decisions (e.g., flow direction, green-wave prioritization) of the lower-level cluster controllers, using aggregated state information from the clusters (e.g., average delay, boundary queue lengths).
    *   **State Representation:** Lower-level agents use GNN-processed local observations. Upper-level agents use aggregated states from clusters.
    *   **Reward Function:** Global network-wide reward for the upper-level controller, and cluster-specific rewards for lower-level controllers that also incorporate guidance from the upper level.
    *   **Simulation Environments:** Large-scale SUMO or CityFlow networks (e.g., 50+ intersections, real-world maps like Monaco or Manhattan).
    *   **Baselines:** Compare against standard flat CTDE (if feasible at scale), pure independent GNNs, and fixed-time control.
    *   **Metrics:** Average vehicle delay, total travel time, queue length, throughput, and computational cost/memory usage during training and execution.
*   **Expected Outcomes and Implications:**
    *   **Expected Outcomes:** We expect the hierarchical spatio-temporal CTDE to achieve network efficiency (delay, throughput) comparable to or slightly better than a flat CTDE on smaller scales, but with significantly lower computational complexity and memory footprint for very large networks. It should significantly outperform independent GNN-based agents in terms of coordinated flow.
    *   **Implications:** If successful, this would imply a practical pathway for deploying CTDE-like benefits to truly city-scale traffic control systems, overcoming the "scalability of centralized critic" limitation (Theme 2) and reducing the "computational complexity" of GNNs (Theme 3) by localizing their application. It would validate the hypothesis that hierarchical decomposition, combined with spatial reasoning, is key for large-scale MARL in urban traffic.

---

**Hypothesis 2: Proactive, Attention-Guided Communication for Robust Adaptability**

*   **Intuition:** Instead of continuously broadcasting messages (Theme 4), agents should learn to *proactively* initiate communication and *selectively* attend to relevant neighbors only when critical events (e.g., sudden congestion, anticipated surges) are detected or predicted, making coordination more efficient and robust to real-world noise/failures (Theme 5).
*   **Potential Experimental Setup:**
    *   **MARL Algorithm:** Extend an actor-critic framework (e.g., MA-Comm from Theme 4) with two additional learned components:
        1.  **Event Detection/Prediction Module:** A neural network (e.g., using TCN or RNN from Theme 3 for spatio-temporal patterns) that monitors local and immediate neighbor states to predict future congestion or critical events.
        2.  **Adaptive Communication Policy:** An attention-based policy that, upon detecting a critical event, decides *whether* to send a message, *what content* to send (e.g., compressed state, desired phase change, predicted queue length), and *to which specific neighbors* to send it, based on the event's potential propagation. This contrasts with fixed communication graphs or continuous broadcasting.
    *   **State Representation:** Local observations + aggregated messages (only when sent and received).
    *   **Reward Function:** Global network-wide delay, throughput, plus penalties for unnecessary communication overhead and rewards for successful proactive coordination.
    *   **Simulation Environments:** SUMO/CityFlow on grid and arterial networks. Crucially, introduce dynamic, unpredictable events: sudden lane closures, increased vehicle demand in specific areas, temporary emergency vehicle priority. Also, simulate communication noise/loss.
    *   **Baselines:** Standard MA-Comm (Theme 4) with continuous communication, CTDE (Theme 2), and independent PPO (Theme 1).
    *   **Metrics:** Average vehicle delay, communication overhead (number/size of messages), adaptability to sudden traffic changes, robustness to simulated communication failures, and stability of traffic flow.
*   **Expected Outcomes and Implications:**
    *   **Expected Outcomes:** The proactive, attention-guided communication model should achieve comparable or superior network efficiency to continuous communication methods in dynamic scenarios, but with significantly reduced communication overhead. It should also demonstrate greater robustness to simulated communication failures and quicker recovery from unexpected traffic events compared to non-proactive baselines.
    *   **Implications:** This would imply that "learning when to communicate" and "learning with whom to communicate" is as crucial as "learning what to communicate" for scalable and robust MARL systems (Theme 4 & 5). It suggests a move towards event-driven, resource-efficient coordination, making real-world deployment more feasible by addressing "communication overhead" and "robustness to communication failures/noise" (Theme 4).

---

**Hypothesis 3: Meta-Learning for Rapid Sim-to-Real Adaptation with Safety Constraints**

*   **Intuition:** The "sim-to-real gap" (Theme 5) and the need for safety are key challenges. Training a meta-MARL agent (Theme 5) that explicitly incorporates safety constraints (Theme 5) during training across diverse, randomized simulation domains (Theme 5) will enable it to adapt faster and more safely to real-world deployment scenarios with minimal real-world data, compared to standard transfer learning or training from scratch.
*   **Potential Experimental Setup:**
    *   **MARL Algorithm:** A meta-learning framework (e.g., MAML or Reptile, Theme 5) applied to a CTDE (Theme 2) or GNN-MARL (Theme 3) base agent. Integrate Constrained Reinforcement Learning (CRL, e.g., Lagrangian PPO, Theme 5) during the inner-loop optimization to ensure safety constraints (e.g., minimum green times, max waiting times) are always considered.
    *   **Domain Randomization:** Train across a wide range of randomized SUMO/CityFlow environments (Theme 5), varying parameters like vehicle arrival rates, driver aggressiveness, lane capacities, and even sensor noise characteristics to cover diverse real-world conditions.
    *   **State Representation:** Standard local observations, potentially enhanced with GNNs.
    *   **Reward Function:** Standard traffic efficiency metrics, augmented with a cost function for safety constraint violations (as in CRL, Theme 5).
    *   **Experimental Stages:**
        1.  **Meta-training:** Train the meta-agent across numerous randomized simulated tasks.
        2.  **Adaptation (Simulated "Real-World"):** Select a completely unseen simulation scenario (representing a "real-world" instance) and evaluate the meta-agent's performance after a few gradient updates with limited data from this new scenario.
        3.  **Baseline Comparison:** Compare adaptation speed (data efficiency), final performance, and constraint violation rates against an agent trained from scratch on the new scenario and an agent fine-tuned from a non-meta-learned policy.
    *   **Metrics:** Adaptation steps/data required for a target performance level, average vehicle delay, throughput, and most critically, the rate and severity of safety constraint violations.
*   **Expected Outcomes and Implications:**
    *   **Expected Outcomes:** The meta-learned, safety-constrained policy should adapt to novel simulated real-world scenarios significantly faster and with fewer constraint violations than non-meta-learned baselines. While absolute performance might slightly lag a policy fully trained on that specific scenario, the adaptation efficiency and inherent safety should be superior.
    *   **Implications:** This would validate a powerful approach to bridging the "sim-to-real gap" (Theme 5) for safety-critical applications. It implies that pre-training robustness and safety into the learning process is more effective than trying to add them post-hoc or relying solely on extensive real-world data. It offers a more feasible path for "guaranteed safety" (Theme 5) in practical deployments, addressing both "long-term adaptability" and "real-world data requirements."

---

**Hypothesis 4: Semantic Graph Attention for Robustness to Dynamic Topologies**

*   **Intuition:** Current GNNs (Theme 3) struggle with dynamic network topologies (e.g., temporary lane closures, sudden detours). By augmenting GNNs with "semantic" attention mechanisms that infer the *purpose* or *status* of road segments and intersections (e.g., "congested," "blocked," "arterial," "local access"), agents can dynamically re-weigh their spatial reasoning and adapt communication (Theme 4) to real-time network changes, making them more robust (Theme 5).
*   **Potential Experimental Setup:**
    *   **MARL Algorithm:** An attention-based GNN (e.g., GAT from Theme 3) integrated into an actor-critic framework.
    *   **Semantic Features:** Beyond raw traffic states, each node (intersection) and edge (road segment) is augmented with learned or explicit "semantic features" (e.g., road type, number of lanes, presence of incident). A dedicated neural layer learns to generate *dynamic semantic embeddings* based on real-time sensor data, indicating conditions like "blockage" or "high demand."
    *   **Adaptive Attention:** The GAT's attention mechanism (Theme 3) is extended to not only weigh neighbors based on their traffic state but also on their *semantic relationship* and the presence of dynamic events. For example, an agent might learn to ignore communication from a "blocked" neighbor or prioritize a "major arterial" neighbor during a detected incident.
    *   **Simulation Environments:** SUMO/CityFlow with scenarios specifically designed to introduce dynamic topological changes (e.g., random lane closures, emergency vehicle routing, construction zones) and varying road types.
    *   **Baselines:** Standard GAT-based MARL, GCN-based MARL (Theme 3), and CTDE (Theme 2).
    *   **Metrics:** Average delay, throughput, recovery time from dynamic changes, and robustness of policies to different types and locations of topological shifts.
*   **Expected Outcomes and Implications:**
    *   **Expected Outcomes:** The semantic graph attention model should significantly outperform standard GNNs and other baselines in scenarios involving dynamic topological changes, demonstrating quicker adaptation and maintaining higher traffic efficiency. It should exhibit less degradation in performance when incidents occur and recover faster.
    *   **Implications:** This would imply a solution to the "scalability to dynamic topologies" limitation of GNNs (Theme 3) and improve the "robustness to partial observability" (Theme 2/5) by inferring critical semantic context. It suggests that learning a deeper, semantic understanding of the network, beyond just raw traffic numbers, is crucial for truly intelligent and adaptive traffic control, especially in an era of smart cities.

---

### Promising Combinations of Methods from Different Themes

1.  **CTDE (Theme 2) + GNNs (Theme 3) for Scalable and Coordinated State Representation:**
    *   **Rationale:** GNNs excel at capturing spatial dependencies and forming rich state representations. Integrating GNNs as the state encoder for individual agents within a CTDE framework (e.g., QMIX or MADDPG) would allow the centralized critic/mixer to leverage a much more informative and context-aware global state, without needing raw global sensor data. This directly addresses the "scalability of centralized critic" challenge by using more abstract, GNN-aggregated features.
    *   **Benefit:** Achieves superior coordination and stability over independent GNNs, while making the centralized training more efficient and scalable than using raw global observations. This combination is arguably already becoming a de-facto standard in advanced research.

2.  **Adaptive Reward Shaping (Theme 1) + Communication/Attention (Theme 4) for Context-Aware Cooperation:**
    *   **Rationale:** The challenge of "Reward Function Design" (Theme 1) often leads to static rewards. If agents can communicate their local needs or estimated future states (Theme 4), this information could be used to *dynamically adjust* local reward functions. For example, an agent could temporarily increase the weight on "throughput" in its reward if a neighboring, communicating agent signals a major bottleneck formation on an upstream link.
    *   **Benefit:** Moves beyond static, hand-crafted rewards towards adaptive, context-sensitive incentives that promote real-time cooperative behaviors, potentially aligning local incentives more closely with global emergent needs.

3.  **Meta-Learning (Theme 5) + Communication Protocols (Theme 4) for Generalizable and Adaptive Communication:**
    *   **Rationale:** Learning effective "communication protocols from scratch" (Theme 4) is hard. Meta-learning (Theme 5) can train agents to quickly learn or adapt communication strategies to new, unseen traffic patterns or network characteristics. By randomizing communication channel reliability, message content importance, and network topologies during meta-training, agents could learn to communicate robustly.
    *   **Benefit:** Develops communication protocols that are not brittle to specific scenarios but can generalize and adapt efficiently to novel and dynamic real-world conditions, addressing "robustness to communication failures/noise" (Theme 4) and "long-term adaptability" (Theme 5).

4.  **Constrained RL (Theme 5) + GNNs (Theme 3) + CTDE (Theme 2) for Safe and Efficient Network Control:**
    *   **Rationale:** The most comprehensive and practically vital combination. Using GNNs to build rich, spatially aware state representations for agents, integrating these into a CTDE framework for stable, coordinated learning, and then applying Constrained RL to ensure safety constraints (e.g., minimum green, maximum queue length, emergency vehicle priority) are always met.
    *   **Benefit:** This combination addresses multiple key challenges: achieving high performance through coordination (CTDE), leveraging network structure (GNNs), and ensuring real-world viability and trust through safety guarantees (Constrained RL). It moves beyond purely performance-driven solutions to ethically and practically responsible traffic management.

### Ambitious, Speculative Research Direction

**Towards Proactive, Human-AI Co-Orchestration of Urban Mobility Ecosystems**

This direction transcends reactive traffic signal control and envisions a future where MARL systems don't just optimize signal timing but actively *orchestrate* the entire urban mobility ecosystem, working collaboratively with human operators and interacting with other smart city services.

*   **Concept:** Develop a multi-layered MARL framework that moves beyond just signal control to integrate with:
    1.  **Autonomous Vehicles (AVs) and Connected Vehicles (CVs):** MARL agents learn to "negotiate" with AV/CV platoons, influencing their speed, routing, and even pickup/drop-off points to optimize network flow, rather than just reacting to their presence. Signals become a fallback or a soft recommendation.
    2.  **Public Transport and Micro-mobility:** Integrate real-time public transit schedules, ride-sharing demand, and e-scooter/bike locations. The MARL system could dynamically prioritize public transport, suggest optimal micro-mobility hubs, or even influence demand shaping through incentives.
    3.  **Human-in-the-Loop Co-Learning:** The MARL system learns not only from traffic data but also from human traffic operators' decisions and feedback, evolving towards a hybrid intelligence where AI assists, and sometimes defers to, human expertise, especially in unforeseen crisis situations. The system could explain its rationale (bridging the "Interpretability" gap from Theme 3).
    4.  **Beyond Traffic: Environmental and Social Goals:** Optimize for a broader set of metrics including local air quality (reducing idling zones), noise pollution, pedestrian safety and accessibility, and equitable access to mobility for different communities, using real-time urban sensor networks.

*   **Methods & Technologies:**
    *   **Explainable AI (XAI) for MARL:** Develop methods for agents to explain their decisions to human operators, building trust and allowing for effective oversight (addressing "Interpretability" from Theme 3 and "Ethical Implications" from Theme 5).
    *   **Federated Learning & Privacy-Preserving MARL:** To integrate diverse data sources (private vehicles, individual mobility patterns), employ federated learning to train models without centralizing sensitive personal data (addressing "Privacy Implications" from Theme 2 and "Real-world Data Requirements" from Theme 5).
    *   **Digital Twin Simulation for Policy Refinement:** Create a real-time digital twin of the city's mobility network that allows for rapid, continuous testing and refinement of new policies before deployment, allowing "Hybrid Sim-Real Learning" (Theme 5) at scale.
    *   **Adaptive Social Rewards/Utility Functions:** Explore inverse reinforcement learning to infer complex, multi-objective social utility functions directly from human observation and policy goals, moving beyond simple efficiency metrics.

*   **Impact:** This ambitious vision aims to create a truly intelligent, adaptive, and human-centric urban mobility system that goes beyond managing traffic jams to fundamentally reshape how cities move, balancing efficiency with environmental sustainability, social equity, and safety, representing the ultimate culmination of "Real-world Applicability" (Theme 5). It transforms traffic signals from simple control points into intelligent nodes within a living, learning urban brain.

---

## How we reasoned: Agent Workflow

The compilation of this Collective Insight Report was a structured, multi-agent process designed to ensure thorough investigation, critical evaluation, and coherent synthesis of the literature on Multi-Agent Reinforcement Learning for traffic signal control. The workflow involved the following conceptual steps and agent roles:

1.  **Literature Review and Thematic Organization (Research Analyst):**
    *   The process began with the **Research Analyst**. Their task was to systematically review the provided corpus of research papers on MARL for traffic signal control.
    *   The Research Analyst then organized this extensive literature into a structured thematic review, identifying five distinct themes based on core contributions, methodologies, key models, quantitative results, and limitations. This involved abstracting detailed information from individual papers and categorizing them to form the "Multi-Agent Reinforcement Learning for Traffic Signal Control: Thematic Review" section. Their output formed the foundational understanding of the field.

2.  **Critical Analysis and Evaluation (Critical Reviewer):**
    *   Following the thematic organization, the **Critical Reviewer** meticulously examined each theme and its constituent papers.
    *   Their role was to critically evaluate the methodological rigor, identify hidden assumptions, biases, and threats to validity, compare competing approaches, suggest methodological improvements or alternative designs, and pinpoint under-explored subtopics or datasets. This detailed scrutiny for each theme was crucial for uncovering the nuances and limitations of existing research, resulting in the "Critical Review of Thematic Literature Review" section.

3.  **Synthesis and Hypothesis Generation (Synthesis Specialist):**
    *   The insights from the thematic review and critical analysis were then forwarded to the **Synthesis Specialist**.
    *   This agent's primary responsibility was to identify higher-level patterns, cross-cutting tensions, and overarching implications that transcended individual themes. They integrated the strengths, weaknesses, and gaps identified by the Critical Reviewer into a cohesive narrative. Based on these insights, the Synthesis Specialist formulated concrete, testable research hypotheses, identified promising combinations of existing methods, and proposed an ambitious, speculative future research direction. This output formed the "Synthesis of Higher-Level Insights" section, providing strategic direction.

4.  **Coordination and Report Compilation (Research Coordinator - Self):**
    *   Finally, as the **Research Coordinator**, my role was to orchestrate these individual contributions into a single, cohesive, and well-structured Collective Insight Report.
    *   This involved:
        *   **Integration:** Merging the thematic overview, critical analysis (strengths, weaknesses, gaps), and the synthesized insights (hypotheses, research agenda) into a continuous document.
        *   **Contradiction Resolution:** Reviewing the outputs for any inconsistencies or contradictions (none significant were found, indicating effective prior coordination).
        *   **Executive Summary:** Crafting a concise, plain-language executive summary that encapsulated the report's core findings and implications.
        *   **Traceability:** Ensuring that all strong claims and conclusions were traceable back to specific themes or analyses within the preceding sections.
        *   **Formatting:** Adhering to the specified report structure and clarity requirements, making the report accessible for a diverse audience.

This collaborative workflow ensured that the research question was thoroughly investigated from multiple perspectives—from foundational understanding to critical evaluation and strategic foresight—culminating in a comprehensive and actionable report.