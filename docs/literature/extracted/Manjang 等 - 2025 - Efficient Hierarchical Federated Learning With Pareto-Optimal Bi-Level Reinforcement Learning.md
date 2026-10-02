# Efficient Hierarchical Federated Learning With Pareto-Optimal Bi-Level Reinforcement Learning

Ousman Manjang , Yanlong Zhai , Member, IEEE, Jun Shen , Senior Member, IEEE, Adil Sarwar Xutian He, Huan Wang, Student Member, IEEE, and Liehuang Zhu , Senior Member, IEEE

Abstract—Hierarchical federated learning (HFL) has emerged as a popular federated learning method by introducing additional aggregation levels using intermediate edge servers. The performance of HFL approaches is contingent upon the aggregation frequency and the number of intermediate aggregation rounds. Existing approaches mainly focus on optimizing the aggregation frequency only, neglecting the impact of intermediate aggregation rounds on training performance. On the other hand, these methods also fail to consider the multidimensional effects of aggregation frequency, consequently focusing only on the performance accuracy while overlooking the detrimental effects on both the communication costs and training latency. This article introduces Efficient HFL with Pareto-Optimal Bi-Level Reinforcement Learning (HFL-PBRL), a novel HFL framework that employs a bi-level reinforcement learning (RL)- based algorithm to jointly optimize aggregation frequencies and rounds across edge servers. This algorithm is accompanied by a Pareto-efficient multiobjective optimization approach to strike an optimal tradeoff among model accuracy, communication cost and convergence time. The varied aggregation frequencies and rounds might introduce inconsistencies; therefore, we employ a hierarchical pullback mechanism that iteratively pulls the client models toward a synchronized anchor model, ensuring effective divergence control. Furthermore, we devise a harmonic weight assignment strategy that dynamically adjusts the aggregation weights of each model based on their current, historical, and anticipated divergence, addressing the model fluctuations and asynchrony. Extensive evaluations demonstrate that HFL-PBRL consistently achieves high model accuracy and faster convergence with minimal communication costs compared to baselines and SOTA.

Index Terms—Aggregation frequency, aggregation rounds, aggregation weight, hierarchical federated learning (HFL).

## I. INTRODUCTION

HE RAPID advancement of edge computing and federated learning (FL) has transformed how machine learning models are trained across decentralized data sources [1]. Traditional FL allows devices to train models locally, sharing only updates with a central server, which reduces data transmission and enhances privacy [2]. However, as the number of devices scale, FL encounters challenges, such as communication overhead, extended training latency, and resource constraints [3]. This has led to the adoption of hierarchical federated learning (HFL) to improve scalability by distributing aggregation across edge and cloud tiers [4], [5]. Nevertheless, the efficiency of HFL hinges on two underexplored factors: 1) the frequency of model synchronizations and 2) the number of intermediate edge server-level aggregation rounds.

However, optimizing either of the above factors is nontrivial. For example, to facilitate model convergence, clients may need frequent updates [4], which in turn increases the communication cost and latency. On the contrary, reducing the aggregation frequency can decrease the communication overhead, but it might lead to overfitting [6]. Additionally, extending the aggregation frequency requires additional rounds of local computation from clients, potentially causing client models to diverge toward their local optima [7].

To address these issues, existing research has sought to optimize aggregation frequency while meeting specific performance objectives. In particular, Yang et al. [8] jointly optimize the aggregation structure and frequency to minimize global loss. The aggregation optimization in [6] is driven by the varying computational and communication capabilities of devices. Meanwhile, in [9], the authors tackle the challenge of weak synchronization, while the aggregation frequency optimization in [3] focuses on the tradeoff between transmission overhead and training loss. Additionally, the work in [10] addresses the challenges of stochastic variability and fixed training time by proposing a deep reinforcement learning (RL)-based framework that optimally adjusts aggregation frequencies to enhance training efficiency.

The optimization approaches explored so far optimizes these factors in isolation and primarily focused on a single aspect of training performance. As a result, these approaches mostly only improve one specific training objective at the expense of others. Additionally, the number of intermediate aggregation rounds significantly impacts the training efficiency [11]. These integral aspects of aggregation control mechanism have not been adequately explored in the current literature.

In response to the above challenges, we propose Efficient HFL with Pareto-Optimal Bi-Level RL (HFL-PBRL). At the core of HFL-PBRL is a RL-based algorithm that integrates hierarchical decision making with Pareto dominance for the joint optimization of aggregation frequency and aggregation rounds. To address the varying complexities and training demands at different levels of the hierarchical network, HFL-PBRL features a bi-level decision-making approach, consisting of high-level and low-level policy network. While operating in unique synergy, these policy networks complement each other to jointly determine the optimal aggregation frequencies and rounds. In this setup, the high-level policy network sets overarching strategies (i.e., RL actions) based on global training objectives, while the low-level policy facilitates realtime adjustments in response to dynamic training conditions at the individual edge servers.

On the other hand, employing such a dynamic RL policy optimization process presents a unique challenge: improving one objective often comes at the expense of the other. Traditional weighted-sum approaches struggle to balance these competing objectives, as they require manual tuning and fail to capture Pareto-optimal solutions–where no objective can be improved without sacrificing another. To address this, we integrate Pareto-efficient multiobjective optimization with our bi-level RL algorithm, enabling adaptive tradeoffs without arbitrary weight assignments. This ensures balanced performance across all objectives while maintaining training efficiency.

Furthermore, prior studies had relied on various metrics, such as dataset size [12], number of completed local epochs [13], or model staleness [14] to achieved weighted aggregation. However, these methods tend to rely solely on the current performance, failing to capture the effects of aggregation frequencies and the fluctuating model performance. To address these issues, we employ a divergence-control mechanism, referred to as hierarchical pullback, to enforce the alignment of local models. This mechanism greedily pulls the locally trained models toward a synchronized anchor, the design of which is inspired by our recent work [5]. Meanwhile, with asynchronous global aggregation, we devise a harmonic weight assignment (HWA) method to ensure stable model improvements while handling model fluctuations. Specifically, the HWA determines the aggregation weight of each model based on its current, historical, and anticipated divergence from the global model. In summary, our work introduces the following key contributions.

1) We propose HFL-PBRL, a new HFL framework that jointly optimizes intermediate aggregation frequency and rounds. HFL-PBRL leverage the hierarchical architecture to devise a bi-level, RL-based algorithm that intelligently facilitates level-wise policy optimization to address both the global and local training objectives.

2) We incorporate a Pareto-dominance principle into the bi-level policy optimization process, ensuring that RL decision-making aligns effectively with the simultaneous optimization of multiple competing objectives.

3) We propose a hierarchical pullback mechanism that utilizes an anchor model to facilitate local model alignment. Concurrently, we devise a HWA method in the global aggregation to address the effect of inconsistent updates.

4) Extensive experiments has been conducted to validate the effectiveness of our overall framework against baseline approaches and state-of-the-art (SOTA) benchmarks.

The upcoming content is arranged as follows. We discuss the related works in Section II. In Section III we explain the system design and illustrate our proposed algorithm in Section IV. In Section V, we present the evaluation of HFL-PBRL. We conclude this article in Section VI.

## II. RELATED WORK

Several studies have adopted hierarchical architecture to improve communication efficiency and scalability in distributed training. For instance, HierFAVG [4] typically leverages massive training datasets by enabling partial model aggregation at edge servers, thereby reducing the communication bottlenecks. Later, [15] presented an asynchronous framework that introduces a gateway layer between the cloud and edge nodes, while performing asynchronous aggregation at the intermediate gateway, and staleness-aware asynchronous cloud aggregation. Deng et al. [16] further exploited the hierarchical architecture to reduce the per-round communication cost and the expected synchronization rounds by shaping data distribution at edge. Zhang et al. [17] introduced an enhanced K-Center algorithm and a deep RL-based strategy for designating Internet of Things (IoT) devices to edge servers. Much more recently, [18] designed a hierarchical dualheterogeneous FL framework that accounts for the impact of noise introduced by the challenge of labeling and the freerider issue of the clients. Similarly, [19] introduced a two-level HFL structure featuring an over-the-air aggregation approach for uplink and a bandwidth-constrained broadcast method for downlink.

Few research have been focused on optimizing aggregation frequencies in FL. Yang et al. [9] alleviated the waiting time and enhanced the resource efficiency of devices at different phases of training by adjusting the aggregation frequency. Similarly, [20] mitigated the influence of stragglers with a unique cluster construction and asynchronous global aggregation. Furthermore, Wu et al. [21] synergized synchronous local and asynchronous global aggregation, which lowered the communication overhead yet considerably complicated the algorithm [22]. More interestingly, Xiang et al. [23] recommended engaging edge servers that possess training data in collaborative model training with clients to optimize local aggregation frequency.

Beyond the heavy reliance on predetermined aggregation schedules, existing studies failed to incorporate multiobjectivebased solutions. Moreover, these methods optimizes either aggregation rounds or frequency in isolation, leading to unbalanced tradeoffs. As a result, they have limited effectiveness in IoT-driven applications, where high performance across different metrics becomes a critical necessity.

## III. PROBLEM DEFINITION AND FORMULATION

Herein, we first present the overview of HFL-PBRL and subsequently formulate the learning problem.

## A. Overview of HFL-PBRL

We illustrate the overview of HFL-PBRL in Fig. 1, highlighting the key processes, which are categorized into four major steps. In each global round, clients utilize the newest global model and start their local training procedures along with the hierarchical pullback mechanism (step 1). Periodically, clients upload their models to the associated edge server for partial model aggregation at designated frequencies $( f _ { \mathrm { a g g } } ^ { * } )$ (step 2). These frequencies are first set by the highlevel policy network, while allowing for flexible and real-time adjustments by the intermediate servers in response to specific training conditions. At each optimal aggregation round $( N _ { \mathrm { a g g } } ^ { * } )$ / edge servers upload their models to the root server (step 3). Subsequently, the root server asynchronously receives the updates and executes global aggregation by using the proposed HWA. Following the global model update, HFL-PBRL leverages the high-level policy $( \pi _ { H } )$ along with the complementary synergy of the low-level policy (πL), to determine the Paretoefficient solutions (step 4).

<!-- image-->  
Fig. 1. Overview of HFL-PBRL framework.

## B. Problem Formulation and Objectives

We consider a hierarchical training paradigm with a central (root) server for global aggregation, a set of edge servers for intermediate model aggregation, and K distributed training clients, indexed by $k = 1 , 2 , \ldots , K$ . These clients are designated to M distinct edge servers for efficient model training and coordination. Let ${ \mathcal { C } } _ { m }$ denotes the set of clients associated with edge server m, and the size of this set is $| { \mathcal { C } } _ { m } |$ , satisfying $\begin{array} { r } { \sum _ { m = 1 } ^ { M } | \bar { \mathcal { C } } _ { m } | = K } \end{array}$ . Each participating client k holds a local dataset $\mathcal { D } _ { k } = \{ ( x _ { \nu } , y _ { \nu } ) \ | \ \nu = 1 , \ldots , | \mathcal { D } _ { k } | \}$ , where $x _ { \nu }$ denotes an input feature and $y _ { \nu }$ is the corresponding label. The model parameters are denoted as $\mathbf { w } \in \mathbb { R } ^ { d }$ , and the loss function for a data point $( x _ { \nu } , y _ { \nu } )$ is given by $\boldsymbol { \ell } ( x _ { \nu } , y _ { \nu } ; { \mathbf w } )$ . We characterize the average local loss function $\mathcal { F } _ { k } ( \mathbf { w } )$ for client k as

$$
\mathcal { F } _ { k } ( \mathbf { w } ) = \frac { 1 } { | \mathcal { D } _ { k } | } \sum _ { \nu = 1 } ^ { | \mathcal { D } _ { k } | } \ell ( x _ { \nu } , y _ { \nu } ; \mathbf { w } ) .\tag{1}
$$

The goal of our system is to minimize the global loss (w) over the K distributed training clients, expressed as

$$
\operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { d } } \mathcal { G } ( \mathbf { w } ) = \frac { 1 } { K } \sum _ { m = 1 } ^ { M } | \mathcal { C } _ { m } | \mathcal { F } _ { m } ( \mathbf { w } )\tag{2}
$$

where $\begin{array} { r } { \mathcal { F } _ { m } ( \mathbf { w } ) = ( 1 / | \mathcal { C } _ { m } | ) \sum _ { k \in \mathcal { C } _ { m } } \mathcal { F } _ { k } ( \mathbf { w } ) } \end{array}$ is the aggregated loss of the intermediate edge server m. Therefore, the optimal global model $\mathbf { w } _ { g } ^ { * }$ is defined as

$$
\mathbf { w } _ { g } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { d } } \mathcal { G } ( \mathbf { w } ) .\tag{3}
$$

## C. Optimization Problem and Constraints

We analytically model three critical system performance metrics that are fundamentally governed by the aggregation frequency $f _ { \mathrm { a g g } }$ and rounds of intermediate aggregations $N _ { \mathrm { a g g } }$

1) Latency: The total system latency comprises the following key components, each capturing distinct bottlenecks.

Aggregation Latency $L _ { a g g } ^ { ( m ) } .$ : The per-round aggregation latency at edge server m, which depends on the processing speed of the edge server, $\Phi _ { m }$ is determined by

$$
L _ { \mathrm { a g g } } ^ { ( m ) } = \operatorname* { m a x } _ { k \in \mathcal { C } _ { m } } \biggl ( \frac { | \mathcal { C } _ { m } | } { p \cdot \Phi _ { m } } \biggr )\tag{4}
$$

where p represents the successful update participation rate. This formulation accounts for system asynchrony by considering the slowest participating client, while the inverse relationship with $\Phi _ { m }$ reflects the natural scaling of aggregation speed with server capability.

Communication Latency $L _ { c o m } ^ { ( m ) }$ : This component constitutes the combined uplink transmission latency $L _ { k } ^ { t r }$ and downlink latency for receiving the aggregated model $L _ { k } ^ { \partial l }$ , formulated as

$$
L _ { \mathrm { c o m } } ^ { ( m ) } = \operatorname* { m a x } _ { k \in \mathcal { C } _ { m } } \Bigl ( L _ { k } ^ { t r } + L _ { k } ^ { d l } \Bigr ) .\tag{5}
$$

Equation (5) captures the bottleneck effect inherent in FL systems, where the aggregator’s progress is constrained by the slowest client’s communication time. Therefore, the overall latency is determined by the maximum communication delay among all clients in the set $\mathcal { C } _ { m }$ at edge server m.

Local Training Latency: Iterative aggregations increases the computational workload for clients because each aggregation round requires additional local training. The time for a set of clients ${ \mathcal { C } } _ { m }$ to complete their local training process is given as

$$
L _ { \mathrm { t r a i n } } ^ { ( m ) } = \operatorname* { m a x } _ { k \in \mathcal { C } _ { m } } \left( \frac { E _ { k } \cdot | \mathcal { D } _ { k } | } { \Phi _ { k } } \right)\tag{6}
$$

where $E _ { k }$ is the number of local iterations and $\Phi _ { k }$ is the corresponding computing capability of client k.

Therefore, the overall latency incurred at edge server $m ,$ is modeled as

$$
L _ { \mathrm { t o t } } ^ { ( m ) } = \operatorname* { m a x } _ { k \in \mathcal { C } _ { m } } \biggl [ \frac { | C _ { m } | } { p \cdot \Phi _ { m } } + \Bigl ( L _ { k } ^ { t r } + L _ { k } ^ { d l } \Bigr ) + \frac { E _ { k } | \mathcal { D } _ { k } | } { \Phi _ { k } } \biggr ] .\tag{7}
$$

By (7), we can deduce that the latency in each communication round scales with the local aggregations.

2) Communication Cost: Within each global training round, we speculate that each edge server performs multiple rounds of local aggregations before the global synchronization. In this context, we focus on the communication cost incurred at the client-to-edge server level, modeled as

$$
C _ { \mathrm { c o m } } = \left( \frac { S \cdot | \mathcal { C } _ { m } | } { B _ { u } } + \frac { S } { B _ { d } } \right)\tag{8}
$$

where S denotes the model size, $B _ { u }$ and $B _ { d }$ represent the upload bandwidth and the download bandwidth, respectively.

3) Convergence: Suboptimal aggregation scheduling and rounds, whether too frequent or too sparse–critically impacts model accuracy and convergence time. This occurs because clients might perform too few or too many local iterations, governed by $f _ { \mathrm { a g g } }$ and $N _ { \mathrm { a g g } }$ , respectively. We quantify this tradeoff through a penalty function as follows:

$$
{ A _ { \mathrm { a c c } } = A _ { \mathrm { g l o b a l } } - \epsilon \cdot A _ { \mathrm { l o s s } } ^ { ( m ) } }\tag{9}
$$

where $A _ { \mathrm { g l o b a l } }$ represents the ideal accuracy. $A _ { \mathrm { l o s s } } ^ { ( m ) }$ captures frequency and round-dependent degradation at edge server m, and  scales the magnitude of this penalty.

Optimization Problem: Our algorithm seeks to determine the optimal aggregation frequency $f _ { \mathrm { a g g } } ^ { * }$ and rounds $N _ { \mathrm { a g g } } ^ { * } ,$ that minimizes the cost $\mathcal { C } ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } ) = L _ { \mathrm { t o t } } ^ { \langle \widetilde { m } \rangle } + C _ { \mathrm { c o m } } ^ { ( m ) } + A _ { \mathrm { l o s s } } ^ { ( m ) }$

$$
\left( f _ { \mathrm { a g g } } ^ { * } , N _ { \mathrm { a g g } } ^ { * } \right) = \operatorname* { m i n } _ { f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } } \mathcal { C } ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } )\tag{10}
$$

$$
\mathrm { ~ \large ~ \left. ~ \right/ ~ } f _ { \mathrm { a g g } } \leq { f } _ { \mathrm { m a x } } \qquad \forall m \in { \cal M }\tag{P1}
$$

$$
N _ { \mathrm { a g g } } \leq N _ { \mathrm { m a x } } \qquad \forall m \in M\tag{P2}
$$

$$
C _ { \mathrm { c o m } } ^ { ( m ) } \leq B _ { \mathrm { t h r e s } } \forall m \in M\tag{P3}
$$

$$
L _ { \mathrm { t o t } } ^ { ( m ) } \leq L _ { \mathrm { m a x } } \qquad \forall m \in M\tag{P4}
$$

$$
A _ { \mathrm { a c c } } \geq A _ { \mathrm { m i n } } \qquad \forall m \in M\tag{P5}
$$

$$
N _ { \mathrm { a g g } } \geq 1 \qquad \forall m \in M .\tag{P6}
$$

To ensure feasibility, (P1) and (P2) impose a limit such that $f _ { \mathrm { a g g } }$ and $N _ { \mathrm { a g g } }$ do not exceed their maximum thresholds. This guarantees that the system remains practical for implementation in resource-constrained environments. Complementing these, P3, P4, and P5 collectively regulate the accumulated communication cost, latency, and accuracy deviation, ensuring each remain within acceptable levels. Lastly, P6 ensures that there is at least one round of intermediate aggregation, ensuring the realization of hierarchical FL mechanism. Most of the above discussed notations are summarized in Table I.

TABLE I SUMMARY OF MAIN NOTATIONS
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $K$ </td><td>Number of clients (devices)</td></tr><tr><td> $\mathcal { D } _ { k }$ </td><td>Dataset of client k</td></tr><tr><td>t</td><td>Current iteration index</td></tr><tr><td> $\mathbf { w } _ { k } ^ { ( t ) }$ </td><td>Model parameter of client (device)k at round t</td></tr><tr><td> $m ^ { \dag }$ </td><td>Index of edge servers</td></tr><tr><td> $\mathcal { F } _ { k } ( \mathbf { w } )$ </td><td>Loss function of client (device) k</td></tr><tr><td> $\mathbf { w } _ { g } ^ { * }$ </td><td>Optimal global model</td></tr><tr><td> $f _ { a g g }$   $\begin{array} { r } { J ^ { a g g } } \\ { . . . ( t ) } \end{array}$ </td><td>Intermediate aggregation frequency</td></tr><tr><td> $Z _ { m } ^ { ( \iota ) }$ </td><td>Anchor model at edge server m</td></tr><tr><td> $N _ { a g g }$ </td><td>Number of intermediate aggregation rounds</td></tr><tr><td> ${ \mathcal { V } } _ { m }$ </td><td>Adjustments to  $f _ { a g g }$  and  $\bar { N } _ { a g g }$  at edge server m</td></tr><tr><td> $\mathcal { C } _ { m }$ </td><td>Set of clients associated with edge server m</td></tr><tr><td> $h _ { m } ^ { ( t ) }$ </td><td>Aggregation weight of model m at round t</td></tr><tr><td> $E _ { k }$ </td><td>Number of local iterations at client k</td></tr><tr><td> $\mathbf { w } _ { g } ^ { ( t ) }$ </td><td></td></tr><tr><td></td><td>Aggregated global model at time t</td></tr><tr><td>E</td><td>Accuracy loss (degradation) factor</td></tr><tr><td> $R$   $L$ </td><td>Function for reward vector</td></tr><tr><td> $\mathcal { H } ( \mathbf { w } )$ </td><td>Latency component</td></tr><tr><td></td><td>Pullback-based empirical risk minimization</td></tr><tr><td> $\mu _ { k } ^ { t }$   $\boldsymbol { \Phi } ^ { \dot { } \dot { } }$ </td><td>Pullback adjustment factor</td></tr><tr><td> $\Psi _ { m }$ </td><td>Processing or computing capability</td></tr><tr><td> $\mathcal { A }$ </td><td>Computational demand on the edge server m</td></tr><tr><td></td><td>Optimal solution for aggregation frequency and rounds</td></tr><tr><td> $\delta ( \cdot )$ </td><td>Key performance indicator for the training condition</td></tr><tr><td> $S$   $A$ </td><td>Size of model (parameters)</td></tr><tr><td></td><td>Accuracy improvement over time (convergence)</td></tr><tr><td> $\Gamma _ { l o c }$ </td><td>Local computation time</td></tr><tr><td> $\theta _ { i }$ </td><td>Dynamic weights for the current system priority</td></tr><tr><td> $\mathcal { C } ( f )$ </td><td>Cost function associated with aggregation frequency</td></tr><tr><td>Ω</td><td>Solution Space for Pareto optimality</td></tr><tr><td> $| P F |$ </td><td>Number of solutions in the Pareto front  $P F$ </td></tr><tr><td> $\dot { B } _ { u } , \dot { B } _ { d }$ </td><td>Uplink and downlink bandwidth</td></tr><tr><td> $\alpha , \beta , \gamma$ </td><td>Trade-off parameter for reward function</td></tr><tr><td> $\boldsymbol { \mathcal { X } } _ { f } , \boldsymbol { \mathcal { X } } _ { N }$ </td><td>Setofallfeasiblevalues of  $f _ { a g g }$  and  $N _ { a g g }$ </td></tr><tr><td> $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ </td><td>Trade-off factors for aggregation weight</td></tr><tr><td> $s _ { H }$ </td><td>State vector for high-level policy network  $\pi _ { H }$ </td></tr><tr><td> $s _ { L } ,$ </td><td>State vector for low-level policy network  $\pi _ { L }$ </td></tr><tr><td> $\beta _ { 1 } , \beta _ { 1 } , \beta _ { 1 }$ </td><td>Weights for latency,communication cost and convergence</td></tr></table>

## IV. PROPOSED BI-LEVEL PARETO-EFFICIENT MULTIOBJECTIVE OPTIMIZATION

Herein, we would first explore in detail the proposed bi-level RL along with the Pareto-efficient multiobjective optimization, and then present the overall HFL-PBRL algorithm.

## A. Bi-Level Reinforcement Learning Approach

The decomposition of RL’s decision-making process into dual levels (high-level and low-level policies) is central to our framework. While these two policies complement each other, each has its own state, actions, and reward functions to handle distinctive processes at different levels.

High-Level State Vector $( s _ { H } )$ : The high-level state vector $s H$ serves as the global observation basis for our bi-level RL framework, providing a comprehensive characterization of the learning system state to inform strategic control decisions. This state representation quantitatively encodes essential system parameters, such as the average bandwidth $( B _ { \mathrm { a v g } } )$ and computational load on the edge servers $\Psi _ { m } ,$ average latency $( L _ { \mathrm { a v g } } )$ and the accuracy improvement in the previous rounds $A _ { h i s t }$ Formally, $s H$ is defined as

$$
s _ { H } \triangleq \{ B _ { \mathrm { a v g } } , L _ { \mathrm { a v g } } , \Psi _ { m } , A _ { h i s t } , C _ { \mathrm { c o m } } \} .\tag{11}
$$

The state vector $s _ { H }$ undergoes iterative updating following each aggregation cycle and serves as the input domain for our high-level policy network $\pi _ { H }$ . This network generates optimal solutions of synchronization frequencies and rounds. This closed-loop observation-action cycle is fundamental to our adaptive optimization framework.

Low-Level State Vector $\left( s _ { L } \right)$ : In HFL, edge servers operate in highly dynamic environments where network conditions fluctuate unpredictably, client data distributions shift over time, and computational loads vary due to device heterogeneity. The low-level state vector $\left( s _ { L } \right)$ addresses these challenges by capturing real-time local conditions as follows:

$$
s _ { L } \triangleq \left\{ \delta , \frac { | \mathcal { C } _ { m } | } { p } , \Psi _ { m } , \Delta \mathcal { D } _ { m } \right\}\tag{12}
$$

where $\Delta \mathcal { D } _ { m }$ represents the data distribution shifts at edge server m. By quantifying these dynamic conditions, $s _ { L }$ enables the detection of localized bottlenecks that would otherwise degrade model performance, serving as essential input for the low-level policy’s adaptive decision-making.

The composite metric δ, serving as our key performance indicators (KPI), quantifies real-time network conditions to trigger low-level adaptations. It combines round-trip time $( T _ { r } )$ bandwidth utilization $( B ^ { u t } )$ , and transmission loss rate $( P _ { l } )$ into a single normalized measure as follows:

$$
\delta = 1 - \left( d _ { 1 } \cdot T _ { r } + d _ { 2 } \cdot B ^ { u t } + d _ { 3 } \cdot P _ { l } \right)\tag{13}
$$

where the weights $d _ { 1 } , \ d _ { 2 }$ , and $d _ { 3 }$ are chosen to reflect the importance of these metrics. In this context, a higher value of δ indicates better network conditions whereas a lower δ signifies degraded performance, thus requiring local policy adjustments.

Reward Vectors for Bi-Level Optimization: Within our framework, each RL decision making is guided by feedback through the reward vectors which capture the evolving training performances. By effectively penalizing RL decisions through the following reward functions, we tackle the complexities and potential conflicts between the low-level and high-level policy networks. Specifically, our system allows for the flexible adjustments of decisions through dynamic weighted balancing of candidate decisions within the Pareto frontier, which collectively emphasize degree prioritization in the global sense over the local objectives, and vice versa.

Communication Cost Reward $( R _ { \mathrm { C o m } } ) \mathrm { : }$ : This reward penalizes high-communication costs, taking into account both the global weighting and local adjustments adaptive to the current network condition and performance as

$$
R _ { \mathrm { C o m } } = - ( \alpha _ { H } + \alpha _ { L } ) \cdot f _ { \mathrm { a g g } } \cdot N _ { \mathrm { a g g } } .\tag{14}
$$

Here, $\alpha _ { H }$ is the weight emphasizing global communication cost minimization and $\alpha _ { L }$ prioritizes the cost reduction at a specific edge servers over the global training objectives. Our bi-level optimization method, compounded by multiobjective performance considerations, reveals inherent tradeoffs in HFL that are often overlooked by conventional methods.

Training Latency Reward $( R _ { \mathrm { T r a i n } } )$ : This reward encourages low-latency by penalizing lower aggregation frequency and fewer aggregations, while accounting for global latency priorities and local computational constraints

$$
R _ { \mathrm { T r a i n } } = - \bigg ( \beta _ { H } + \frac { \beta _ { L } } { \Phi _ { m } } \bigg ) \cdot \frac { 1 } { f _ { \mathrm { a g g } } \cdot N _ { \mathrm { a g g } } }\tag{15}
$$

where $\beta _ { H }$ is the high-level weight, scaling the importance of latency in relation to other objectives. The term $( \beta _ { L } / \Phi _ { m } )$ is the local adjustment factor where $\beta _ { L }$ scales the latency penalty based on computational load $( \Phi _ { m } )$ at the edge server m.

Model Convergence Reward $( R _ { \mathrm { A c c } } ) { \mathrm { : } }$ : This reward promotes convergence by incentivizing higher aggregation frequency and rounds. We use a logarithmic function to account for diminishing returns, such that $R _ { \mathrm { A c c } }$ incorporates both global and local performance as follows:

$$
R _ { \mathrm { A c c } } = ( \gamma + \gamma _ { L } ) \cdot \log \left( f _ { \mathrm { a g g } } \cdot N _ { \mathrm { a g g } } + 1 \right)\tag{16}
$$

where $\gamma$ is the high-level weight prioritizing overall model convergence. $\gamma _ { L }$ is the local adjustment, reflecting emphasis on convergence status at individual edge servers.

The ensuing discussion will unveil the core concepts behind the proposed Pareto-efficient RL policy optimization.

## B. Multiobjective RL Policy Optimization

A common method for addressing multiobjective problems is the weighted sum approach, which, while straightforward and effective in certain scenarios, heavily relies on predefined weights. This can be suboptimal, particularly when the objectives conflict. In contrast, our Pareto approach directly identifies areas of conflict and optimizes for diverse compromise solutions without the need for prior weight adjustments, ensuring a more adaptable and effective solution strategy. This approach is explained in the following key steps.

1) High-Level Action Optimization With Pareto Efficiency: The high-level policy $\pi _ { H }$ creates a Pareto-optimal set of candidate solutions, providing a spectrum of efficient RL action solutions across these objectives in the following steps.

Step 1 (Candidate Solution Generation via Pareto-Optimal Exploration): The high-level policy $\pi _ { H }$ generates candidate solutions through systematic exploration of the feasible parameter space , guided by Pareto dominance principles. We formally define this space as

$$
\Omega = \{ ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } ) \mid f _ { \mathrm { a g g } } \in \mathcal { X } _ { f } , N _ { \mathrm { a g g } } \in \mathcal { X } _ { N } \}\tag{17}
$$

where $\mathcal { X } _ { f } = \{ f \in \mathbb { R } ^ { + } \mid f \leq f _ { \operatorname* { m a x } } \}$ denotes the set of feasible aggregation frequencies. ${ \mathcal X } _ { N } ~ = ~ \{ N ~ \in ~ { \mathbb Z } ^ { + } ~ | ~ N ~ \leq ~ N _ { \operatorname* { m a x } } \}$ represents the constrained set of viable intermediate rounds.

Moreover, the above solution space construction adheres to the operational constraints ((P1) and (P2)) specified in (P6), ensuring computational tractability through explicit bounds $( f _ { \operatorname* { m a x } } , N _ { \operatorname* { m a x } } )$ . It facilitates adaptation to evolving the system states via dynamic updates to sH and $s _ { L }$ and Paretooptimal exploration by evaluating nondominated solutions along the efficiency frontier. This adaptive generation mechanism enables $\pi _ { H }$ to efficiently navigate the tradeoff surface between high-level and low-level policy, mediated through state-aware constraints.

Step 2 (Objective Function Evaluation): Our optimization framework evaluates each candidate solution defined as ${ \mathcal { A } } =$ $( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } ) \in \Omega$ through a multiobjective reward vector that simultaneously reflects three critical performance aspects

$$
\mathbf { R } _ { H } ( A ) = [ R _ { \mathrm { C o m } } ( A ) , R _ { \mathrm { T r a i n } } ( A ) , R _ { \mathrm { A c c } } ( A ) ] .\tag{18}
$$

The optimization process seeks to minimize the objective vector min $\boldsymbol { A } { \in } \Omega ^ { \mathbf { R } _ { H } ( \mathcal { A } ) }$ , which drives our Pareto optimization by quantifying the tradeoffs between these competing objectives. This unified evaluation enables direct comparison of candidate solutions, where superior actions—those that simultaneously minimize communication costs and latency while maximizing model accuracy—are adopted for the subsequent training rounds.

Step 3 (Identifying Pareto-Optimal Points): The Pareto optimality criterion is key to our multiobjective optimization framework, as it systematically identifies solutions that optimally balance competing objectives without subjective weighting. A solution ${ \mathcal { A } } ^ { * } = ( f _ { \mathrm { a g g } } ^ { * } , N _ { \mathrm { a g g } } ^ { * } )$ is Pareto-optimal when no alternative  exists that simultaneously improves all reward components, as formalized by the dominance condition

$$
\begin{array} { r l } & { \forall \mathcal { A } ^ { \prime } \neq \mathcal { A } ^ { * } , \quad \exists \mathcal { A } ^ { \prime } } \\ { \mathrm { s . t . ~ } } & { \left\{ \begin{array} { l l } { R _ { \mathrm { C o m } } \bigl ( \mathcal { A } ^ { \prime } \bigr ) \geq R _ { \mathrm { C o m } } ( \mathcal { A } ^ { * } ) } \\ { R _ { \mathrm { A c c } } \bigl ( \mathcal { A } ^ { \prime } \bigr ) \geq R _ { \mathrm { A c c } } ( \mathcal { A } ^ { * } ) } \\ { R _ { \mathrm { T r a i n } } \bigl ( \mathcal { A } ^ { \prime } \bigr ) \geq R _ { \mathrm { T r a i n } } ( \mathcal { A } ^ { * } ) } \end{array} \right. . } \end{array}\tag{19}
$$

By (19), HFL-PBRL ensures automatic discovery of parameter combinations of $f _ { \mathrm { a g g } }$ and $N _ { \mathrm { a g g } }$ where communication efficiency, model accuracy, and training latency are mutually optimized—a crucial capability given their inherent tradeoffs in hierarchical learning structures. The complete collection of these nondominated solutions forms the Pareto front

$$
P F = \{ \mathcal { A } \in \Omega \mid \nexists \mathcal { A } ^ { \prime } \in \Omega \colon R \big ( \mathcal { A } ^ { \prime } \big ) \succ R ( \mathcal { A } ) \}\tag{20}
$$

where $R ( \mathcal { A } ^ { \prime } ) \succ R ( \mathcal { A } )$ denotes strict improvement in at least one objective without degradation in others. The Pareto front construction is particularly important because it: 1) preserves all optimal tradeoff possibilities for dynamic policy selection; 2) eliminates the need for premature weight assignments between objectives; and 3) provides theoretical guarantees that no superior solutions exist outside the identified frontier. Thus, our approach is well-suited for scenarios where network conditions and computational resources fluctuate unpredictably.

Step 4 (High-Level Policy Selection): The high-level policy $\pi _ { H }$ dynamically selects optimal synchronization parameters from the Pareto front PF to adapt to changing system conditions. This selection process is formalized as follows:

$$
\begin{array} { r l r } {  { \pi _ { H } \bigl ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } | s _ { H } \bigr ) } } \\ & { } & { = \arg \operatorname* { m a x } _ { ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } ) \in P F } \sum _ { i \in { \bf R } _ { H } ( \mathcal { A } ) } \theta _ { i } ( s _ { H } ) R _ { i } \bigl ( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } \bigr ) . } \end{array}\tag{21}
$$

The dynamic weights $\theta _ { i } ( s _ { H } )$ , where $\theta _ { i } \in [ \theta _ { \mathrm { C o m } } , \theta _ { \mathrm { A c c } } , \theta _ { \mathrm { T r a i n } } ] ,$ systematically quantify current system priorities according to the high-level state $s _ { H } .$ . Specifically, $\theta _ { \mathrm { C o m } } ( s _ { H } )$ scales with available bandwidth $B _ { \mathrm { a v g } }$ to emphasize communication efficiency during network congestion, while $\theta _ { \mathrm { { A c c } } } ( s _ { H } )$ increases during critical learning stages to prioritize model accuracy. Conversely, $\theta _ { \mathrm { T r a i n } } ( s _ { H } )$ increases steadily when focusing the optimization on latency reduction. This adaptive weighting mechanism enables our framework to automatically prioritize its optimization focus between competing objectives based on real-time training requirements.

Step 3 [Action Solution $( { \mathcal { A } } _ { H } ) { \mathit { l } } .$ The high-level policy produces a set of candidate actions, offering a balanced tradeoff across the global objectives. This action set is defined as

$$
\mathcal { A } _ { H } = \{ \left( f _ { \mathrm { a g g } } ^ { ( i ) } , N _ { \mathrm { a g g } } ^ { ( i ) } \right) | i = 1 , 2 , \ldots , | P F | \}\tag{22}
$$

where $f _ { \mathrm { a g g } } ^ { ( i ) }$ and $N _ { \mathrm { a g g } } ^ { ( i ) }$ represent the aggregation frequency and rounds, respectively, for each candidate solution $i , | P F _ { H } |$ is the total number of solutions on the Pareto front. Upon completing high-level policy optimization, local training begins, while the edge server adheres to the designated local synchronization frequencies and rounds.

2) Low-Level Action Optimization With Pareto Efficiency: The low-level policy $\pi _ { L }$ implements conditional adaptation at edge servers. When local network conditions deteriorate significantly, $\pi _ { L }$ generates Pareto-optimal adjustments to aggregation parameters by optimizing weighted local rewards. We consider the maximization over the low-level Pareto front $( P F _ { L } )$ to ensures adaptations respect both current edge conditions and global training objectives as follows:

$$
\pi _ { L } ( a | s _ { L } , f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } ) = \arg \operatorname* { m a x } _ { \mathcal { A } _ { L } \in P F _ { L } } \sum _ { j } \theta _ { j } R _ { j }\tag{23}
$$

where $\theta _ { j }$ represents the dynamic local weights based on realtime conditions. Subsequently, localized adjustments are made, where the extent is determined by $\mathcal { Y } _ { m } = \{ f _ { \mathrm { a g g } } ^ { \mathrm { a d j } } , N _ { \mathrm { a g g } } ^ { \mathrm { a d j } } \}$ . Here, $f _ { \mathrm { a g g } } ^ { \mathrm { a d j } }$ and $N _ { \mathrm { a g g } } ^ { \mathrm { a d j } }$ are the adjustments to the aggregation frequency and rounds, respectively. The resulting action $\mathcal { A } _ { L } ( f _ { \mathrm { a g g } } ^ { \mathrm { a d j } } , N _ { \mathrm { a g g } } ^ { \mathrm { a d j } } )$ triggered by low-level policy is defined as

$$
\begin{array} { r } { A _ { L } \Big ( f _ { \mathrm { a g g } } ^ { \mathrm { a d j } } , N _ { \mathrm { a g g } } ^ { \mathrm { a d j } } \Big ) = \left\{ \begin{array} { l l } { \hfill A _ { H } \hfill \pm \mathcal { V } _ { m } ( \cdot ) , \hfill \mathrm { ~ f o r ~ } \delta ( \cdot ) < \delta _ { \mathrm { m i n } } } \\ { \hfill A _ { H } , \hfill \mathrm { f o r ~ } \delta ( \cdot ) \geq \delta _ { \mathrm { m i n } } } \end{array} \right. } \end{array}\tag{24}
$$

where $\delta _ { \mathrm { m i n } }$ denotes the threshold value for changes in the training state. The resulting adjustments, $\mathcal { \partial } _ { m } ( \cdot )$ , at the edge servers lead to an increase or decrease in $N _ { \mathrm { a g g } }$ , or to either shortening or postponing the aggregation in the case of $f _ { \mathrm { a g g } }$

In principle, HFL-PBRL employs a dynamic joint optimization strategy, where RL actions remain conditionally independent across training stages. This adaptive approach enables two distinct operational modes: 1) isolated modification of either $f _ { \mathrm { a g g } }$ or $N _ { \mathrm { a g g } } ;$ and 2) simultaneous optimization of both parameters. The policy strategically selects actions by sampling Pareto-optimal solutions from the frontier, guided by real-time reward feedback and the network condition quantified by the KPI $\delta ( \cdot )$

Algorithm 1 Pareto-Efficiency-Based RL Policy Optimization   
1: Input: $\delta _ { \mathrm { m i n } } ( \cdot )$ , parameters $\gamma , \gamma _ { L } , \beta , \beta _ { L } , \alpha , \alpha _ { L }$   
2: Output: Pareto-optimal actions $A _ { H } , A _ { L } .$   
3: Initialize: Feasible solutions $\Omega = \emptyset .$ , Pareto front $P F = \emptyset$   
4: function HIGHLEVELOPTIMIZATION(πH)   
5: Generate candidate solutions $\boldsymbol { \mathcal { A } } _ { H }$ within   
6: for each $( f _ { a g g } , N _ { a g g } )$ in feasible solution space do   
7: Update reward vector ${ \bf R } _ { H }  { \bf R } _ { t o t a l }$   
8: $\Omega \gets \Omega \cup \{ ( f _ { a g g } , N _ { a g g } , R _ { H } ) \}$   
9: $\mathbf { R } _ { H } ( \mathcal { A } ) $ Evaluate objective function by Eq. (18)   
10: $P F _ { H } \gets$ Identify Pareto-optimal solutions from   
11: $( f _ { a g g } ^ { * } , N _ { a g g } ^ { * } )$ ← Select optimal solution from $P F _ { H }$   
12: Optimize policy $\pi _ { H } ( f _ { a g g } , N _ { a g g } | s _ { H } )$ by Eq. (21)   
13: Action $ ( f _ { a g g } ^ { * } , N _ { a g g } ^ { * } )$   
14: function LOWLEVELOPTIMIZATION(sL, AH)   
15: Define adjustment function $\mathcal { V } _ { m } ( \cdot )$   
16: Collect local state information $s _ { L }$ as in Eq. (12)   
17: if the condition $\{ \delta ( \cdot ) < \delta _ { \mathrm { m i n } } \}$ then   
18: Update reward $\mathbf { R } _ { L }$ with state vector $s _ { L }$   
19: $P F _ { L } \gets$ Identify Pareto-optimal adjustments   
20: $( f _ { a g g } ^ { a d j } , N _ { a g g } ^ { a d j } )$ ← Select adjusted action $P F _ { L }$   
21: Adjust $\ddot { \mathcal { A } } _ { L }  \mathcal { A } _ { H } \pm \mathcal { y } _ { m } ( \phi )$ as in Eq. (24)   
22: else   
23: $( f _ { a g g } ^ { a d j } , N _ { a g g } ^ { a d j } )  ( f _ { a g g } ^ { * } , N _ { a g g } ^ { * } )$   
24: Obtain low-level policy $\pi _ { L } ( a | s _ { L } , f _ { a g g } , N _ { a g g } )$   
25: Action $ ( f _ { a g g } ^ { a d j } , \bar { N } _ { a g g } ^ { a d j } )$   
26: $( f _ { a g g } ^ { * } , N _ { a g g } ^ { * } )$ ← HIGHLEVELOPTIMIZATION(πH)   
27: $( f _ { a g g } ^ { \bar { a } \bar { d } j } , N _ { a g g } ^ { a \bar { d } j } )$ ← LOWLEVELOPTIMIZATION $( \pi _ { L } , A _ { H } )$   
28: Apply $( f _ { a g g } ^ { a d j } , N _ { a g g } ^ { a d j } )$ and $( f _ { a g g } ^ { * } , N _ { a g g } ^ { * } )$ to HFL system

3) Description of Algorithm 1: Our joint optimization framework, illustrated in Algorithm 1, coordinates global and local decision-making through a two-tiered Pareto-optimal selection process. The high-level policy $\pi _ { H }$ operates on the system state sH (11) to generate candidate $( f _ { \mathrm { a g g } } , N _ { \mathrm { a g g } } )$ pairs within the constrained solution space  (17). Each candidate RL action is evaluated through the multiobjective reward vector $\mathbf { R } _ { H }$ by (18). Using the dominance criteria in (19) and (20), it constructs a Pareto front $P F _ { H }$ of nondominated solutions, then selects the optimal action via weighted sum maximization (21) with dynamically adjusted priorities θi. Concurrently, the low-level policy $\pi _ { L }$ monitors local conditions through state $s _ { L }$ defined by (12), triggering adjustments when the training condition, quantified by the KPI metric δ in (13), falls below threshold $\delta _ { \mathrm { m i n } } .$ . Specifically, global decisions derived from $s _ { H }$ are refined locally with $\pi _ { L }$ using the low-state vector $s _ { L } ,$ with the composite network metric δ (13) serving as the primary adaptation trigger. It computes adapted actions $( f _ { \mathrm { a g g } } ^ { \mathrm { a d j } } , N _ { \mathrm { a g g } } ^ { \mathrm { a d j } } )$ through adjustment function $\mathcal { V } _ { m } ( \cdot )$ and adaptation rule formulated in (24), ensuring local responsiveness while preserving global alignment through continuous reward recomputation with localized weights $( \alpha _ { L } , \beta _ { L } , \gamma _ { L } )$ . As a result, our approach achieves adaptation to training dynamics while preserving convergence guarantees through the proposed bi-level policy optimization and coordination.

## C. Hierarchical Pullback and Harmonic Weight Assignment for Model Divergence-Fluctuation Control

We propose two adaptive control mechanism for stabilizing the training process. On the one hand, we devise a hierarchical pullback mechanism, which we build upon our previous work [5] to handle the local model divergence. On the other hand, we employ a HWA method, addressing the effects of asynchrony and model fluctuations.

1) Hierarchical Pullback Mechanism: We extends the empirical minimization defined in (1) by incorporating the hierarchical pullback mechanisms into the local training process. This mechanism, as illustrated by (25), seamlessly pulls the locally trained models toward the reference anchor model $Z _ { m } ^ { ( t ) }$ constraining their deviation from the global model. Meanwhile, we introduce a dynamic parameter $\bar { \mu } _ { k } ^ { ( t ) }$ that scales the influence of the anchor model in accordance with the number of aggregation rounds completed and the frequency at which these aggregations occur as follows:

$$
\underset { \mathbf { w } _ { k } } { \arg \operatorname* { m i n } } \ \mathcal { H } ( \mathbf { w } ) = \mathcal { F } _ { k } ( \mathbf { w } ) + \frac { \mu _ { k } ^ { ( t ) } } { 2 } \Big \| \mathbf { w } _ { k } ^ { ( t ) } - Z _ { m } ^ { ( t ) } \Big \| ^ { 2 }
$$

$$
\mu _ { k } ^ { ( t ) } = ( 1 / \Bigl ( f _ { \mathrm { a g g } } ^ { \ast } + \epsilon _ { f } \Bigr ) + \Bigl ( 1 / N _ { \mathrm { a g g } } ^ { \ast } + \epsilon _ { N } \Bigr )\tag{25}
$$

where $Z _ { m } ^ { ( t ) }$ , defined by (27) is the reference anchor model at each edge server. $\epsilon _ { f }$ and $\epsilon _ { N }$ are small constants to ensure numerical stability. Thus, the mechanism in (25) enhances the strength of the pullback and model alignment in response to the variations of $f _ { \mathrm { a g g } } ^ { * }$ and $N _ { \mathrm { a g g } } ^ { \ast }$ which could introduce inconsistent updates and model divergence.

At designated frequencies $f _ { \mathrm { a g g } } ^ { * } ,$ each edge server receives updates from the associated $C _ { m }$ clients, and performs partial model aggregations as follows:

$$
\begin{array} { r } { \mathbf { w } _ { m } ^ { ( t + 1 ) } = \left\{ \begin{array} { l l } { \frac { \sum _ { k = 1 } ^ { \lfloor C _ { m } \rfloor } | \mathcal { D } _ { k } | \mathbf { w } _ { k } ^ { ( t ) } } { \sum _ { k \in C _ { m } } | \mathcal { D } _ { k } | } , \mathrm { f o r } \Delta t = f _ { \mathrm { a g g } } ^ { * } } \\ { \mathbf { w } _ { m } ^ { ( t ) } , \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{26}
$$

Following the aggregation in (26), each edge server updates its anchor model $\because \bar { Z } _ { m } ^ { ( t ) }$ , which serves as a reference model that is particularly important for clients with low-aggregation frequencies and therefore requires extensive local update rounds. The design the anchor model $Z _ { m } ^ { ( t ) }$ follows the hierarchical training paradigm; edge servers leverage the models received from clients and the central server, creating a unique synergism to devise the anchor model as

$$
\begin{array} { r l r } {  { Z _ { m } ^ { ( t + 1 ) } = Z _ { m } ^ { ( t ) } + \frac { \mu _ { N } ^ { t } } { N _ { \mathrm { a g g } } ^ { * } } \Big ( \mathbf { w } _ { m } ^ { ( t ) } - \mathbf { w } _ { g } ^ { ( t ) } \Big ) } } \\ & { } & { + \mu _ { f } ^ { t } ( \frac { f _ { \mathrm { a g g } } ^ { * } } { f _ { \mathrm { a g g } } ^ { * } + 1 } ) \Big ( \mathbf { w } _ { m } ^ { ( t ) } - \mathbf { w } _ { g } ^ { ( t ) } \Big ) } \end{array}\tag{27}
$$

where $\mu _ { N } ^ { t }$ and $\mu _ { f } ^ { t }$ are tunable parameters to balance the influence of $N _ { \mathrm { a g g } } ^ { \ast }$ and $f _ { \mathrm { a g g } } ^ { * } ,$ respectively. Subsequently, each client employs the updated anchor model $Z _ { m } ^ { ( t + 1 ) }$ to execute pullback-based local optimization as defined in (25).

2) Harmonic Weight Assignment for Asynchronous Global Aggregation: Our bi-level optimization approach fully exploits the diversity of participating clients, yet it has one inherent drawback: differences in aggregation rounds or frequencies can introduce inconsistencies into the resulting model updates. Additionally, the dynamic nature of edge computing, characterized by limited resources and fluctuating performance, would exacerbate this challenge. To address this issue, HFL-PBRL employs a HWA strategy that dynamically assigns aggregation weights to each model, accounting for their recent contributions, historical performance, and anticipated future trends. Specifically, the root server computes the aggregation weight $h _ { m } ^ { ( t ) }$ for each received model following:

```perl
h(t)m = λcQc + λaQa + λf Qf
⎧⎪ Qc = w(t)g − w(t)m 2,
where ⎨ Qa = -tt =tpre wt g − wt m2 for $t ^ { \mathrm { p r e } } < t ^ { \prime } < t$
⎪⎩ Qf = wg (t) − w(t)m 2 − w(t−1)g − w(t−1) m 2
```

(28)

where, $\lambda _ { c } , \lambda _ { a } ,$ , and $\lambda _ { f }$ represent the weighing factors, regulating the model divergence in terms of the current, historic (accumulated), and the anticipated future performance, respectively. We define $Q _ { c }$ in (28), which quantifies the degree of divergence; hence, $\lambda _ { c } Q _ { c }$ in (28) represents the current contribution of $\mathbf { w } _ { m } ^ { ( t ) }$ . Similarly, the term $\lambda _ { a } Q _ { a }$ in (28) accumulates the model performance from the past training rounds $t ^ { \mathrm { p r e } }$ . This is particularly useful in scenarios where clients experience performance fluctuations due to variations in their computing and communication resources, as well as differing aggregation frequencies and rounds. By accounting for historical performance, we ensure that persistent under-performance or over-performance is reflected in the aggregation weight. Meanwhile, the term $\lambda _ { f } Q _ { f }$ in (28) predicts future performance by considering the rate of change in the model performance. This allows our framework to adjust weights preemptively, ensuring that sudden changes in the model updates are adequately addressed.

After computing the weight $h _ { m } ^ { ( t ) }$ , the global model $\mathbf { w } _ { g } ^ { ( t ) }$ is updated with the newly received model $\mathbf { w } _ { m } ^ { ( t ) }$ as follows:

$$
\mathbf { w } _ { g } ^ { ( t + 1 ) } = \left( 1 - h _ { m } ^ { ( t ) } \right) \mathbf { w } _ { g } ^ { ( t ) } + h _ { m } ^ { ( t ) } \mathbf { w } _ { m } ^ { ( t ) } .\tag{29}
$$

This holistic approach is essential for ensuring steady and robust improvements in the global model, particularly when client updates are asynchronous and fluctuating.

## D. Training Algorithm of HFL-PBRL

He complete training procedure of HFL-PBRL is presented in Algorithm 2, which coordinates three parallel execution threads across the hierarchy. The algorithm begins with global model initialization and broadcast (lines 3 and 4), followed by high-level policy $\pi _ { H }$ determining the optimal aggregation parameters $( N _ { \mathrm { a g g } } ^ { * } , f _ { \mathrm { a g g } } ^ { * } )$ through (21) by generating candidate actions $\boldsymbol { \mathcal { A } } _ { H }$ within the Pareto front $P F$ as in (22) (line 5). In client-side training (Thread 2), local updates are performed using the hierarchical pullback mechanism from (25) to constrain model divergence (lines 23–25). Clients synchronize with their edge server at the optimized frequency $f _ { \mathrm { a g g } } ^ { * }$ (lines

Algorithm 2 Training Procedure of HFL-PBRL   
1: Input: T, K, δ(·), learning rate η   
2: Initialize: $P F = \varnothing , \delta _ { \operatorname* { m i n } } .$ global model ${ \bf w } _ { g } ^ { 0 }$   
3: for each global round $t = 1 , 2 , \dots , T$ do   
4: Root server broadcast the global model ${ \bf w } _ { g } ^ { 0 }$   
5: Assign $\mathcal { A } _ { H } ( N _ { a g g } ^ { * } , f _ { a g g } ^ { * } )$ by policy $\pi _ { H }$ through high-  
6: -level action (k ← ClientUpdate $\scriptstyle { \mathcal { A } } _ { H }$ via Eqs. (21) and (22). $\{ \bar { D } _ { k } , \mathbf w _ { m } ^ { ( t ) } , Z _ { m } ^ { ( t ) } , f _ { a g g } ^ { * } \}$   
7: $\ddot { \mathbf { u } } \ddot { \mathbf { f } } \delta ( \cdot ) < \delta _ { \operatorname* { m i n } }$ then  Compute δ(·) via Eq. (13)   
8: Adjust $\mathcal { A } _ { L } ( f _ { a g g } ^ { a d j } , N _ { a g g } ^ { a d j } )  \bar { \mathcal { A } _ { H } } \pm \mathcal { y } _ { m } ( \cdot )$ by Eq. (24)   
9: $\mathbf { w } _ { m } ^ { ( t ) } \gets \mathrm { E d g e }$ server execute THREAD 1   
10: $\mathbf { w } _ { g } ^ { ( t ) }$ ← Root server execute THREAD 3   
11: Return global model $\mathbf { w } _ { g } ^ { ( t ) }$   
12: Intermediate Edge Server Executes  THREAD 1   
13: for $\Delta t \in f _ { a g g } ^ { * }$ do  By policies Eq. (21) and Eq. (23).   
14: if $N _ { a g g } ^ { ( t ) }$ mod $N _ { a g g } ^ { * } \neq 0$ then   
15: Receive clients updates $\begin{array} { r } { \mathbf { w } _ { k } ^ { ( t ) } , k = 1 , 2 , \cdots | \mathcal { C } _ { m } | . } \end{array}$   
16: Aggregation $\begin{array} { r } { \mathbf { w } _ { m } ^ { ( t + 1 ) } = \sum _ { k = 1 } ^ { | \tilde { C } _ { m } | } \frac { n _ { k } } { n } \mathbf { w } _ { k } ^ { ( t ) } } \end{array}$ by Eq. (26).   
17: Update anchor model $Z _ { m } ^ { ( i + 1 ) }$ by Eq. (27)   
18: Send aggregated model $\mathbf { w } _ { m } ^ { ( t + 1 ) }$ to the clients.   
19: if $N _ { a g g } ^ { ( t ) }$ mod $N _ { a g g } ^ { * } = 0$ then  Optimal no. of Agg.   
20: Send $\mathbf { w } _ { m } ^ { ( t + 1 ) }$ to global server and pull new $\mathbf { w } _ { g } ^ { ( t + 1 ) }$   
21: Client Training and Model Updating  THREAD 2   
22: for clients $k \in \{ 1 , 2 , \dots , | \mathcal { C } _ { m } | \}$ do   
23: Collect random samples $\xi _ { j }$ for gradient computation.   
24: Local iteration: $\mathbf { w } _ { k } ^ { ( t + 1 ) }$ ← arg min $\mathcal { H } ( \mathbf { w } )$ by Eq. (25)   
25: if $\Delta t$ mod $f _ { a g g } ^ { * } = 0$ then   
26: Push $\mathbf { w } _ { k } ^ { ( t ) }$ to edge server m and obtain $\mathbf { w } _ { m } ^ { ( t + 1 ) }$   
27: Root Server Task Execution  THREAD 3   
28: if $N _ { a g g } ^ { ( t ) }$ mod $N _ { a g g } ^ { * } = 0$ then   
29: Receive model $\mathbf { w } _ { m } ^ { ( t ) }$ from a edge server m.   
30: Compute harmonic weight $h _ { m } ^ { ( t ) }$ by Eq. (28).   
31: Server update $\mathbf { w } _ { g } ^ { ( t + 1 ) }  \{ h _ { m } ^ { ( t ) } , \mathbf { w } _ { m } ^ { ( \tilde { t } ) } \}$ via Eq. (29).   
32: Send $\mathbf { w } _ { g } ^ { ( t + 1 ) }$ back to edge server and clients.   
33: Return final global model $\mathbf { w } _ { g } ^ { * } .$

26–28), where the pullback term $\mu _ { k } ^ { ( t ) }$ in (25) dynamically adapts to the current aggregation schedule.

Edge servers (Thread 1) aggregate received models according to (26) (lines 14–16) while maintaining the anchor model $Z _ { m } ^ { ( \tilde { t } ) }$ as specified in (27) (line 17). Meanwhile, the lowlevel policy $\pi _ { L }$ monitors the network condition metric δ(·) from (13) (lines 7 and 8), triggering localize adjustments ${ \mathcal { V } } _ { m } ( \cdot )$ through (24) when the training condition falls below certain threshold, i.e., at $\delta ( \cdot ) \ < \ \delta _ { \mathrm { m i n } }$ . This enables dynamic adaptation to local network fluctuations while preserving global optimization objectives. Once the optimal rounds of intermediate aggregations is attained, as determined by lowlevel $\pi _ { L }$ through (23), the aggregated edge server model $\mathbf { w } _ { m } ^ { ( t + 1 ) }$ is sent to the root server (lines 19 and 20). Global aggregation occurs when $N _ { \mathrm { a g g } } ^ { ( t ) }$ mod $N _ { \mathrm { a g g } } ^ { * } = 0$ (lines 28–31), where the root server computes harmonic weights $h _ { m } ^ { ( t ) }$ via (28) and updates the global model following (29). The complete procedure iterates until convergence or completion of T global rounds, where the final global model $\mathbf { w } _ { g } ^ { * }$ is then returned (line 33).

## E. Computational Complexity and Scalability Analysis

The computational efficiency of HFL-PBRL is characterized through rigorous analysis of its core components. The highlevel policy $\pi _ { H }$ operates on a state space $s H$ of dimension $| s _ { H } | = D \left( 1 1 \right)$ , evaluating actions from the discrete space $\mathcal { X } _ { f } \times$ $\mathcal { X } _ { N }$ as in (17). Its computational complexity is governed by the state-action space product, yielding $O ( D \cdot | \mathcal { X } _ { f } | \cdot | \mathcal { X } _ { N } | )$ operations per decision round. This centralized computation occurs at the global server, with complexity independent of the number of clients K. At the edge tier, each of the M servers executes a low-level policy πL processing localized state information $s _ { L } .$ . The per-edge complexity scales as $O ( | s _ { L } | \cdot | \mathcal { V } _ { m } | )$ , where $| s _ { L } |$ denotes the local state dimension and $| { \mathcal { D } } _ { m } |$ represents the action adjustment space.

The Pareto optimization component maintains solution efficiency through nondominated sorting of candidate actions. The computational burden is determined by the maintained Pareto front size |PF| and candidate solution count ||, following O(|| log ||) complexity. Through systematic clustering and warm-start mechanisms, we bound both  and |PF| to constant values independent of clients size K. Meanwhile, the model alignment introduces $O ( M \cdot S )$ global operations and $O ( | \mathcal { C } _ { m } |$ S) per-edge costs for pullback and anchor computation in (25) and (27), respectively. The communication scales as $O ( K S / f _ { \mathrm { a g g } } ^ { * } )$ for client-edge exchanges and $O ( M S / N _ { \mathrm { a g g } } ^ { * } )$ for edge-global synchronization, with RL-optimized frequencies balancing cost and convergence. Compared to FedAvg [2], HFL-PBRL maintains the same $O ( E _ { k } \cdot L \cdot B )$ client-side computation while distributing edge-tier operations from O(K) to $O ( K / M )$ . The global server’s M-dependent scaling enables support for large K through edge server provisioning, demonstrating superior scalability.

## V. EVALUATION

In this section, we first substantiate the significance of our key contributions, and then presents the performance comparison of our framework against the well-established baselines and SOTAs.

## A. Experimental Settings

Evaluation Platform: In this work, all experiments were conducted using a PyTorch framework on a system running Ubuntu 20.04.3 LTS. We simulate a hierarchical learning system on a deep learning workstation supplied with an Intel Xeon Gold 5218R CPU @ 2.10-GHz, featuring two sockets with 20 cores per socket, and four NVIDIA GeForce RTX 4090 GPUs. The system is configured with 256 GB of RAM.

Parameter Settings: During the training, we utilized stochastic gradient descent (SGD) optimizer for parameter updating, the local batch size is 64, selecting 10% of the clients in each training round. The learning rate is set to 0.001 for MNIST and 0.01 for Fashion-MNIST (FMNIST) and CIFAR-10. To create variability, the network topology is formed randomly, with a 0.15 chance of direct connections with client devices, and each link has a range of bandwidth values. Additionally, the threshold for the key network performance metric, $\delta _ { \mathrm { m i n } }$ is varied between 0.35 and 0.45 at different stages of the training.

For the bi-level policy optimization, our configuration favored system-wide performance, setting the associated weights $\gamma , \beta ,$ and α—defined, respectively, in (14), (15), and (16)—all to 0.7. Consequently, the corresponding parameters $( \gamma _ { L } , \beta _ { L } ,$ and $\alpha _ { L } )$ , which prioritize local performance improvements, were initially set to 0.3. These parameters are independently adjusted, guiding our bi-level RL process toward Pareto optimality. In the global aggregation process, we maintained the values of $\lambda _ { c } , \lambda _ { a } .$ , and λf —defined in (28)— at 0.5, 0.3, and 0.2, respectively, ensuring effective harmonic weighting. Unless specified otherwise, the above experimental configurations remain unchanged.

Models and Dataset: In the experiments, we trained a convolution neural network (CNN) model on popular datasets, namely, MNIST [24], FMNIST [25], and CIFAR-10 [26]. To diversify the training data, we utilize Dirichlet distribution, as in [27], to generate nonindependent and identically distributed (non-IID) data.

Baselines: We compare HFL-PBRL with the following.

1) HierFAVG [4] introduced a training method for HFL with reduced global aggregation frequency with intermediate aggregations. However, HierFAVG lacks a specialized mechanism to determine optimal frequencies or rounds and their work primarily focuses on homogeneous settings.

2) FedAda [6] is a recent SOTA framework that introduced adaptive aggregation frequency adjustment based on model convergence and heterogeneous capabilities of the training devices. However, its approach does not account for the number of intermediate aggregation rounds, nor does it integrate simultaneous multiobjective optimization. These limitations present opportunities for further innovation.

3) RAF [9] allows devices with different capabilities to use different aggregation frequencies. It employs a heuristic to compute aggregation frequencies, setting the frequency of the slowest client and edge server to one and adjusting the frequencies of other nodes based on these benchmarks.

4) AutoSF [8] is a SOTA framework that utilizes an automated machine learning approach to dynamically adapt aggregation structures and frequencies in timevarying edge environments. Thus, the introduction of AutoSF highlighted the rapid pace of innovation in this domain, making it a relevant baseline for comparison.

## B. Performance Evaluation of System Components

In this section, we validate our key insights to substantiate the rationale behind the HFL-PBRL design approach.

1) Examining the Effects Bi-Level RL Approach: We executed ablation experiments to investigate the impact of our bi-level RL algorithm, by comparing it against two constrained variants: 1) a high-level-only (πH) approach and 2) a lowlevel-only (πL) approach. We compared these variants in terms of both accuracy and communication cost, as illustrated in Fig. 2.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 2. Comparing the performance of our bi-level RL method against the constrained variants. Task: CNN on CIFAR-10. (a) Test Accuracy. (b) Communication Cost.

The high-level policy method, with its globally optimized aggregation control solution, attained a competitive test accuracy but often fell short of the bi-level variant as shown in Fig. 2(a). Its slower response to local changes led to periods of stagnation, where accuracy improvements plateaued early. On the other hand, the low-level-based method πL demonstrated rapid improvement, especially in the initial rounds of training. Nevertheless, this approach of locally optimizing aggregation rounds and frequencies at individual edge servers, without global objective-based coordination, would eventually lead to inefficient training and degrade the overall performance. The bi-level framework, in contrast, harnessed the synergy between high-level and low-level policy networks to address both global and local training demands. This unique synergy enhanced the learning process, resulting in expedited convergence and higher accuracy compared to the others.

The comparison of communication cost is illustrated in Fig. 2(b). At the 65% accuracy benchmark, the bi-level approach achieved the lowest communication cost. Both the high-level and low-level methods incurred approximately 1.5× more cost than the bi-level algorithm. Similarly, at a higher accuracy threshold, that is 70%, the performance margin in terms of communication cost became even more pronounced. The high-level method continued to require about 1.5×–1.6× more resources than the bi-level system. More strikingly, the low-level variant incurred a communication cost 3× higher than the bi-level method under higher accuracy demand. These findings underscore the inherent tradeoffs of merely optimizing the global or local objectives, highlighting that neither could match the performance of the bi-level approach.

2) Impact of Hierarchical Pullback and Harmonic Weight Assignment: We quantified the effectiveness of the proposed hierarchical pullback (Pb) mechanism and HWA method in an ablation study summarized in Table II. We compared four configurations of our framework: 1) Base model, which used neither pullback nor HWA; 2) HWA model, which applied only the HWA mechanism but no divergence control; 3) Pb model, which integrated only the pullback strategy without incorporating the HWA; and 4) Pb+HWA, which combined both strategies into our complete framework. We utilized a CNN framework on CIFAR-10 dataset under non-IID conditions, adhering to the experimental configurations outlined in Section V-A.

TABLE II  
EVALUATING THE SIGNIFICANCE OF HIERARCHICAL PULLBACK AND HWA USING TEST ACCURACY, COMMUNICATION COST, AND TIME TO DELIVER THE TARGET ACCURACIES
<table><tr><td rowspan="2">System Variants</td><td rowspan="2">Test Accuracy</td><td colspan="2">Comm. Cost (GB)</td><td colspan="2">Conv. Time (s)</td></tr><tr><td>70%</td><td>75%</td><td>70%</td><td>75%</td></tr><tr><td>Base model</td><td>76.06%</td><td>45.03</td><td>99.67</td><td>4129.56</td><td>10772.6</td></tr><tr><td>HWA model</td><td>77.06%</td><td>38.57</td><td>86.23</td><td>2977.34</td><td>4930.43</td></tr><tr><td>Pb model</td><td>74.46%</td><td>41.02</td><td>77.68</td><td>1013.79</td><td>1694.65</td></tr><tr><td>Pb+HWA</td><td>79.18%</td><td>34.73</td><td>71.39</td><td>937.21</td><td>1589.62</td></tr></table>

As shown in Table II, the Base model achieved a test accuracy of 76.06%, requiring the highest overall communication and the longest time to converge. Its lack of mechanisms for mitigating divergence or adaptively weighting updates makes it a suitable benchmark for comparison. By applying harmonic weighting in the global aggregation, the HWA variant reduced its communication cost to about 1.2× lower at 70% accuracy and 1.16× lower at 75% accuracy compared to Base model. It also converged about 1.4× faster at the 70% mark and more than 2× faster at the 75% mark. This underscores how adaptive weighting improved early-stage efficiency, though local divergence could still slow progress toward higher accuracy.

Conversely, the pullback variant (Pb model) constrained model divergence during client-side training but did not weight updates based on client performance. Although its accuracy was slightly lower than HWA variant, the Pb variant cut the communication cost to around 1.1× lower at 70% and 1.3× lower at 75% compared to the base model. We emphasize that, in this case, the accuracy progression is significantly faster in the early training phase, reaching 70% accuracy more than 4× faster and 75% accuracy approximately 6× faster than the Base model. This phenomenon indicates that local alignment via pullback mechanism greatly accelerated convergence.

By combining both strategies, Pb+HWA incurred a communication cost which is 1.3× lower at 70% and 1.4× lower at 75% compared to the Base model, while also converging around 4× faster at 70% and nearly 7× faster at 75%. These marked gains highlight how hierarchical pullback and HWA collectively stabilize the learning process and improve the overall model performance.

3) Evaluating the Impact of Aggregation Frequency and Rounds: To validate the significance of our key insights, we further present an ablation study of our framework as illustrated in Figs. 3 and 4. We generated four variants of our model, each constrained in a specific way to establish baseline comparisons. These constraints involve setting either a fixed aggregation frequency (low and high) or a fixed number of aggregations (small and large). Specifically, for Type 1-1 and Type 1-2, we varied the aggregation frequencies while keeping the number of aggregations fixed to a low-value for

<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 3. Test accuracy comparison of our “Complete” framework versus model variants. Task: CNN on (a) MNIST and (b) CIFAR-10.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 4. Convergence time comparison of HFL-PBRL versus model variants. Task: CNN on (a) MNIST and (b) CIFAR-10.

Type 1-1 and high-value for Type 1-2. In the case of Type 2- 1, the number of intermediate aggregations is dynamic while maintaining a fixed, low-aggregation frequency. Conversely, Type 2-2 involved varying the aggregation frequency while keeping the number of aggregations constant to a large value. In contrast, the “Complete” variant represents our fully optimized model, dynamically adjusting both the aggregation frequency and rounds based on training condition.

While all the compared methods achieved a comparable performance, the accuracy of “Complete” is adequately higher than others as evidenced in Fig. 3. Thus, adjusting these factors optimize performance by aligning system parameters with evolving training conditions, leading to an improved training efficiency. In contrast, maintaining a fixed aggregation frequency or rounds may improve one aspect but compromise others, as shown by the other variants.

This effect is particularly evident in the case of Type 1- 1 and Type 1-2, where we dynamically vary the aggregation frequency and set the aggregation number fixed. The highaccuracy of Type 1-2 over Type 1-1 as in Fig. 3 demonstrates the advantage of allowing large number of intermediate aggregations at the edge servers. This allows clients to perform more rounds of local computation and subsequent model synchronization, leading to improved accuracy. However, as a result of increased communication between clients and edge servers, the communication cost of Type 1-2 increased as shown in Fig. 5(a). However, as we move to CIFAR-10 dataset, the complexity of data and task intensifies. Hence, the extremely small number of intermediate aggregation for Type 1-1 becomes insufficient for fast performance improvement, requiring additional training rounds to deliver the targeted accuracies. This led to the increase in communication cost of

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 5. Communication cost comparison of HFL-PBRL versus model variants. Task: CNN on (a) MNIST and (b) CIFAR-10.

Type 1-1 over Type 1-2 in Fig. 5(b). We would emphasize that, under a fixed global training round, the communication cost of Type 1-2 would generally be higher than Type 1-1 due to the increasing communication at the clients-edge server interface.

Comparing Type 1-2 and Type 2-2 variants, both utilizing constant aggregation frequency, Type 2-1 has a high-accuracy improvement and faster convergence compared to Type 2-2 as shown in Figs. 3 and 4, respectively. Furthermore, in the case of Type 2-1, the tradeoff is made between convergence time and communication cost. With the extended aggregation frequency, Type 2-2 frequently collects local models for aggregation, thereby shortening the local computation time for the clients. In principle, very frequent aggregation increases the training time and as well incurs extra communication cost, leading to the overall increase in convergence time and communication cost of Type 2-2 compared to Type 2-1 and “Complete” as shown in Fig. 5. These findings validate the significance of our key contributions; highlighting the multidimensional impact of aggregation frequency and rounds. These results also underscore the necessity of incorporating multiobjective optimization to handle the intricate relationship between aggregation parameters and performance.

## C. Performance Comparison With Baselines

1) Test Accuracy Comparison With Baselines: We conducted extensive experiments to examine the performance of our framework against established baselines by considering non-IID data distribution. The results in Fig. 6(a)–(c) shows a consistent trend across all datasets is the significantly faster initial convergence achieved by HFL-PBRL. For instance, on the MNIST dataset, where the baselines exhibit only a marginal difference in accuracy, HFL-PBRL obtained 90–98% test accuracy within the first 100 training rounds, significantly outperforming baselines, as shown in Fig. 6(a). HierFAVG, for instance, reached only 80% accuracy during the same period and requires at least 200 rounds to approach comparable performance. Likewise, RAF and FedAda lag further behind, stabilizing at 75–80% accuracy by 100 rounds. AutoSF performed better than these methods but still requires additional training rounds to cross the 90% threshold.

On the FMNIST dataset, our framework continues to showcase its adaptability and efficiency. As illustrated in Fig. 6(b), HFL-PBRL achieved 85% accuracy within 200 communication rounds, surpassing baselines, such as HierFAVG and RAF, which reached only 70–75% accuracy during the same period. FedAda performed slightly better, achieving 80% accuracy by 300 rounds, while AutoSF lagged marginally behind, reaching 82–85% accuracy by 400 rounds. HFL-PBRL, however, obtained 90% accuracy after 400 rounds, exceeding the best performance of the baselines. Furthermore, on the CIFAR-10 dataset, our method attained substantial improvements over the baselines, as shown in Fig. 6(c). Precisely, HFL-PBRL surpassed 75% test accuracy within 400 communication rounds, significantly outperforming HierFAVG and AutoSF, which reached plateau at 65–70% accuracy during the same period. FedAda performed slightly better, achieving 70% accuracy, but failed to improve significantly beyond this accuracy level. Meanwhile, RAF achieved the lowest accuracy, stabilizing at around 60–65% accuracy. These findings collectively demonstrate the consistent superiority of our framework over the baselines in terms of test accuracy.

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
（c）

Fig. 6. Test accuracy comparison of HFL-PBRL against the baselines. Task: CNN Model on (a) MNIST, (b) FMNIST, and (c) CIFAR-10. TABLE III  
COMPARISON OF COMPUTATIONAL COST WITH DEVICE SCALABILITY. TASK: CNN MODEL ON CIFAR-10 (NON-IID FACTOR IS 0.8)
<table><tr><td></td><td colspan="2">50 Devices (65%)</td><td colspan="2">200 Devices (65%)</td><td colspan="2">500 Devices (55%)</td><td colspan="2">1000 Devices (55%)</td></tr><tr><td>Method</td><td>TFLOPs</td><td>Round Time(s)</td><td>TFLOPs</td><td>Round Time(s)</td><td>TFLOPs</td><td>Round Time(s)</td><td>TFLOPs</td><td>Round Time(s)</td></tr><tr><td>HierFAVG</td><td>8.99</td><td>5811</td><td>45.85</td><td>6311</td><td>10.10</td><td>6394</td><td>42.56</td><td>6526</td></tr><tr><td>RaF</td><td></td><td> $6 . 7 9 ( - 2 . 2 ) \uparrow 1 8 0 6 ( \times 3 . 2 2 ) \uparrow$ </td><td></td><td> $4 8 . 0 9 ( + 2 . 2 ) \downarrow 1 5 8 1 ( \times 3 . 9 9 ) \uparrow$ </td><td>58.55(+48.5) ↓</td><td> $1 7 9 4 ( \times 3 . 5 7 ) \uparrow$ </td><td> $8 7 . 0 0 ( + 4 4 . 4 ) $ </td><td>↓1931(×3.38) ↑</td></tr><tr><td>FedAda</td><td></td><td> $1 3 . 8 9 ( + 4 . 9 ) \downarrow 1 7 7 8 ( \times 3 . 2 7 ) \uparrow$ </td><td></td><td> $4 8 . 5 9 ( + 2 . 7 ) \downarrow 1 7 8 6 ( \times 3 . 5 3 ) \uparrow$ </td><td>60.40(+50.3)↓</td><td> $1 8 1 4 ( \times 3 . 5 2 ) \uparrow$ </td><td> $8 8 . 7 2 ( + 4 6 . 2 ) ~ .$ </td><td>1926(×3.39)↑</td></tr><tr><td>AutoSF</td><td></td><td> $4 . 6 7 ( - 4 . 3 ) \uparrow 1 3 3 4 ( \times 4 . 3 6 ) \uparrow$ </td><td></td><td> $6 . 4 4 ( - 3 9 . 4 ) \uparrow 1 3 1 2 ( \times 4 . 8 1 ) \uparrow$ </td><td>6.65(-3.5) 个</td><td> $1 3 8 9 ( \times 4 . 6 0 ) \uparrow$ </td><td> $1 2 . 7 3 ( - 2 9 . 8 ) \uparrow$ </td><td>1211(×5.39) ↑</td></tr><tr><td>HFL-PBRL</td><td></td><td> $8 . 7 3 ( - 0 . 3 ) \uparrow 1 9 6 2 ( \times 2 . 9 6 ) \uparrow$ </td><td> $6 . 1 5 ( - 3 9 . 7 ) \uparrow 7 3 4 ( \times 8 . 6 0 ) \uparrow$ </td><td></td><td>6.01(-4.1)↑</td><td> $7 6 7 ( \times 8 . 3 4 ) \uparrow$ </td><td></td><td>6.24(-36.3) ↑ 794(×8.22) ↑</td></tr></table>

↑ indicates improvement (lower TFLOPs or shorter time） vs HierFAVG.↓indicates worse performance.  
(±x） shows absolute TFLOPs diffrence.(×y） shows speedup factor (baseline time/method time).

2) Evaluation of Computational Efficiency: We conducted a series of experiments evaluating computational efficiency using two key metrics: 1) aggregate floating-point operations (FLOPs) and 2) wall-clock training time. Target accuracies were scaled with system size (65% for 50-200 devices; 55% for 500-1000 devices) to account for the increasing data heterogeneity. As evidenced in Table III, the results demonstrate HFL-PBRL’s remarkable computational efficiency, requiring only 6.15 TFLOPs for 200 devices compared to HierFAVG’s 45.85 TFLOPs, representing an 86.6% reduction in computational overhead. This efficiency gain scales progressively, with the 1000-device configuration demonstrating an 85.3% reduction in FLOPs (6.24 versus 42.56 TFLOPs). Crucially, these computational savings do not come at the expense of temporal efficiency. HFL-PBRL demonstrated fast training acceleration, completing rounds 8.60× faster than HierFAVG at $K \_ { } = { }$ 200 (734s versus 6311s) and maintaining 8.22× speedup at K = 1000 (794s versus 6526s). Among the benchmark methods, RAF showed a moderate FLOPs reduction (24.5% at K = 50) but suffered from severe scalability limitations, with computational costs ballooning to 87.00 TFLOPs at $K = 1 0 0 0$ FedAda exhibited similar limitations, demonstrating that existing approaches to aggregation frequency optimization struggle to maintain efficiency at scale. AutoSF presents an interesting case–while achieving notable FLOPs reduction (48.1% at K = 200), its temporal performance proves inconsistent, with speedup factors varying unpredictably from 4.36× to 5.39× across configurations. This instability suggests that singlefactor optimization approaches cannot reliably maintain Pareto efficiency across multiple objectives.

Fundamentally, the Pareto-optimal bi-level optimization explicitly negotiates between competing objectives. At K = 500, for instance, the framework accepts a marginal 4.1% FLOPs reduction from the optimal to achieve an 8.34× speedup, demonstrating efficient tradeoff management. Notably, the performance deltas between HFL-PBRL and baselines widen with system scale, suggesting our framework’s architectural advantages become increasingly significant in large deployments. At K = 50, HFL-PBRL shows modest improvements (2.9% FLOPs reduction, 2.96× speedup), but by K = 1000, it achieves order-of-magnitude better efficiency. This scaling behavior validates the framework’s particular suitability for real-world large-scale deployments, where both computational resources and training time constitute critical constraints.

3) System Scalability: To rigorously evaluate the scalability of HFL-PBRL, we conducted comprehensive experiments comparing our framework against the baselines across varying numbers of edge devices (50, 200, 500, and 1000 clients). As shown in Fig. 7(a)–(d), while all methods exhibited performance degradation with increasing system scale–primarily due to amplified data heterogeneity and coordination overhead, HFL-PBRL demonstrated superior robustness, maintaining consistently higher accuracy across all tested configurations. As shown in Fig. 7(a), HFL-PBRL achieves over 78% accuracy with 50 clients and demonstrates resilience as the system scales, experiencing only marginal degradation of 0.42, 3.9, and 4.79 for 200, 500, and 1000 clients, respectively. In contrast, AutoSF, while competitive at lower client counts, suffered a significant accuracy drop at 1000 clients, as illustrated in Fig. 7(d). Similarly, FedAda performs well with 50 clients (≥74% accuracy), surpassing RAF and HierFAVG, but its performance declined sharply beyond 200 clients, with pronounced degradation at 500 and 1000 clients.

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
（c）

<!-- image-->  
(d)  
Fig. 7. Comparison of device scalability. Task: CNN on CIFAR-10 with non-IID factor set to 0.8. (a) K =50 Clients. (b) K =200 Clients. (c) K =500 Clients. (d) K =1000 Clients.

The superior scalability of HFL-PBRL presented in Fig. 7(a)–(d) can be attributed to its integrated design features. The hierarchical framework with the bi-level synchronization optimization, ensures efficient coordination across large-scale deployments. Additionally, the proposed hierarchical pullback mechanism and HWA mitigate model divergence caused by data heterogeneity and system asynchrony. These components collectively enable HFL-PBRL to maintain stable performance even under extreme scalability conditions, validating the theoretical advantages established in Section IV-E.

4) Comparison of Convergence Speed: We conducted experiments to compare the convergence time of HFL-PBRL with the baseline methods. The experiments utilized a CNN model with accuracy budgets set to 80% and 85% for FMNIST, and 65% and 70% for CIFAR-10 dataset, as observed in Fig. 8. To ensure fairness, all methods were tested under identical configurations as detailed in Section V-A. Considering FMNIST dataset, our framework converged significantly faster than the baseline methods across both accuracy budgets, as shown in Fig. 8(a). At the 80% accuracy budget, HFL-PBRL is 4.1× faster than RAF, 3.3× faster than FedAda, 1.07× faster than HierFAVG, and 1.3× faster than AutoSF. A larger performance gap is observed when the accuracy budget is extended to 85%, where HFL-PBRL achieved convergence that is 4.2× and 2.5× faster than RAF and FedAda, respectively. On comparison to HierFAVG and AutoSF, HFL-PBRL also demonstrates expedited convergence, being 1.05× faster than the former, and 1.5× than the later.

<!-- image-->  
(a)

<!-- image-->  
（b）  
Fig. 8. Convergence speed comparison of HFL-PBRL against the baselines. Task: CNN on (a) FMNIST and (b) CIFAR-10.

Similarly, on the CIFAR-10 dataset, HFL-PBRL continued to showcase substantial improvements, as shown in Fig. 8(b). For example, at the 65% accuracy budget, HFL-PBRL is 2.8× faster than RAF, 2.7× faster than FedAda, 1.6× faster than HierFAVG, and 1.04× faster than AutoSF. At the 70% accuracy budget, despite the convergence efficiency of AutoSF, the performance of HFL-PBRL remained superior. On comparison to the other baselines, the performance gaps is even more pronounced with HFL-PBRL surpassing the 70% accuracy 3.8×, 3.5×, and 2.9× faster than RAF, FedAda and HierFAVG, respectively. The results in Fig. 8(a)–(b) also revealed a crucial limitation in the baselines: handling conflicting objectives during the aggregation frequency optimization. We attribute this phenomenon to the lack of adequate consideration for simultaneous multiobjective optimization in the baseline algorithms.

5) Comparison of Communication Cost: We conducted a series of experiments to validate the communication efficiency of our framework, which demonstrated significant communication savings compared to baseline approaches, as shown in Fig. 9. In particular, at the 80% accuracy benchmark with FMNIST dataset illustrated in Fig. 9(a), HFL-PBRL incurred the lowest communication cost, resulting in an 86.2% and 80% reduction compared to HierFAVG and RAF, respectively. Similarly, compared to AutoSF, HFL-PBRL saved 33.3% in communication cost, while demonstrating a near-equivalent efficiency to FedAda, which also incurred a low-communication cost. However, as the accuracy benchmark increased to 85%, the performance gap among the baselines fluctuates, while the communication efficiency of HFL-PBRL remained superior. Specifically, at the 85%, HFL-PBRL obtained a 88.5% reduction compared to HierFAVG and an 87.5% reduction compared to RAF, shown in Fig. 9(b). Compared to AutoSF, HFL-PBRL cut down the communication costs by 62.5%, while FedAda incurred approximately 1.67× the communication cost of HFL-PBRL.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 9. Comparison of communication cost against the baselines. Task: CNN on (a) FMNIST and (b) CIFAR-10.

On the CIFAR-10 dataset, a similar trend is observed as shown in Fig. 9(b). At the 65% accuracy level, HFL-PBRL incurred a 81.25% lower communication cost than HierFAVG and 31.8% lower than AutoSF. On comparison to a communication efficient framework, FedAda, the communication cost of HFL-PBRL is lower. At a higher accuracy budget of 70%, HFL-PBRL further optimized communication costs, requiring a approximately 84% reduction compared to HierFAVG and RAF and 42.8% lower than AutoSF. In contrast, FedAda incurred about 1.5× cost of HFL-PBRL. The results in Fig. 9(a)–(b) highlight the fine-grained adaptivity of our algorithm that prevents redundant model synchronizations and excessive aggregations. This capability is notably absent in the baseline methods, representing an inefficiency that inadvertently incurred additional communication costs.

## VI. CONCLUSION

In this article, we introduced HFL-PBRL, a unique HFL framework that leveraged a hierarchical training architecture to develop a bi-level RL-based approach for optimizing aggregation frequencies and rounds at intermediate edge servers. The novelty of our algorithm is a Pareto-based multiobjective optimization strategy, ensuring the generation of Paretoefficient solutions for aggregation rounds and frequencies. This approach enabled our algorithm to effectively maintain an optimal tradeoff among communication cost, convergence time, and model accuracy throughout the training process. Additionally, the proposed hierarchical pullback mechanisms facilitated effective divergence control and expedited training via the iterative anchor model synchronization. Complementing this mechanism, the HWA strategy stabilized the global convergence by adjusting aggregation weights based on current, historical, and anticipated model divergence. Extensive experiments demonstrated that HFL-PBRL consistently outperformed baselines in terms of performance accuracy, communication and convergence efficiency.

Future research will focus on extending HFL-PBRL to support multitier architectures, enhancing robustness and performance across diverse FL environments.

## REFERENCES

[1] D. Xu et al., “Edge intelligence: Empowering intelligence to the edge of network,” Proc. IEEE, vol. 109, no. 11, pp. 1778–1837, Nov. 2021.

[2] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. 20th Int. Conf. Artif. Intell. Statist., 2017, pp. 1273–1282.

[3] H. Zhou, M. Li, P. Sun, B. Guo, and Z. Yu, “Accelerating federated learning via parameter selection and pre-Synchronization in mobile edge-cloud networks,” IEEE Trans. Mobile Comput., vol. 23, no. 11, pp. 10313–10328, Nov. 2024.

[4] L. Liu, J. Zhang, S. H. Song, and K. B. Letaief, “Client-edge-cloud hierarchical federated learning,” in Proc. IEEE Int. Conf. Commun. (ICC), 2020, pp. 1–6.

[5] O. Manjang, Y. Zhai, J. Shen, J. Tchaye-Kondi, and L. Zhu, “Anchor model-based hybrid hierarchical federated learning with overlap SGD,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 12540–12557, Dec. 2024.

[6] L. Luo, C. Zhang, H. Yu, G. Sun, S. Luo, and S. Dustdar, “Communication-efficient federated learning with adaptive aggregation for heterogeneous client-edge-cloud network,” IEEE Trans. Services Comput., vol. 17, no. 6, pp. 3241–3255, Nov./Dec. 2024.

[7] J. Tchaye-Kondi, Y. Zhai, J. Shen, A. Telikani, and L. Zhu, “Adaptive period control for communication efficient and fast convergent federated learning,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 12572–12586, Dec. 2024.

[8] L. Yang, Y. Gan, J. Chen, and J. Cao, “AutoSF: Adaptive distributed model training in dynamic edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 6, pp. 6549–6562, Jun. 2024.

[9] L. Yang, Y. Gan, J. Cao, and Z. Wang, “Optimizing aggregation frequency for hierarchical model training in heterogeneous edge computing,” IEEE Trans. Mobile Comput., vol. 22, no. 7, pp. 4181–4194, Jul. 2023.

[10] T. Qi, Y. Zhan, P. Li, J. Guo, and Y. Xia, “Hwamei: A learningbased synchronization scheme for hierarchical federated learning,” in Proc. IEEE 43rd Int. Conf. Distrib. Comput. Syst. (ICDCS), 2023, pp. 534–544.

[11] W. Fang, D.-J. Han, and C. G. Brinton, “Submodel partitioning in hierarchical federated learning: Algorithm design and convergence analysis,” in Proc. IEEE Int. Conf. Commun. (ICC), 2024, pp. 268–273.

[12] Y. Zhou, X. Pang, Z. Wang, J. Hu, P. Sun, and K. Ren, “Towards efficient asynchronous federated learning in heterogeneous edge environments,” in Proc. IEEE Conf. Comput. Commun. (INFOCOM), 2024, pp. 2448–2457.

[13] L. Leconte, V. M. Nguyen, and E. Moulines, “FAVANO: Federated averaging with asynchronous nodes,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2024, pp. 5665–5669.

[14] T. Ortega and H. Jafarkhani, “Quantized and asynchronous federated learning,” IEEE Trans. Commun., vol. 73, no. 4, pp. 2361–2374, Apr. 2025.

[15] X. Yu et al., “Async-HFL: Efficient and robust asynchronous federated learning in hierarchical IoT networks,” in Proc. 8th ACM/IEEE Conf. Internet Things Design Implement., 2023, pp. 236–248.

[16] Y. Deng et al., “A communication-efficient hierarchical federated learning framework via shaping data distribution at edge,” IEEE/ACM Trans. Netw., vol. 32, no. 3, pp. 2600–2615, Jun. 2024.

[17] T. Zhang, K.-Y. Lam, and J. Zhao, “Device scheduling and assignment in hierarchical federated learning for Internet of Things,” IEEE Internet Things J., vol. 11, no. 10, pp. 18449–18462, May 2024.

[18] Y. Jiang, D. Wang, B. Song, and S. Luo, “HDHRFL: A hierarchical robust federated learning framework for dual-heterogeneous and noisy clients,” Future Gener. Comput. Syst., vol. 160, pp. 185–196, Nov. 2024.

[19] S. M. Azimi-Abarghouyi and V. Fodor, “Scalable hierarchical over-theair federated learning,” IEEE Trans. Wireless Commun., vol. 23, no. 8, pp. 8480–8496, Aug. 2024.

[20] Z. Wang, H. Xu, J. Liu, Y. Xu, H. Huang, and Y. Zhao, “Accelerating federated learning with cluster construction and hierarchical aggregation,” IEEE Trans. Mobile Comput., vol. 22, no. 7, pp. 3805–3822, Jul. 2023.

[21] Q. Wu et al., “HiFlash: Communication-efficient hierarchical federated learning with adaptive staleness control and heterogeneity-aware clientedge association,” IEEE Trans. Parallel Distrib. Syst., vol. 34, no. 5, pp. 1560–1579, May 2023.

[22] Z. Feng, X. Chen, Q. Wu, W. Wu, X. Zhang, and Q. Huang, “FedDD: Toward communication-efficient federated learning with differential parameter dropout,” IEEE Trans. Mobile Comput., vol. 23, no. 5, pp. 5366–5384, May 2024.

[23] T. Xiang et al., “Federated learning with dynamic epoch adjustment and collaborative training in mobile edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 5, pp. 4092–4106, May 2024.

[24] L. Bottou, F. E. Curtis, and J. Nocedal, “Optimization methods for largescale machine learning,” SIAM Rev., vol. 60, no. 2, pp. 223–311, 2018.

[25] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms,” 2017, arXiv:1708.07747.

[26] A. Krizhevsky, (Univ. Toronto, Toronto, ON, Canada). Learning Multiple Layers of Features from Tiny Images, 2009. [Online]. Available: http://www.cs.toronto.edu/ kriz/learning-features-2009-TR.pdf

[27] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the objective inconsistency problem in heterogeneous federated optimization,” in Proc. 34th Adv. Neural Inf. Process. Syst., 2020, pp. 7611–7623.

<!-- image-->

Adil Sarwar received the B.Sc. Engineering degree in computer systems from NFC Institute of Engineering and Technological Training, Multan, Pakistan, in 2008, and the M.Phil. degree in computer science from the National College of Business Administration and Economics, Rahim Yar Khan, Pakistan, in 2015. He is currently pursuing the Ph.D. degree with the School of Cyberspace Science and Technology, Beijing Institute of Technology, Beijing, China.

He is currently associated with the School of

Cyberspace Science and Technology, Beijing Institute of Technology. His research interests include edge computing and data privacy.

<!-- image-->  
Ousman Manjang received the B.Tech. degree from the Department of Electronics and Communication Engineering, Jawaharlal Nehru Technological University Anantapur, Anantapur, India, in 2018, and the master’s degree in information and communication engineering from the University of Science and Technology Beijing, Beijing, China, in 2022. He is currently pursuing the Ph.D. degree with the School of Cyberspace Science and Technology, Beijing Institute of Technology, Beijing.

His research interests include federated learning and edge computing.

<!-- image-->

Xutian He received the B.Eng. degree in computer science and technology from Beijing Institute of Technology, Beijing, China, in 2023, where she is currently pursuing the master’s degree with the School of Computer Science and Technology.

Her research interests include federated learning, edge computing, and client scheduling.

<!-- image-->

Yanlong Zhai (Member, IEEE) received the B.Eng. and Ph.D. degrees in computer science from Beijing Institute of Technology, Beijing, China, in 2004 and 2010, respectively.

He is an Associate Professor with the School of Cyberspace Science and Technology, Beijing Institute of Technology. He was a Visiting Scholar with the Department of Electrical Engineering and Computer Science, University of California at Irvine, Irvine, CA, USA. His research interests include cloud computing, big data, and edge computing.

<!-- image-->

Huan Wang (Student Member, IEEE) received the B.Eng. degree in computer science and technology from Anhui University of Science and Technology, Huainan, Anhui, China, in 2020, and the master’s degree in computer science and technology from Xidian University, Xi’an, China, in 2023. He is currently pursuing the Ph.D. degree with the School of Computing and Information Technology, University of Wollongong, Wollongong, NSW, Australia.

His research interests include federated learning, diffusion models, and bioinformatics.

<!-- image-->

Jun Shen (Senior Member, IEEE) received the Ph.D. degree from Southeast University, Nanjing, China, in 2001.

He held positions with Swinburne University of Technology, Melbourne, VIC, Australia, and University of South Australia, Adelaide, SA, Australia, before 2006. He is currently a Full Professor with the School of Computing and Information Technology, University of Wollongong, Wollongong, NSW, Australia. He has published more than 420 papers in journals and conferences in

CS/IT areas. His publications appeared at IEEE TRANSACTIONS ON SERVICE COMPUTING, IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS, and Briefs in Bioinformatics. His research interests include services and cloud computing, computational intelligence, and bioinformatics.

Prof. Shen has been an editor, the PC chair, the guest editor, for numerous journals and conferences. He is a Senior Member of professional institutions, such as ACM. He was also a member of ACM/AIS Task Force on Curriculum MSIS 2016.

<!-- image-->

Liehuang Zhu (Senior Member, IEEE) received the B.Eng. and master’s degrees in computer application from Wuhan University, Wuhan, Hubei, China, in 1998 and 2001, respectively, and the Ph.D. degree in computer application from Beijing Institute of Technology, Beijing, China, in 2004.

He is currently a Professor with the School of Cyberspace Science and Technology, Beijing Institute of Technology, Beijing. He is selected into the Program for New Century Excellent Talents in University from Ministry of Education, China.

His research interests include Internet of Things, cloud computing security, Internet, and mobile security.