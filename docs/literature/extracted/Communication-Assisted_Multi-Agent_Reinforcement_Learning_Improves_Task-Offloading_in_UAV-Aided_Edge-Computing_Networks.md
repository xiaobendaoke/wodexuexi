# Communication-Assisted Multi-Agent Reinforcement Learning Improves Task-Offloading in UAV-Aided Edge-Computing Networks

Siyang Tan, Binqiang Chen , Dong Liu , Member, IEEE, Jianglong Zhang, and Lajos Hanzo , Life Fellow, IEEE

Abstract—Equipping unmanned aerial vehicles (UAVs) with computing servers allows the ground-users to offload complex tasks to the UAVs, but the trajectory optimization of UAVs is critical for fully exploiting their maneuverability. Existing studies either employ a centralized controller having prohibitive communication overhead, or fail to glean the benefits of interaction and coordination among agents. To circumvent this impediment, we propose to intelligently exchange critical information among agents for assisting their decision-making. We first formulate a problem for maximizing the number of offloaded tasks and the offloading fairness by optimizing the trajectory of UAVs. We then conceive a multi-agent deep reinforcement learning (DRL) framework by harnessing communication among agents, and design a communication-assisted decentralized trajectory control algorithm based on value-decomposition networks (VDN) for fully exploiting the benefits of messages exchange among agents. Simulation results demonstrate the superiority of the proposed algorithm over the state-of-the-art DRL-based algorithms.

Index Terms—Multi-agent reinforcement learning, UAV, trajectory planning.

## I. INTRODUCTION

M OBILE edge computing (MEC) is a key technique ofimproving the quality of experience (QoE) of mobile improving the quality of experience (QoE) of mobile users by offloading the computation tasks from users [1], [2]. Typically, MEC servers are deployed at fixed locations near the wireless edge, limiting their capability of providing flexible services.

Given their maneuverability, high-end unmanned aerial vehicles (UAVs), having unexploited computing and storage hardware, might be harnessed as the flexible servers for mobile users [3], [4], [5]. Jeong et al. [6] optimized both the resource allocation and the UAV’s trajectory for minimizing the overall energy consumption by utilizing a successive convex approximation-based algorithm. To tackle the resource allocation problem, Lyu et al. [7] propose a quantized dynamic programming algorithm for offloading delay-sensitive tasks in MEC. To reduce the complexity, Wu and Zhang [8] designed a UAV trajectory discretization method, and formulated a tractable problem for optimizing the consecutive UAV locations. However, due to the limited coverage range of UAVs and non-existence of a centralized control node, the UAV-aided MEC system environment is only partially observable for each UAV. Moreover, neither the model nor the dynamics of the environment are known a priori. Thus, it is difficult to formulate tractable problems for complex environments. Additionally, the computational complexity usually grows exponentially with the number of time slots considered in MEC settings, especially in multi-UAV scenarios.

To tackle the challenges of partial observations, unknown model and computational complexity, some authors harnessed reinforcement learning (RL) for UAV-aided MEC [9], [10], [11], [12], [13]. In [9], Zhang et al. proposed to minimize the energy consumption and computation latency by optimizing both the UAV trajectory control and task scheduling policy, by applying deep Q-network (DQN) methods for single-UAV scenarios. Wang et al. [10] optimized the user association, resource allocation and trajectory of UAVs for minimizing the energy consumption of all UEs. In order to maximize the energy efficiency while ensuring wide coverage by the UAVs, Liu et al. [11] proposed DRL-based methods for controlling the UAVs’ trajectory. However, both [10] and [11] rely on a centralized controller for information processing and decisions for all UAVs, which would have an excessive overhead.

In order to obtain a decentralized policy, Khoramnejad et al. [12] and Yu et al. [2] employed a distributed learner for each MEC agent, which however may result in a non-stationary problem [12], [14]. To tackle this problem, Wang et al. in [13] resorted to a centralized training and decentralized execution (CTDE) framework to control a UAV’s trajectory, aiming at maximizing the offloading fairness, while minimizing the overall energy consumption. Nonetheless, in both independent learner and CTDE methods, each agent chooses its action independently, unaware of the status and intentions of others, which hinders the cooperation among agents.

Against the above background and inspired by the observation that leveraging communications between the agents can provide critical information for decision making [14], [15], we conceive a novel communication assisted learning framework to deal with the aforementioned challenges in UAV trajectory control supporting MEC networks. Our major contributions can be summarized as follows:

• We formulate a new multi-agent RL (MARL) problem to optimize the trajectory of each UAV, aiming for maximizing both the number of offloaded tasks and the fairness.

• We propose a communication-assisted value decomposition network (CAVDN). By allowing agents to exchange the messages learned, each agent can take into account both its local observations and messages from others.

Our simulation results show the superiority of CAVDN over state-of-the-art DRL-based UAV control algorithms.

## II. SYSTEM MODEL

In this section, we describe the UAV-aided MEC system. Let $\mathcal { K } = \{ k | k = 1 , 2 , \ldots , K \}$ denote the IoT user equip-= = 1 2ment (UE) set. Due to the limited computing capability of IoT devices, their tasks have to be offloaded to UAVs for execution, where N rotary-wing UAVs with on-board MEC servers fly over the target area to provide computation services. Let $\bar { \mathcal { N } } \triangleq \{ n | n = 1 , 2 , \dots , N \}$ represent the UAV set. We assume = 1 2that all UAVs can be charged by the stations positioned on the roof tops before ruining out of battery as in [13] or lasercharged as in [16]. Thus, the energy consumption of UAVs is not considered.

The system is operated in discrete time steps (TSs), each with a duration of τ . In TS t, the coordinates of the nth UAV and the kth user are denoted by ${ { d } _ { n , t } ^ { u a v } } = [ { { d } _ { n , t } ^ { x } } , { { d } _ { n , t } ^ { y } } , { { d } _ { n , t } ^ { z } } ]$ and $d _ { k , t } ^ { u e } = [ d _ { k , t } ^ { x } , d _ { k , t } ^ { y } , 0 ]$ , respectively.

## A. UAV Motion and Offloading Decision

Let us denote the heading and normalized speed of the nth UAV at TS t by $\phi _ { n , t } \in [ 0 , 2 \pi ]$ , and $v _ { n , t } \in [ 0 , 1 ]$ , respectively. [0 2 ] [0 1]Consequently, the position of UAV n at TS t 1 is

$$
{ { d } _ { n + 1 , t } ^ { u a v } } = { { d } _ { n , t } ^ { u a v } } + ( { { v } _ { n , t } } { { V } _ { \operatorname* { m a x } } } + { { w } _ { n , t } } ) \tau ,\tag{1}
$$

where $\pmb { v } _ { n , t } = [ v _ { n , t } \cos ( \phi _ { n , t } ) , v _ { n , t } \sin ( \phi _ { n , t } ) , 0 ] , \ V _ { \operatorname* { m a x } }$ is the = [ cos( )maximal speed of the UAVs, and $w _ { n , t }$ ) 0]is a random variable reflecting the uncertainty of the environment, e.g., wind.

We consider the “full buffer” scenario, where each UE always has local tasks for offloading. If there is at least one UAV in the coverage of the UE, the UE will forward its local task to the nearest UAV. Otherwise, the UE will execute its task locally. Moreover, we assume that the computational capability of the UAV is powerful enough to complete each offloaded task in a single TS. The offloading decision of the kth UE for the nth UAV at TS t is denoted as $x _ { k , n , t }$ . Specifically, $x _ { k , n , t } = 1$ represents that UE k offloads its task to UAV n at = 1TS t, where $\textstyle \sum _ { n = 1 } ^ { N } x _ { k , n , t } \leq 1$ , and $x _ { k , n , t } = 0$ otherwise.

## B. Communication Channel and Task Offloading Process

Given the high probability of line-of-sight (LoS) connections between the UEs and UAVs, we only consider LOS channels. The channel gain between UE m and UAV n in TS t is $h _ { k , n , t } = d _ { k , n , t } ^ { - \alpha } \mu .$ where α is the path loss exponent, and =μ denotes the reference channel’s power gain at one meter.

We assume that all UEs share the same bandwidth W to communicate with the UAVs, and the ground-to-air links are scheduled using TDMA. Then, the offloading data rate from the kth UE to the nth UAV at TS t can be written as

$$
R _ { k , n , t } = \frac { W } { \sum _ { k = 1 } ^ { K } \sum _ { n = 1 } ^ { N } x _ { k , n , t } } \log _ { 2 } \biggl ( 1 + \frac { P h _ { k , n , t } } { \sigma ^ { 2 } } \biggr ) ,\tag{2}
$$

where P is the transmit power of the UE, and $\sigma ^ { 2 }$ is the additive white Gaussian noise power.

If UE k offloads its local task to UAV n at TS t, the duration of offloading the data is $\Delta _ { k , n , t } = S ( k ) / R _ { k , n , t } .$ , where S(k) is Δ = ( )the task size of UE k. If the offloading requires more than one time slot, i.e., $\Delta _ { k , n , t } \geq \tau .$ , the UAV will refrain from maneu-Δvering in the next TS for completing the offloading process. Let $\widetilde { x } _ { k , n , t }$ represent the task offloading status. Specifically, if the task sent from UE k to UAV n is finished at TS t, we have $\widetilde { x } _ { k , n , t } = 1$ . Otherwise, $\begin{array} { r } { \widetilde { x } _ { k , n , t } = 0 . } \end{array}$

## C. Performance Metric and Problem Formulation

Our primary objective is to maximize the average ratio of completed offloading tasks, where the ratio is expressed as

$$
e _ { t } = \sum _ { k = 1 } ^ { K } \sum _ { n = 1 } ^ { N } \frac { \widetilde { x } _ { k , n , t } } { K } .\tag{3}
$$

To balance the load among UAVs, we consider the fairness among each UAV’s load as another objective to guide the optimization. We define $\begin{array} { r } { y _ { n , t } = \sum _ { k = 1 } ^ { K } x _ { k , n , t } / K } \end{array}$ as the rela-=tive load of UAV n at TS t. Then, by applying Jain’s fairness index, the UAV’s fairness index $f _ { t } ^ { u a \bar { v } }$ can be expressed as

$$
f _ { t } ^ { u a v } = \frac { ( \sum _ { n = 1 } ^ { N } \sum _ { t ^ { \prime } = 1 } ^ { t } y _ { n , t ^ { \prime } } ) ^ { 2 } } { K \sum _ { n = 1 } ^ { N } ( \sum _ { t ^ { \prime } = 1 } ^ { t } y _ { n , t ^ { \prime } } ) ^ { 2 } } ,\tag{4}
$$

which approaches 1 when the number of served tasks for all UAVs at each TS is similar.

Meanwhile, we also consider the fairness between UEs, which is defined similarly by

$$
f _ { t } ^ { u e } = \frac { ( \sum _ { k = 1 } ^ { K } \sum _ { t ^ { \prime } = 1 } ^ { t } \sum _ { n = 1 } ^ { N } x _ { k , n , t ^ { \prime } } ) ^ { 2 } } { K \sum _ { k = 1 } ^ { K } ( \sum _ { t ^ { \prime } = 1 } ^ { t } \sum _ { n = 1 } ^ { N } x _ { k , n , t ^ { \prime } } ) ^ { 2 } } ,\tag{5}
$$

which approaches 1 if the number of offloaded tasks for all UEs at each TS is similar.

In contrast to metrics like data rate, the offloading fairness indices have no explicit threshold regarding their impact on the QoE of users. Thus, we include them into the objective function as in [13]. By jointly considering the number of offloaded tasks and the fairness indices, we formulate the trajectory control problem as follows

$$
\begin{array} { r l } { \underset { \phi _ { n , t } , v _ { n , t } } { \operatorname* { m i n } } } & { \mathbb { E } \left[ - \displaystyle \sum _ { t = 1 } ^ { T } ( e _ { t } + \lambda f _ { t } ^ { u a v } + \beta f _ { t } ^ { u e } ) \right] } \\ { s . t . } & { \phi _ { n , t } \in [ 0 , 2 \pi ] , v _ { n , t } \in [ 0 , 1 ] , } \end{array}\tag{6}
$$

where λ and $\beta$ are hyper-parameters used for adjusting the relative importance of the terms in the optimization objective function of (6). The expectation is taken over all random variables, including the UAV locations and fading channels in each TS.

At the initial TS, both the position of UEs and the randomness factor of UAV mobility are all unknown. Additionally, the objective function involving the derivation of $\widetilde { x } _ { k , n , t }$ is not tractable. Therefore, traditional optimization methods are unsuitable, and we apply MARL to solve the problem in the following.

## III. COMMUNICATION-ASSISTED VALUE DECOMPOSITION NETWORK ALGORITHM

In this section, we first introduce some basic notions of MARL. Then, we reformulate Problem (6) in the form of MARL by designing the observation, action and reward.

Finally, we introduce the communication-assisted MARL framework and its solution.

## A. MARL Problem Formulation

In MARL settings, usually the multi-agent decentralized partially observable Markov decision process (Dec-POMDP) is employed [17], [18]. Specifically, a Dec-POMDP can be represented by the state space S, the action space A, and the observation space O for each agent [14]. At the beginning of each TS t, the nth agent receives its local observation $o _ { n , t } \in \mathcal { O }$ , which is a part of the state $\mathbf { } s _ { t } .$ . Then, it chooses an action $\mathbf { } a _ { n , t } \in { \mathcal { A } } _ { \cdot }$ , according to its local observation $\scriptstyle o _ { n , t }$ based on policy $\pi _ { n } .$ . After all agents’ actions are executed, agent n obtains reward $r _ { n , t } ,$ which depends on the actions of all agents $\{ a _ { n , t } | 1 \le n \le N \}$ . At the end of TS t, the current state $\mathbf { \boldsymbol { s } } _ { t }$ 1evolves into the next state $\mathbf { } _ { s _ { t + 1 } }$ . For cooperative tasks, all agents have the same objective of maximizing the team reward $\begin{array} { r } { \mathbb { E } [ \sum _ { t = 1 } ^ { T } \sum _ { n = 1 } ^ { M } \gamma ^ { t - 1 } \bar { r } _ { n , t } ] } \end{array}$ , where $\gamma$ is the discount factor.

To employ RL methods for solving Problem (6), we firstly define the local observation, as well as the action and reward function for each agent at TS t as follows.

1) Observation $\mathbf { \alpha } _ { o n , t } \mathbf { \dot { \alpha } } $ Since both the offloading decision and offloading data rate depend on the distance between UAVs and users, we include their coordinates into the observation vector. Additionally, we include the accumulated number of tasks offloaded for UEs and the accumulated load of UAVs, which determine the fairness among UEs and among UAVs, respectively. Since the UAVs have limited reception range, we only take the information of the nearest $M _ { \mathrm { d } } ^ { u e }$ users and the nearest $M _ { \mathrm { d } } ^ { u a v }$ UAVs into account. Then, the observation vector of UAV n can be formulated as

$$
\begin{array} { l } { { \displaystyle { \pmb o } _ { n , t } = \left[ d _ { 1 , t } ^ { u e } , \dots , d _ { M _ { d } ^ { u e } , t } ^ { u e } , \sum _ { t ^ { \prime } = 1 } ^ { t } \sum _ { n = 1 } ^ { N } x _ { 1 , n , t ^ { \prime } } , \dots , \sum _ { t ^ { \prime } = 1 } ^ { t } \right. } } \\ { { \displaystyle \left. \sum _ { n = 1 } ^ { N } x _ { M _ { d } ^ { u e } , n , t ^ { \prime } } , d _ { 1 , t } ^ { u a v } , \dots , d _ { M _ { d } ^ { u a v } , t } ^ { u a v } , y _ { 1 , t } , \dots , y _ { M _ { d } ^ { u a v } , t } \right] } . } \end{array}\tag{7}
$$

2) Action $\mathbf { } a _ { n , t } \mathrm { : }$ The action of each UAV agent includes its heading and speed, $\mathrm { i . e . , } a _ { n , t } = [ \phi _ { n , t } , v _ { n , t } ]$

3) Reward $\boldsymbol { r } _ { n , t } .$ = [ ]Since we consider cooperative tasks, the team reward

$$
r _ { n , t } = e _ { t } + \lambda f _ { t } ^ { u a v } + \beta f _ { t } ^ { u e } - \sum _ { n = 1 } ^ { N } \frac { c _ { n } ^ { \mathrm { c o l } } } { N } ,\tag{8}
$$

is shared among all agents, where $c _ { n } ^ { \mathrm { c o l } }$ is a penalty term introduced for avoiding collisions between UAVs. Specifically, $c _ { n } ^ { \mathrm { c o l } } = 1$ when UAV n collides with others, otherwise $c _ { n } ^ { \mathrm { c o l } } = \dot { 0 }$ = 1 = 0Then, the goal of each agent is to cooperatively minimize the loss function expressed by

$$
\operatorname* { m i n } _ { \pi _ { n } , 1 \le n \le N } J = \mathbb { E } \left[ - \frac { 1 } { N } \sum _ { t = 1 } ^ { T } \sum _ { n = 1 } ^ { N } \gamma ^ { t - 1 } r _ { n , t } \right] .\tag{9}
$$

## B. Communication Assisted Decentralized Reinforcement Learning Framework

Existing RL-based papers on wireless MEC networks tend to implicitly assume conditional independence of actions from different agents [12], [13]. Consequently, each agent chooses its action purely based on its local observation during execution, while neglecting any interactions among agents. By contrast, we allow each agent to exchange its local information with the agents within its communication range during both training and execution, and more importantly, to learn what information should be exchanged for promoting cooperation among agents for better informed decision making.

<!-- image-->  
Fig. 1. CAVB framework and agent network structure.

Specifically, we introduce the communication-assisted value-based (CAVB) MARL framework of Fig. 1(a). Each agent stores and trains an agent network locally to estimate the local state-action value function of $Q _ { n } ( \pmb { o } _ { n } , \pmb { a } _ { n } ) \triangleq$ E $\begin{array} { r } { \texttt { \ i } [ \sum _ { i = t } ^ { T } \gamma ^ { i - t } r _ { n , t } | \pmb { o } _ { n , t } \ = \ \pmb { o } _ { n } , \pmb { a } _ { n , t } \ = \ \pmb { a } _ { n } ] } \end{array}$ . Based on this, [ = = ]the optimal action of agent n can be determined as $\begin{array} { r l } { \mathbf { { \boldsymbol { a } } } _ { n } } & { { } = } \end{array}$ arg $\begin{array} { r } { \operatorname* { m a x } _ { \pmb { a } _ { n } ^ { \prime } } Q _ { n } \mathopen { } \mathclose \bgroup \left( \pmb { o } _ { n } , \pmb { a } _ { n } ^ { \prime } \aftergroup \egroup \right) } \end{array}$ =. In contrast to the existing MARL arg max ( )frameworks, where each agent only estimates the local stateaction value based on its own local observation, we allow each agent to transfer messages based on its local observation between nearby agents. In this way, the observation gleaned from nearby agents can also be taken into account at a moderate overhead, when estimating the local state-action value. As a benefit, the local state-action can be estimated more accurately. Then, a mixing network combines the output of all agents’ networks for approximating the global state-action value function of $\begin{array} { r l } { Q _ { t o t } ( o , \pmb { a } ) } & { { } \triangleq \mathbb { E } [ \sum _ { i = t } ^ { T } \gamma ^ { i - t } r _ { n , t } | \pmb { o } _ { t } } \end{array}$ 二 $\mathbf { \nabla } _ { \mathbf { 0 } } , \mathbf { \vec { a } } _ { t } \ = \ \mathbf { \vec { a } } ] .$ ( ), where we have $\textbf { \em o } = \{ \pmb { o } _ { 1 } , \pmb { o } _ { 2 } , \dots , \pmb { o } _ { N } \}$ =, and $\textbf { \em a } = \{ \boldsymbol { a } _ { 1 } , \boldsymbol { a } _ { 2 } , \ldots , \boldsymbol { a } _ { N } \}$ =. For instance, $Q _ { t o t } ( \pmb { o } , \pmb { a } )$ can be = ( )the summation of all the individual action value functions in the value decomposition network (VDN) [18], yielding $\begin{array} { r } { Q _ { t o t } ( o , \pmb { a } ) \ = \sum _ { n = 1 } ^ { N } Q _ { n } ( \pmb { o } _ { n } , \pmb { a } _ { n } ) } \end{array}$

## C. Agent Network Structure and Training Algorithm

In the following, we first design the agent network and then formulate the above-mentioned algorithm for training the agent network.

Each agent network contains three modules: encoding module, intention module, and combining module, as shown in Fig. 1(b). To mitigate the model’s complexity and accelerate the training process, the same values of θ are shared among all agents’ network. Therefore, in the following, we consider agent n as an example for characterizing the agent network.

(1) The encoding module takes the local observation $\scriptstyle { o _ { n } }$ as its input and outputs the encoded message $m _ { n } .$ It consists of a recurrent neural network (RNN) followed by a fully-connected neural network (FNN). This recurrent structure can integrate the previous observations into the encoded messages that are transferred between agents.

<!-- image-->  
Fig. 2. The training and execution process of the nth UAV agent.

(2) The intention module also takes the observation $\scriptstyle { o _ { n } }$ as its input, and outputs the intention $h _ { n }$ that indicates the agent’s behavior without considering others. It shares the same RNN with the encoding module. The motivation for introducing the intention module is to accelerate the learning process. At the beginning of the training phase, the messages from others may be noisy due to the randomly initialized neural networks, and hence they may negatively affect the decision making. The intention is used for compensating such an effect and helps learn local policies more quickly.

(3) The combining module is a FNN, which takes both the local intention $h _ { n }$ and the messages arriving from the nearest $M _ { \mathrm { d } } ^ { u a v }$ agents (denoted as $\mathcal { M } _ { n }$ , including local message $m _ { n } )$ , and outputs the estimated local action-value function.

The training and execution process is shown in Fig. 2. The training process includes sample collection and agent network updates. During the sampling process, we harness the -greedy policy to determine each agent’s action. In particular, the agent chooses $\begin{array} { r } { \mathbf { a } _ { n } = \arg \operatorname* { m a x } _ { \pmb { a } _ { n } ^ { \prime } } Q _ { n } \big ( \pmb { o } _ { n } , \pmb { a } _ { n } ^ { \prime } \big ) } \end{array}$ with a probability of $1 - \epsilon ,$ = arg max ( ) and opts for a random action with prob-1ability . The interactions of all agents with the environment, including the messages received by each agent, are collected and stored in an experience replay buffer D. Each sample is denoted by a five-tuple $\left( \pmb { o } _ { t } , \ \mathcal { M } _ { t } , \ \pmb { a } _ { t } , \ r _ { t } , \ \pmb { o } _ { t + 1 } \right)$ , where $\mathcal { M } _ { t } = \{ \mathcal { M } _ { 1 , t } , \dotsc , \mathcal { M } _ { N , t } \}$ ( )represents the received messages =of all agents.

The agent networks are trained as shown in Fig. 2 via updating the parameters θ by minimizing the following loss function via gradient descent,

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { ( o , \mathcal { M } , \pmb { a } , r , \pmb { \sigma } ^ { \prime } ) \sim \mathcal { D } } [ ( Q _ { t o t } ( \pmb { o } , \pmb { a } ; \pmb { \theta } ) - y ) ^ { 2 } ] ,\tag{10}
$$

where $\begin{array} { r } { y = r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { t o t } ^ { \prime } \left( o ^ { \prime } , a ^ { \prime } ; \widehat { \pmb { \theta } } \right) } \end{array}$ and $\widehat { \pmb { \theta } }$ is the param-= + max ( ; )eter of the target network, which has the same architecture as the agent network, but have different parameter values. It is updated via $\widehat { \pmb { \theta } }  \tau \pmb { \theta } + ( 1 - \tau ) \widehat { \pmb { \theta } }$ using a small value + (1 )of τ . This soft update helps reduce the correlations between $Q _ { t o t } \big ( o , a ; \pmb \theta \big )$ and the target value $y ,$ hence stabilizing learning. ( ; )The detailed procedures are provided in Algorithm 1.

## IV. SIMULATION RESULTS

In this section, we compare the performance of our CAVDN algorithm to that of several baseline methods via simulation. Specifically, the following state-of-the-art DRL-based trajectory plan methods are compared: (1) VDN of [18] (2) multiagent deep deterministic policy gradient (MADDPG) of [17]; (3) Double deep Q-network (DQN) of [19]; (4) CommNet of [15], which directly processes the received messages using the arithmetic mean; (5) Targeted Multi-Agent Communication (TarMAC) of [20], which processes the received messages using the weighted mean via multi-head attention. For methods that cannot be directly applied to continuous action spaces, we discretize the action space, where the heading set is $\{ 0 , \pi / 4 , \ldots , 7 \pi / 4 \}$ and the speed set is {0, 0.5, 1}, result-0 4 7 4ing in an action space of size 24. Moreover, we also compare our CAVDN algorithm to the random and greedy UAV trajectory control methods, where each UAV moves randomly or towards the nearest UE sequentially.

```latex
Algorithm 1 The CAVDN Algorithm
1: Randomly initialize θ and $\widehat { \pmb { \theta } }$ following Gaussian distribution. Set
${ \mathcal { D } } = \emptyset .$
2: for episode = 1 to max-episode-number do
3: Reset the environment, and obtain the initial local observation
$\mathbf { \nabla } _ { o n }$ for $n = 1$ to N.
4: for t = 1 to T do
5: Get message $m _ { n , t } = e ( o _ { n , t } ; \theta _ { e } )$ for agent $n = 1$ to N.
6: Select action $\mathbf { } _ { \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Sigma } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \Sigma \mathbf { } \mathbf \mathbf { } \mathbf \Sigma \mathbf { } \mathbf \mathbf { } \mathbf \Sigma \mathbf { } \mathbf \Sigma \mathbf { } \mathbf \Sigma \mathbf \Sigma \mathbf { } \mathbf \Sigma \mathbf \Sigma \mathbf { } \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \mathbf \Sigma \mathbf \Sigma \Sigma $ from based on $^ { o } n , t$ and $\mathcal { M } _ { n , t }$ using
-greedy $Q _ { n } ( \boldsymbol { o } _ { n , t } , \boldsymbol { a } _ { n , t } ; \pmb { \theta } )$ for agent ${ \bf \chi } _ { n } = 1 { \bf \chi } _ { \mathrm { t o } } N .$
7: Execute actions $\{ { \pmb a } _ { 1 } , { \dot { \pmb a } } _ { 2 } , . . . , { \pmb a } _ { N } \} .$ . Then, obtain reward r
and next state observations ${ \pmb o } _ { n } ^ { \prime }$ for each agent n.
8: Push $( o , \pmb { a } , \mathcal { M } , r , o ^ { \prime } )$ into replay buffer D and set $_ { o n } \gets$
$\pmb { o } _ { n } ^ { \prime }$ for $n = 1$ to N.
9: if length of D larger than given length then
10: Randomly generate index set $\boldsymbol { B } \stackrel { \smile } { = } \{ b _ { 1 } , . . . , b _ { B } \}$
11: Obtain the sample batch $\mathrm { ~ \bf ~ o f ~ } \mathrm { ~ \bf ~ B ~ } ^ { - }$ samples
$\{ ( o ^ { i } , a ^ { i } , \mathcal { M } ^ { i } , r ^ { i } , o ^ { \prime { \dot { \iota } } } ) \} _ { i \in \mathcal { B } }$ from D.
12: Compute the critic loss as L(θ)
$\begin{array} { r } { \frac { 1 } { i } \sum _ { i } ( Q _ { t o t } ( o ^ { i } , { \mathbf { a } } ^ { i } ) \mathrm { ~  ~ \omega ~ } - \mathrm { ~  ~ \omega ~ } y ^ { i } ) ^ { 2 } } \end{array}$ , where - $y ^ { \ i } \quad =$
$\begin{array} { r } { r ^ { i } + \gamma \operatorname* { m a x } _ { a } Q _ { t o t } ( \pmb { o } ^ { \prime } { } ^ { i } , a ; \widehat { \pmb { \theta } } ) . } \end{array}$
13: Update θ by minimizing $\mathcal { L } ( \pmb { \theta } ) _ { }$
14: Update target network parameters by ${ \widehat { \pmb \theta } }  \tau { \pmb \theta } + ( 1 - \tau ) { \widehat { \pmb \theta } } .$
15: end if
16: end for
17: end for
```

We consider a $2 0 0 \times 2 0 0 ~ \mathrm { m } ^ { 2 }$ square area containing $M =$ 12 randomly distributed UEs and $\bar { N } = 4 ~ \mathrm { U A V s }$ =at the altitude =of 100 m. Only the users with a “horizon distance” smaller than $D _ { \mathrm { t h } } = 2 0$ m can be discovered by and connected to the = 20UAV, and we set $M _ { \mathrm { d } } ^ { u a v } = 3$ and $M _ { \mathrm { d } } ^ { u \bar { e } } = 6$ . For the channel model, we set $\alpha = 2 , \mu = - 3$ = 6 dB as in [21]. The noise power is $\sigma ^ { 2 } = - 1 0 0$ 2 = dBm, and the transmit power is $P = 5 0 0$ =mW. The maximal speed is $V _ { \mathrm { m a x } } = 4 0$ = m/s as in [22]. Detailed =settings regarding the neural networks and the training process can be found at our github repository.1

In Fig. 3, we compare the average reward versus the number of episodes in the training phase for different methods. It can be observed that VDN and MADDPG achieve better performance than Double DQN, because they maintaining a global value-function or global critic network during training. Among all methods, CAVDN achieves the highest average reward, due to the extra information received via communication during both training and testing phases, which provides each agent with extra information about the others for promoting cooperation. The performance of CommNet is inferior to that of CAVDN and that of TarMAC. This is because CommNet simply uses the arithmetic mean for processing the messages received, which results in information loss.

<!-- image-->  
Fig. 3. Learning curves of CAVDN and other baselines. The reward is averaged over all UAVs and timesteps in each episode.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4. Comparison of offloading fairness and the number of offloaded tasks.

Moreover, our proposed CAVDN performs slightly better than TarMAC because we also use RNN for encoding the messages. Besides, compared to CAVDN_no_int (CAVDN without intention module), the reward of CAVDN increases more rapidly and converges earlier, which validates the benefits of the intention module.

The UE fairness, UAV fairness, and the number of offloaded tasks, are further investigated in Fig. 4 after the convergence of the RL-based methods. During testing, the positions of UEs are randomly generated and they are different from that used in the training phase. We can see that both the UE fairness and UAV fairness of CAVDN are the highest compared to others methods. Moreover, all learning-based methods achieve similar performance in terms of the number of offloaded tasks. It is worth noting that for the Double DQN and greedy methods, both the UE fairness and UAV fairness first increase and then decrease. This is because each UAV only has partial observations, hence some UAVs may have excessive load in the absence of efficient communication between agents, which prevents load-balancing.

## V. SUMMARY AND CONCLUSION

A novel MARL based UAV trajectory planning scheme was proposed for UAV-aided MEC networks. The objective is to maximize the number of offloaded tasks and offloading fairness by optimizing the trajectory control policies of UAVs. To intelligently exchange critical information among agents to assist decentralized decision making, we conceived a communication-assisted MARL framework and proposed the CAVDN algorithm for training each UAV. Our simulation results showed that our proposed CAVDN algorithm improves the offloading fairness among users and balances the load among UAVs, compared to both the state-of-theart DRL-based and classic greedy methods. This indicates that the messages learned and exchanged via communication provide valuable knowledge for each agent to “understand” the situation and the intention of others, which is critical for multi-agent cooperation in sophisticated trajectory control policies. In the future, we will investigate the impact of the cost introduced by message exchange.

## REFERENCES

[1] C. Park and J. Lee, “Mobile edge computing-enabled heterogeneous networks,” IEEE Trans. Wirel. Commun., vol. 20, no. 2, pp. 1038–1051, Feb. 2021.

[2] J. Yu et al., “IRS assisted NOMA aided mobile edge computing with queue stability: Heterogeneous multi-agent reinforcement learning,” IEEE Trans. on Wirel. Commun., vol. 22, no. 7, pp. 4296–4312, Jul. 2022.

[3] N. H. Motlagh, M. Bagaa, and T. Taleb, “UAV-based IoT platform: A crowd surveillance use case,” IEEE Commun. Mag., vol. 55, no. 2, pp. 128–134, Feb. 2017.

[4] Z. Yang, C. Pan, K. Wang, and M. Shikh-Bahaei, “Energy efficient resource allocation in UAV-enabled mobile edge computing networks,” IEEE Trans. Wirel. Commun., vol. 18, no. 9, pp. 4576–4589, Sep. 2019.

[5] M. Li, N. Cheng, J. Gao, Y. Wang, L. Zhao, and X. Shen, “Energyefficient UAV-assisted mobile edge computing: Resource allocation and trajectory optimization,” IEEE Trans. Veh. Technol., vol. 69, no. 3, pp. 3424–3438, Mar. 2020.

[6] S. Jeong, O. Simeone, and J. Kang, “Mobile edge computing via a UAV-mounted cloudlet: Optimization of bit allocation and path planning,” IEEE Trans. Veh. Technol., vol. 67, no. 3, pp. 2049–2063, Mar. 2018.

[7] X. Lyu, H. Tian, W. Ni, Y. Zhang, P. Zhang, and R. P. Liu, “Energyefficient admission of delay-sensitive tasks for mobile edge computing,” IEEE Trans. Commun., vol. 66, no. 6, pp. 2603–2616, Jun. 2018.

[8] Q. Wu and R. Zhang, “Common throughput maximization in UAVenabled OFDMA systems with delay consideration,” IEEE Trans. Commun., vol. 66, no. 12, pp. 6614–6627, Dec. 2018.

[9] L. Zhang et al., “Task offloading and trajectory control for UAV-assisted mobile edge computing using deep reinforcement learning,” IEEE Access, vol. 9, pp. 53708–53719, 2021.

[10] L. Wang, K. Wang, C. Pan, W. Xu, N. Aslam, and A. Nallanathan, “Deep reinforcement learning based dynamic trajectory control for UAVassisted mobile edge computing,” IEEE Trans. Mobile Comput., vol. 21, no. 10, pp. 3536–3550, Oct. 2022.

[11] C. H. Liu, Z. Chen, J. Tang, J. Xu, and C. Piao, “Energy-efficient UAV control for effective and fair communication coverage: A deep reinforcement learning approach,” IEEE J. Sel. Areas Commun., vol. 36, no. 9, pp. 2059–2070, Sep. 2018.

[12] F. Khoramnejad and M. Erol-Kantarci, “On joint offloading and resource allocation: A double deep Q-network approach,” IEEE Trans. Cogn. Commun. Netw., vol. 7, no. 4, pp. 1126–1141, Dec. 2021.

[13] L. Wang, K. Wang, C. Pan, W. Xu, N. Aslam, and L. Hanzo, “Multiagent deep reinforcement learning-based trajectory planning for multi-UAV assisted mobile edge computing,” IEEE Trans. Cogn. Commun. and Netw., vol. 7, no. 1, pp. 73–84, Mar. 2021.

[14] P. Hernandez-Leal, B. Kartal, and M. E. Taylor, “A survey and critique of multiagent deep reinforcement learning,” Auton. Agents Multi-Agent Syst., vol. 33, pp. 750–797, Oct. 2019.

[15] S. Sukhbaatar, A. Szlam, and R. Fergus, “Learning multiagent communication with backpropagation,” in Proc. NeurIPS, 2016, pp. 2244–2252.

[16] Q. Liu et al., “Charging unplugged: Will distributed laser charging for mobile wireless power transfer work,” IEEE Veh. Technol. Mag., vol. 11, no. 4, pp. 36–45, Dec. 2016.

[17] L. Ryan et al., “Multi-agent actor-critic for mixed cooperativecompetitive environments,” in Proc. NeurIPS, 2017, pp. 6379–6390.

[18] P. Sunehag et al., “Value-decomposition networks for cooperative multiagent learning based on team reward,” in Proc. AAMAS 2018, Stockholm, Sweden, July 10-15, 2018, 2018, pp. 2085–2087.

[19] J. Hu et al., “Cooperative Internet of UAVs: Distributed trajectory design by multi-agent deep reinforcement learning,” IEEE Trans. on Commun., vol. 68, no. 11, pp. 6807–6821, Nov. 2020.

[20] A. Das et al., “TarMAC: Targeted multi-agent communication,” in Proc. ICML, 2019, pp. 1–10.

[21] Y. Wang, Z. Hu, X. Wen, Z. Lu, J. Miao, and H. Qi, “Three-dimensional aerial cell partitioning based on optimal transport theory,” in Proc. IEEE ICC, 2020, pp. 1–6.

[22] Y. Zeng, J. Xu, and R. Zhang, “Energy minimization for wireless communication with rotary-wing UAV,” IEEE Trans. on Wirel. Commun., vol. 18, no. 4, pp. 2329–2345, Apr. 2019.