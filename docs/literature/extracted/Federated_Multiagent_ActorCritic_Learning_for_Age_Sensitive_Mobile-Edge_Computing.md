# Federated Multiagent Actor–Critic Learning for Age Sensitive Mobile-Edge Computing

Zheqi Zhu , Shuo Wan , Pingyi Fan , Senior Member, IEEE, and Khaled B. Letaief , Fellow, IEEE

Abstract—As an emerging technique, mobile-edge computing (MEC) introduces a new scheme for various distributed communication-computing systems, such as industrial Internet of Things (IoT), vehicular communication, smart city, etc. In this work, we mainly focus on the timeliness of the MEC systems where the freshness of the data and computation tasks is significant. First, we formulate a kind of age-sensitive MEC models and define the average Age-of-Information (AoI) minimization problems of interests. Then, a novel mixed-policy-based multimodal deep reinforcement learning (RL) framework, called heterogeneous multiagent actor–critic (H-MAAC), is proposed as a paradigm for joint collaboration in the investigated MEC systems, where edge devices and center controller learn the interactive strategies through their own observations. To improve the system performance, we develop the corresponding online algorithm by introducing the edge federated learning mode into the multiagent cooperation whose advantages on learning convergence can be guaranteed theoretically. To the best of our knowledge, it is the first joint MEC collaboration algorithm that combines the edge federated mode with the multiagent actor– critic RL. Furthermore, we evaluate the proposed approach and compare it with popular RL-based methods. As a result, the proposed algorithm not only outperforms the baselines on average system age, but also promotes the stability of training process. Besides, the simulation outcomes provide several insights for collaboration designs over MEC systems.

Index Terms—Federated learning (FL), joint collaboration, mixed policies, mobile-edge computing (MEC), multiagent deep reinforcement learning (RL), multimodal learning.

## I. INTRODUCTION

## A. Backgrounds

N THE past decades, a large number of smart devices has I been massively deployed in numerous fields, such as industrial Internet of Things (IoT), nets of vehicles, environment monitoring networks. As an emerging paradigm for such distributed data systems, edge computing (EC), especially mobile EC (MEC), couples the communication with the computation and expands the cloud computing (CC) through allowing the computation to be executed at the edge nodes deployed along the path from data sources to the center cloud. In real applications, an MEC system is composed of three layers. The bottom level contains the sources of the data, such as users’ equipments (UEs), sensors or Web cameras that generate the data in order to provide certain services. At the middle layers, the edge devices can be smartphones, smart routers, intelligent base stations and vehicles with processors on board, which play a relay role. Edge devices can collect data from edge sensors, assist to execute some computation tasks and then relay them to cloud center [1]. At the top level, the data and the service requirements, refer to tasks, are assembled at the cloud center. The edge processing schemes need to be designed, so as to improve the system efficiency as well as flexibility [2]. That is, it can efficiently utilize the edge computation capabilities to reduce the redundant communication consumption of the networks.

Recently, benefiting from the development of 5G networks, more data can be transmitted with lower latency, which make it possible for MEC systems to be applied in several promising real-time applications. For instance, MEC can support the augment reality (AR), virtual reality (VR), and to enhance the user’s experience and promote the quality of the media streaming. In such cases, the images or videos are transmitted and processed in the network and the demands for the data rate and timeliness become stricter than the conventional distributed scenes [3]. Besides, in some urban security scenes, such as urgency monitoring, the delay of the data transmission and the processing directly impacts the quality of service [4]. To stress out such time-sensitive scenarios, Age of Information (AoI) [5] has been introduced to the related investigations as a metric of data freshness.

Generally, the computing resources and capacities of the mobile-edge devices are limited. Additionally, the arrival of data as well as the computation tasks are dynamic, and the communication rates between the entities are also constrained by restricted bandwidth. Thus, the scheduling of the MEC systems should be well-designed. To some degree, such a problem is usually cast into a joint stochastic optimization problem with the certain targets. In particular, considering the strategies for the entity operation in the MEC networks, the MEC collaboration shall be fully studied [6]. The collaborations are mainly considered from the following three aspects: 1) the resource management, including computation resources, power and bandwidth allocation, etc. [7], [8]; 2) the edge-level control, i.e., the data collection and trajectory planning for mobile-edge devices [9]; and 3) the data scheduling, including the task execution at the edge devices, the computation assignment for the edge network and the data offloading to the cloud center [10], [11]. The critical point of the joint collaboration is to achieve the global optimal strategies for MEC systems. Unfortunately, due to the dynamics of the MEC environment and the complex coupling of the entities from different layers, the related optimization problems are always nonconvex and NP-hard [12]. Therefore, the conventional vanilla optimization methods may not work well for the joint MEC collaboration problems and the iterated online approaches with better learning ability and intelligence shall be investigated.

## B. Motivations

Further, for future 6G networks, edge-native artificial intelligence (AI) is regarded to be a potential subject which leverages the MEC systems together with the distributed computing applications. The concepts of AI for EC and EC for AI are promising visions for future data systems [13]. In this article, we mainly focus on AI for EC and consider how to exploit AI or deep learning (DL) techniques to improve the performance of EC systems. Reinforcement learning (RL), especially deep RL (DRL) [14] has been incorporated into several decisionmaking scenarios, such as automatic robot control and game playing [15]. A widely used class of RL approaches are value-based RL, e.g., Q-learning and deep Q-network (DQN) which predict the system targets of each action and the make decisions according to an action-value function [16]. For the continuous control or the cases where the action space is extremely large, policy-based methods, such as deep deterministic policy gradient (DDPG ) are introduced [17]. The basic notion is that an actor module and a critic module are built to fit the best strategies and the evaluation values, respectively. Moreover, multiagent RL (MARL) [18] and federated learning (FL)-based RL [19] are also developed for the distributed scenes where agents learn to make decisions through their local observations and cooperate for the same system targets.

Motivated by these, we consider a MARL approach for joint MEC collaboration to maintain the data freshness. In particular, we investigate the cooperative learning framework with mixed policies where the agents learn multiple strategies through local observations and states. Besides, the communication mechanism among learning agents is also a critical point to be studied in this article.

## C. Related Works

In the literature, some related researches on the joint collaboration, the age-optimal optimization and the applications of DL in MEC systems has been broadly studied.

Ndikumana et al. [20] proposed a joint framework for communication, task computation, data caching and the distributed control in big data EC and evaluated several performance metrics for different procedures. To enhance the mobility and adaptability of the EC, MEC systems with unmanned aerial vehicle (UAV) assisted were investigated [21], [22]. The model formulations, including task arrival, computation, data scheduling and communication, were, respectively, studied in [23]–[26]. Zhou et al. [27] and Liu et al. [28] studied the energy efficient joint optimization of UAV-assisted system, and latency aware collaboration were also studied in [29]. In terms of the age sensitive MEC system, AoI-based metrics are introduced to measure the freshness of data [30]. Wang et al. [31] proposed the age of critical information in mobile computing and developed a partially observed scheduling approach. The AoI aware radio resource allocation of multivehicular communications was studied in [32]. Besides, Liu et al. [33] and Hu et al. [34] studied the age optimal joint collaboration in time sensitive MEC systems.

A number of researches exploited the multistage optimization method or iterative algorithms to solve the joint collaboration problems. However, these existing approaches suffered from the challenges that the real-world scenes are dynamics and complicated, which makes it difficult for these algorithms to extract the latent connections between the environment variation and the entity operation. Hence, some RL-based online algorithms are developed [35] for MEC data systems. The Q-learning and DQN-based RL methods were applied for resource allocation [36], task offloading [37], as well as trajectory planning [38]. Particularly, Chen et al. [39] addressed the limitation of Q-learning and proposed a double DQN-based offloading algorithm. Additionally, taking the multiple edge devices into consideration, some extended versions of MARL methods were proposed. A multiagent actor–critic-based offloading approach was designed in [40]. Peng and Shen [41] and Wang et al. [42] adopted multiagent DDPG (MADDPG) frameworks to resource management and trajectory planning in MEC networks. Besides, in [43], the vehicular layer MADDPG with attention mechanism was studied for multi-UAV assisted networks. Moreover, by employing FL into multiagent control [44], Wang et al. [45] combined the FL and DQN as a decentralized cooperative framework to improve the performance of edge caching.

## D. Contributions and Paper Organization

The main contributions of this work can be summarized as follows.

1) We put forward the system model and a multiagent Markov decision process (MDP) formulation to characterize the problems in age sensitive MEC for further investigations.

2) We build a simulation environment as a gym module1 for these MEC systems, which can be easily employed to test the performance of different collaboration approaches.

3) We present a multiagent DRL framework, H-MAAC, for MEC joint collaboration. It is a multimodal framework that takes heterogeneous inputs to learn the mixed policies for trajectory planning, data scheduling and resource allocation.

4) We develop the corresponding multiagent cooperation algorithm for the online joint collaboration by introducing the edge FL mode into the MEC collaboration, abbreviated as EdgeFed H-MAAC,1 which outperforms

<!-- image-->  
Fig. 1. 3-tier MEC system.

DDPG and MADDPG on both system metrics and convergence. To the best of our knowledge, it is the first joint MEC collaboration algorithm that combines the edge federated mode with the multiagent actor– critic RL.

5) The convergence analysis of the proposed MEC collaboration algorithm is also provided in theory which implies that the EdgeFed H-MAAC method reaps better convergence. Besides, the parameter design is also discussed.

The remaining of this article is organized as follows. In Section II we propose an age sensitive MEC system model and present the problems of interests in this work. In Section III, we first formulate the MDP for the age minimization problem, and then build up a multiagent edge federated actor–critic learning framework as well as develop the corresponding cooperation algorithm. The learning convergence of the proposed algorithm are presented as a theorem in Section IV. The simulation results and more discussions are presented in Section V. Finally, in Section VI, we conclude this work and give several potential research directions.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

In this section, a classic 3-tier multiagent EC system will be first introduced, including the corresponding communication and operation models. Then, the problems concerned in age sensitive scenarios will be defined.

As shown in Fig. 1, a classic 3-tier EC system with data collection is studied. The bottom level are data sources (S), such as sensors, Web cameras and UEs that continuously generate data packets. Then middle level consists of mobile-edge devices (E), such as automobile base stations and UAV base stations (UAV-BSs) which manage to move in the area, communicate with the data sources and the cloud center, as well as carry out some processing of the data. At the top level, the cloud center (C) is usually the data center with computing clusters that execute the computing tasks, store the data and implement centralized control for the whole system. The related models of data generation, edge mobility, edge operation, data scheduling, transmission and center controller will be introduced in detail. Additionally, some necessary notations and their explanations are listed in Table I. To explicitly investigate the operations in each time slot, we divide continuous time into small slots and use the integer t to denote the tth time slot.

## A. Data Generation Model

Consider the data sources generate the packets of same formatted data structures, for instance, the sensor data from distributed sensors in IoTs, the image or video data from Web cameras in urban security monitoring or VR/AR scenarios, and the data for a certain class of computing tasks from users with smart equipments. Assuming that the data sources generates independently, we describe the data packets using a tuple of data size, elapsed time and the index of their sources, denoted by $d _ { . } ( t ) , w _ { . } ( t ) , i d x _ { . } ( t )$ , respectively. The arrival packets are temporally stored in the source buffers and wait to be collected.

## B. Edge Mobility Model

In the MEC system introduced above, edge devices are considered to be vehicular base stations with mobility and computing capacity. All edge devices move in the area, collect the packets from data sources, process the data locally and offload the data to cloud center. In the following discussion, we assume that the edge devices E are a set of UAVs flying at certain heights over the data sources. Then, the mobility of edge devices can be modeled as

$$
\mathbf { p o s } _ { k } ( t + 1 ) = \mathbf { p o s } _ { k } ( t ) + \mathbf { m o v e } _ { k } ( t )\tag{1}
$$

where $\mathbf { p o s } _ { k } ( t )$ denotes the position of $E _ { k }$ at the beginning of tth time slot, and the movement movek(t) in each time slot is constrained by

$$
\| \mathbf { m o v e } _ { k } ( t ) \| _ { 2 } \leq r _ { \mathrm { m o v e } } ^ { k }\tag{2}
$$

where $\| \cdot \| _ { 2 }$ is the 2-norm of the vectors.

## C. Edge Processing and Data Scheduling Model

While edge agent $E _ { k }$ hovering over the data source $S _ { n } ,$ , all the packets in $S _ { n } \mathrm { \ ' } _ { \mathrm { s } }$ data buffer will be collected and take up one piece of the collected data buffer, $D _ { \mathrm { c o l } } ^ { k } ( t )$ . The data pieces in $D _ { \mathrm { c o l } } ^ { k } ( t )$ will be scheduled to be preprocessed using local processor on $E _ { k }$ . Since the data packets are assumed to be of formatted structure and the edge preprocess algorithms are determined, the computation on edge devices can be modelled by the data size and the EC rate [46]. Then, the accumulated execution duration of packets collected from $S _ { n }$ with $E _ { k }$ executing edge processing at the time slot t can be obtained by the equation

$$
\tau _ { n } ^ { k } ( t ) = \frac { \sum _ { i } \mathbb { 1 } _ { \left\{ i d x _ { \mathrm { c o l } } ^ { k , i } ( t ) = n \right\} } \cdot d _ { \mathrm { c o l } } ^ { k , i } ( t ) } { f _ { c } ^ { k } ( t ) }\tag{3}
$$

where $f _ { c } ^ { k } ( t )$ , related to CPU-cycle frequency, is the $E _ { k } \mathrm { ^ { \circ } s }$ edge execution data rate for the preset tasks at time slot t.

On each edge device $E _ { k }$ , collected data buffer $D _ { \mathrm { c o l } } ^ { k } ( t )$ cache those data from data sources but have not been edge processed. Assume that in every operation slot, $E _ { k }$ allocates its edge computation resources for one data piece in the buffer. Thus, the edge execution decision at the tth slot can be denoted by a one-hot vector

TABLE I MAIN NOTATIONS
<table><tr><td colspan="4"></td></tr><tr><td colspan="3"></td></tr><tr><td rowspan="8">Environment Notations</td><td> $N _ { e }$ </td><td>The number of the edge devices.</td></tr><tr><td></td><td>The set of data sources.</td></tr><tr><td> $\begin{array} { r } { \mathbb { S } = \{ S _ { 1 } , \bar { \cdot } \cdot \cdot , S _ { N _ { s } } \} } \\ { \mathbb { E } = \{ E _ { 1 } , \bar { \cdot } \cdot \cdot , E _ { N _ { e } } \} } \\ { \mathbb { C } \qquad \mathbb { C } } \end{array}$ </td><td>The set of edge devices.</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">Cloud data center. Attributes of each packet considered in this research,i.e.,the data size,  $d . ( t ) , w . ( t ) , i d x . ( t )$ </td></tr><tr><td colspan="2">the elapsed time and the index of its source.  $D _ { n } ( t ) = \left\{ \left[ d _ { n , i } ( t ) , w _ { n , i } ( t ) , n \right] \right\}$   $S _ { n } \mathbf { \dot { s } }$  data buffer at t-th time slot.</td></tr><tr><td colspan="2"> $\Delta ( t ) = \left\{ \Delta _ { 1 } ( t ) , \cdot \cdot \cdot , \Delta _ { N _ { s } } ( t ) \right\} ^ { \prime }$  The list storing the age of all data sources at t-th time slot.</td></tr><tr><td colspan="2" rowspan="2"> $D _ { c o l } ^ { k } ( t ) = \left\{ \left[ d _ { c o l } ^ { k , i } ( t ) , w _ { c o l } ^ { k , i } ( t ) , i d x _ { c o l } ^ { k , i } ( t ) \right] \right\}$   $E _ { k } { ' } s$ </td></tr><tr><td rowspan="3">collected data buffer at caching the data collected but not processed. executed data buffer caching the processed data waiting to be offloaded.</td></tr><tr><td> $D _ { e x e } ^ { k } ( t ) = \left\{ \left[ d _ { e x e } ^ { k , i } ( t ) , w _ { e x e } ^ { k , i } ( t ) , i d x _ { e x e } ^ { k , i } ( t ) \right] \right\}$   $E _ { k } \mathrm { ' s }$  B The collected data buffer size of</td></tr><tr><td> $E _ { k }$  The executed data buffer size of  $E _ { k } .$  The position of  $E _ { k }$  at t-th time slot.</td></tr><tr><td rowspan="6">Learning Framework Notations</td><td colspan="2"> $\{ \mathbf { } a _ { k } ( t ) \} = \left\{ \left[ \mathbf { m o v e } _ { k } ( t ) , \ \mathbf { e x e } _ { k } ( t ) , \ \mathbf { o f f } _ { k } ( t ) \right] \right\}$   $\pmb { b } \dot { ( } t ) = [ b _ { 1 } ( t ) , \cdots , b _ { N _ { e } } ( t ) ]$   $\overline { { \mathcal { A } _ { k } , \mathcal { A } _ { k } ^ { \prime } } }$ </td></tr><tr><td colspan="2">The actor net and target actor net on k-th learning agent. The critic net and target critic net on k-th learning agent.  $\pmb { \mathcal { A } } = \left\{ \mathcal { A } _ { 1 } , \cdot \cdot \cdot , \overset { \cdot \cdot } { \mathcal { A } } _ { N _ { e } } , \mathcal { A } _ { \mathbb { C } } \right\}$   $\pmb { c } = \{ { \mathcal { C } } _ { 1 } , \cdot \cdot \cdot , { \mathcal { C } } _ { N _ { e } } , { \mathcal { C } } _ { \mathbb { C } } \}$ </td></tr><tr><td colspan="2">The set of actor net learning agents for edge devices and center controller. The set of critic net learning agents for edge devices and center controller.</td></tr><tr><td colspan="2" rowspan="2">The parameters of k-th learning agent&#x27;s actor net and target actor net. Learning rates for actor/critic nets. Experience buffer for replay.</td></tr><tr><td rowspan="3">The set of target actor net learning agents corresponding to A. The set of target critic net learning agents corresponding to C.</td></tr><tr><td> $\pmb { \mathscr { s } } ^ { \prime } = \left\{ \mathcal { A } _ { 1 } ^ { \prime } , \cdot \cdot \cdot , \mathcal { A } _ { N _ { o } } ^ { \prime } , \mathcal { A } _ { \mathbb { C } } ^ { \prime } \right\}$   $\pmb { \mathcal { C } } ^ { \prime } = \left\{ \mathcal { C } _ { 1 } ^ { \bar { \prime } } , \cdots , \mathcal { C } _ { N _ { e } } ^ { \bar { \prime } ^ { \mathrm { ~ \tiny ~ { ~ \ddots ~ } ~ } } } , \mathcal { C } _ { \mathbb { C } } ^ { \prime } \right\}$   $\{ \pmb { s } _ { k } ( t ) \} , \pmb { s } _ { \mathbb { C } } ( \breve { t } )$ </td></tr><tr><td colspan="2" rowspan="2"> $\pmb { \theta } _ { k } , \pmb { \theta } _ { k } ^ { \prime }$   $\phi _ { k } , \phi _ { k } ^ { \prime }$   $\eta _ { \mathrm { { A } } } , \eta c$  B  $\gamma \in [ 0 , 1 ]$   $\tau \in [ 0 , 1 ]$   $\omega \in [ 0 , 1 ]$ </td></tr><tr><td>The input states of edge device and center controller at t-th time slot. The parameters of k-th learning agent&#x27;s actor net and target actor net.</td></tr></table>

$$
\mathbf { e x e } _ { k } ( t ) = \left[ \mathrm { e x e } _ { 1 } ^ { k } ( t ) , \cdot \cdot \cdot , \mathrm { e x e } _ { B _ { \mathrm { c o l } } ^ { k } } ^ { k } ( t ) \right]\tag{4}
$$

where $\textstyle \operatorname { e x e } _ { i } ^ { k } ( t ) \in \{ 0 , 1 \}$ is the CPU allocation flag for each data piece in $E _ { k } \mathrm { ^ { \circ } s }$ collected data buffer and it obviously satisfies the condition

$$
\sum _ { i = 1 } ^ { B _ { \mathrm { c o l } } ^ { k } } \mathrm { e x e } _ { i } ^ { k } ( t ) = 1 \qquad \mathrm { f o r } \ k = 1 , \ldots , N _ { e } .\tag{5}
$$

After local execution on edge, the data will be cached in executed data buffers, $\{ D _ { \mathrm { e x e } } ^ { k } ( t ) \}$ , and wait to be offloaded to the cloud center. Similarly, assume that in every operation slot, $E _ { k }$ decides to offload one piece of packets in $D _ { \mathrm { e x e } } ^ { k } ( t )$ . Thus, the offloading scheduling can be described by a one-hot vector

$$
\mathbf { o f f } _ { k } ( t ) = \left[ \mathrm { o f f } _ { 1 } ^ { k } ( t ) , \cdot \cdot \cdot , \mathrm { o f f } _ { B _ { \mathrm { e x e } } ^ { k } } ^ { k } ( t ) \right]\tag{6}
$$

where $\mathrm { o f f } _ { i } ^ { k } ( t ) \in \{ 0 , 1 \}$ are the offloading decisions for each data piece, and they also satisfy the condition

$$
\sum _ { i = 1 } ^ { B _ { \mathrm { e x e } } ^ { k } } \mathrm { o f f } _ { i } ^ { k } ( t ) = 1 \qquad \mathrm { f o r } \ k = 1 , \dots , N _ { e } .\tag{7}
$$

## D. Communication Model

There are three kinds of transmission links between the entities in the environment, containing source-edge, edge-edge, and edge cloud. In this work, we investigate the cases where edge devices only share the states, observations and learning parameters which means that neither data nor tasks are transmitted between edge devices. Therefore, the transmission costs of edge-edge communication shall be negligible. The transmission process of source-edge and edge-cloud can be modelled as an air to ground (A2G) channel [26] where Line-of-Sight (LoS) path loss as well as non-LoS (NLoS) loss shall be considered [47]

$$
P L _ { \xi } ( t ) = \left( \frac { 4 \pi f } { c } \right) ^ { 2 } \cdot d ^ { 2 } ( t ) \cdot \eta _ { \xi }\tag{8}
$$

where $d ( t ) = \sqrt { x ^ { 2 } ( t ) + y ^ { 2 } ( t ) + h ^ { 2 } ( t ) }$ is the distance between the edge device and the ground entity (a chosen data sources or cloud center), f is the carrier frequency and c is the speed of light. Besides, $\eta _ { \xi }$ with $\xi = \{ 0 , 1 \}$ represents the excessive path loss of LoS and NLoS cases. Hence, the average A2G path loss of the communication channel for $E _ { k } - S _ { n }$ (or $E _ { k } - \mathbb { C } )$ at tth slot can be obtained by

$$
\overline { { L } } _ { k , n } ( t ) = p _ { 0 } ( t ) \cdot P L _ { 0 } ^ { k , n } ( t ) + p _ { 1 } ( t ) \cdot P L _ { 1 } ^ { k , n } ( t )\tag{9}
$$

where $p _ { 0 } ( t ) , p _ { 1 } ( t )$ are the probability of LoS and NLoS which can be closely approximated by the following form:

$$
p _ { 0 } ( t ) = \frac { 1 } { 1 + a \exp ( - b ( \psi - a ) ) }\tag{10}
$$

where $\psi = \tan ^ { - 1 } ( [ h ( t ) ] / [ \sqrt { x ^ { ( } t ) + y ^ { 2 } ( t ) } ] )$ is the angle between the edge-ground link and the horizontal plane. Moreover, a and b are parameters related to the environment. Then, considering the frequency division mode with total bandwidth W, for the channel with allocated bandwidth proportion $b _ { k , n } ( t )$ the transmission rate between $E _ { k }$ and $S _ { n }$ (or C) can be expressed by

$$
R _ { k , n } ( t ) = b _ { k , n } ( t ) W \log _ { 2 } \biggl ( 1 + \frac { P _ { \mathrm { t r } } ^ { k } ( t ) } { \bar { L } _ { k , n } ( t ) N _ { 0 } b _ { k , n } ( t ) W } \biggr )\tag{11}
$$

where $N _ { 0 }$ is the noise power spectral density and $P _ { \mathrm { t r } } ^ { k } ( t )$ represents the power for transmission satisfying

$$
0 \leq P _ { \mathrm { t r } } ^ { k } ( t ) \leq P _ { \mathrm { t r , m a x } } ^ { k } .\tag{12}
$$

## E. Problem Formulation

In the age sensitive scenarios, the freshness of the data shall be significantly stressed. Recapping the concept, AoI [48], in the system introduced above, the age of data source $S _ { n }$ at tth time slot is defined as the subtraction of current time and the generation time of the latest data at the receiver [49], which can be expressed as

$$
\Delta _ { n } ( t ) = t - T _ { g } ^ { n } ( t )\tag{13}
$$

where $T _ { g } ^ { n } ( t )$ denotes the generation time of $S _ { n } \mathrm { ^ { * } s }$ latest data packet received by the cloud center. Different from the delay of each data packet, $\Delta _ { n } ( t )$ is a duration measure for each data source which implies how frequently the data of $S _ { n }$ are collected, edge executed and offloaded, which is especially important to such time-sensitive MEC system. Thus, an $N _ { s ^ { - } }$ dimension vector $\pmb { \Delta } ( t )$ is used to record the age of each data source. To maintain the timeliness, the target is to minimize the average age of all data sources by controlling the edge devices, scheduling the data and allocating the resources effectively. Therefore, taking the above system models into account as the constraints, we obtain the following optimization problem:

$$
\mathcal { P } _ { 1 } : \operatorname* { m i n } _ { \{ \pmb { a } _ { k } ( t ) \} , \pmb { b } ( t ) } \left[ \overline { { \pmb { \Delta } } } ( t ) : = \frac { 1 } { N _ { s } } \sum _ { n = 1 } ^ { N _ { s } } \Delta _ { n } ( t ) \right]\tag{14}
$$

Before solving the optimization problem above, let us rethink the introduced system models and corresponding problems from following three perspectives. First, the constraints of $\mathcal { P } _ { 1 }$ are heterogeneous and the optimization objectives are of two stages (edge stage and cloud stage). Thus, the optimal solutions cannot be explicitly expressed and an iterative algorithm shall be adopted. Second, note that $\mathcal { P } _ { 1 }$ is an instant optimization problem and the environment sates are stochastic, which means that the strategies for each time slot should be variant. Moreover, because $\Delta _ { n } ( t )$ depends on not only current states but also the states of previous time, the optimal solutions for each time slot may not lead to full-time optimization. For above reasons, we are to formulate the optimization problem into MDP game [51] and discuss the RL-based solutions [50].

## III. EDGE FEDERATED ACTOR–CRITIC LEARNINGFRAMEWORK FOR MULTIAGENT COOPERATION

In this section, an MDP will be modelled for optimization problem $\mathcal { P } _ { 1 }$ . And then, with the MDP formulation for MEC collaboration, we shall adopt MARL approaches to solve the MDP problem. Q-Learning and DQN are popular value-based RL methods which learn the action-value function $Q ( \pmb { s } , \pmb { a } )$ related to system reward/penalty. However, while the action spaces grow too large, the search for optimal actions becomes extremely hard. To overcome the complexity of the action space, policy-based methods, such as A2C and DDPG are introduced, where dual neural networks are employed to estimate the action a and Q value, respectively. Moreover, in multiagent scenes, the single learning agent mode or centralized RL requests a large neural network with a complex structure and massive model parameters, which may suffer from some difficulties on training convergence and model generalization [52], [53]. For above reasons, we present a heterogeneous multiagent actor critic learning framework (H-MAAC) for MEC collaboration and the corresponding algorithm to solve the optimization problem. Besides, the convergence analysis of the proposed algorithm will be given.

## A. MDP Formulation for MEC Collaboration

MDP is a common model to formulate such environmentinteractive systems [54]. An MDP game can be expressed by a tuple with four elements, M{S, A, R, T }, standing for the states, actions, rewards and transition policies, respectively. Note that in optimization problem $\mathcal { P } _ { 1 }$ , the target is to minimize the objective average AoI. Therefore, we rewrite the element R as $P$ to represent the penalty of the game and obtain the substitute tuple of MDP formulation, M{S, A, P, T }. Furthermore, a multiagent extended version of MDP contains a number of agents to match the scenarios with multiple controllable entities. In the above MEC collaboration system, all mobile-edge devices and the center controller can be regarded as the agents. Overall we have $N _ { e }$ edge agents and one center agent. All the agents observe their states and act with certain strategies.

1) States: For edge agents, their states $\{ s _ { k } ( t ) \}$ contain the local observations of the environment, the status of the edge devices, including buffer states, allocated offloading bandwidth, etc. The states of center agent, $\pmb { S } _ { \mathbb { C } } ( t )$ , consists of all edge device status.

2) Actions: In the cases of interest, edge devices move, collect data, locally execute and offload tasks to cloud center. Then, the edge agents’ actions are composed of movement, execution decision and offloading scheduling, denoted by

$$
\{ { \pmb a } _ { k } ( t ) \} = \{ [ { \bf m o v e } _ { k } ( t ) , ~ { \bf e x e } _ { k } ( t ) , ~ { \bf o f f } _ { k } ( t ) ] \} .\tag{15}
$$

Meanwhile, the cloud center controller allocates the offloading bandwidth for each edge device. Thus, the action of center agent, $\mathbf { a } _ { \mathbb { C } } ( t )$ , is the one-sum bandwidth proportion vector ${ \pmb b } ( t )$

<!-- image-->  
Fig. 2. Architecture of the proposed H-MAAC-based RL collaboration framework.

3) Penalties: Since we focus on the edge collaboration in age sensitive MEC system where all agents collaborate to minimize the average age of data sources, the global penalty will be shared for all agents. Then, the current penalty at tth slot for each agent can be described as

$$
p _ { k } ( t ) = \overline { { \Delta } } ( t ) \mathrm { ~ f o r ~ } k = 1 , \ldots , N _ { e } , \mathbb { C } .\tag{16}
$$

To investigate the global optimization of the system, the following long-time penalty with decay is considered:

$$
P _ { k } ( t ) = \sum _ { i = 0 } ^ { T } \gamma ^ { i } \cdot p _ { k } ( t + i )\tag{17}
$$

where $\gamma$ is the decay coefficient and T is the length of time window.

4) Transition Policies: For MEC system, it is hard to find a formatted policy to cover all the state transitions of data sources, edge devices, cloud center as well as resource allocation. As a result, we use

$$
\begin{array} { r } { \mathcal { T } \Big ( \{ s _ { k } ( t + 1 ) \} , s _ { \mathbb { C } } ( t + 1 ) \Big | \{ s _ { k } ( t ) \} , s _ { \mathbb { C } } ( t ) , \{ a _ { k } ( t ) \} , a _ { \mathbb { C } } ( t ) \Big ) } \end{array}\tag{18}
$$

to represent the entities’ interactions in the system.

## B. Edge-Federated Heterogeneous Multiagent Actor–Critic Framework

To control the edge devices and optimize the center bandwidth allocation, we present a heterogeneous multiagent actor–critic (H-MAAC) framework. The construction of the framework and the learning procedures are shown in Fig. 2, where learning agents interact with the environment, memorize the experience replay and learn the optimal actions to minimize the system penalty [55]. The heterogeneity comes from three aspects: 1) the multimodality of input states; 2) the multiple output actions; and 3) the ensemble neural network models to learn the mixed policies. Overall, dual lightweight neural networks are built for each learning, containing original actor/critic nets and target actor/critic nets. Since the structures of the networks are relatively simple which will cost less computation resources, the framework can be deployed on each edge devices. The details of the framework are as follows.

1) Neural Network Design: Actor nets, $\mathcal { A } _ { k } ( s _ { k } ( t ) ; \pmb \theta _ { k } )$ , take the states of each agent as input and output the current actions ${ \pmb a } _ { k } ( t )$ . Since the input states and the output actions are multimodal, the network should be designed according to the specific data structures. For edge agents, we build a multiple input-output neural network for each device to learn the diversified actions through the ensemble states, including the local observation for data sources, the edge buffer states and the offloading channel states. The structure of the edge agent actor net is shown in Fig. 3(a) where a convolutional neural network (CNN) with average pooling is employed to obtain the movement action while multilayer perceptrons (MLPs) are employed for edge execution as well as offloading scheduling. Specifically, we format the local observation for data sources into an $r _ { \mathrm { o b s } } ^ { k } \times r _ { \mathrm { o b s } } ^ { k } \times 2$ map. The observation maps can be regarded as 2-channel images where the third dimension refers to the aggregated size and delay of the data packets sensed by edge device $E _ { k }$ . Inspired by the successful applications of CNNs in computer visions, we adopt a CNN-based neural network to extract the area with larger data packets as well as higher AoI. Then, through the average pooling, we project the $r _ { \mathrm { o b s } } ^ { k ^ { - } } \times r _ { \mathrm { o b s } } ^ { k } \times 2$ observation map onto the $r _ { \mathrm { m o v e } } ^ { k } \times r _ { \mathrm { m o v e } } ^ { k } \times 1$ movement map to decide the trajectory actions. The other inputs, buffer states and allocated bandwidth are formatted as lists and scalars. And the data scheduling vectors are outputted by an MLP net. For center actor net, its inputs consist of the state lists of the edge devices and the output is a one-sum vector for bandwidth allocation. As for the center agent actor, we use an MLP to combine the multiple edge state lists to allocate bandwidth proportion for edge-center communication, as shown in Fig. 3(c). Additionally, on each learning agent, a critic net, $\mathcal { C } _ { k } ( s _ { k } ( t ) , \pmb { a } _ { k } ( t ) ; \pmb { \phi } _ { k } )$ , is also deployed to approximate the action-value function $Q ( s _ { k } ( t ) , \pmb { a } _ { k } ( t ) )$ with the current states and actions as inputs. Note that the objectives of all $\mathcal { C } _ { k }$ are consistent in order to minimize the average age of the whole system. The critic nets are designed by the main structures of actor nets plugging the layers for action evaluation, as shown in Fig. 3(b) and (d).

2) Target Net Update: In addition to the original actor– critic nets, the target actor–critic nets are also built. Target nets have the same structure and initialization as the original nets. While training the network parameters, target nets estimate the future actions $\pmb { a } _ { k } ^ { \prime } ( t + 1 )$ as well as $\mathcal { Q } ^ { \prime } ( s _ { k } ^ { \prime } ( t + 1 ) , \pmb { a } _ { k } ^ { \prime } ( t + 1 ) )$ values based on the states of next slot. The employ of target nets improves the stability and convergence of replay training [56]. The parameters of target nets are slowly updated by the original nets every $T _ { u }$ period with the mixing weight τ , i.e.,

$$
\begin{array} { r l } & { \pmb { \theta } _ { k } ^ { \prime } = \tau \pmb { \theta } _ { k } ^ { \prime } + ( 1 - \tau ) \pmb { \theta } _ { k } } \\ & { \pmb { \phi } _ { k } ^ { \prime } = \tau \pmb { \phi } _ { k } ^ { \prime } + ( 1 - \tau ) \pmb { \phi } _ { k } . } \end{array}\tag{19}
$$

<!-- image-->  
Fig. 3. Neural network models of each actor–critic agent. (a) Edge Actor Net. (b) Center Actor Net. (c) Edge Critic Net. (d) Center Critic Net.

3) Experience Replay: To improve the online learning efficiency, an experience replay approach is exploited here. The interactions between the entities and the environment, denoted as tuples, $\{ s ( t ) , \pmb { a } ( t ) , \overline { { \Delta } } ( t ) , s ^ { \prime } ( t + 1 ) \}$ , are stored in an experience buffer B. The experience buffer is of finite capacity. When the buffer is full, new records will replace the oldest ones. As for the sampling rules, to guarantee synchronization of learning and the environment interactions, the latest $B / 2$ data are always exploited for training. In each learning epoch, each agent samples interaction data with batch size B from the experience buffer and updates the network parameters with the learning agents $\eta _ { A }$ and $\eta _ { \mathcal { C } }$ . Specifically, the critic nets are updated by minimizing the MSE loss function

$$
\ell _ { \mathcal { C } _ { k } } ( \pmb { \phi } _ { k } ) : = E \Big [ \big \| \mathcal { C } _ { k } ( s _ { k } , \pmb { a } _ { k } ; \pmb { \phi } _ { k } ) - \hat { y _ { k } } \big \| ^ { 2 } \Big ]\tag{20}
$$

where

$$
\hat { y } _ { k } = \overline { { \Delta } } + \gamma \mathcal { C } _ { k } ^ { \prime } \big ( s _ { k } ^ { \prime } , \pmb { a } _ { k } ^ { \prime } ; \pmb { \phi } _ { k } ^ { \prime } \big )\tag{21}
$$

is the estimated long-time $Q$ value. Since the goal is to minimize the penalty, the loss function of actor nets can be defined as the predicted Q value

$$
\ell _ { \mathcal { A } _ { k } } ( \pmb \theta _ { k } ) : = \mathcal { C } _ { k } \big ( \pmb s _ { k } , \mathcal { A } _ { k } ( \pmb s _ { k } ; \pmb \theta _ { k } ) ; \pmb \phi _ { k } \big ) .\tag{22}
$$

Algorithm 1 Experience Replay Procedure   
1: for each agent k in $\overline { { \{ 1 , \ldots , N _ { e } , \mathbb { C } \} } }$ do   
2: Sample $\left\{ s _ { k } , \pmb { a } _ { k } , \overline { { \Delta } } , s _ { k } ^ { \prime } \right\}$ from B[k];   
3: Predict new actions: $\ddot { \pmb { a } } _ { k } ^ { \prime } = \mathcal { A } _ { k } ^ { \prime } ( s _ { k } ^ { \prime } ; \pmb { \theta } _ { k } ^ { \prime } ) ;$   
4: Predict new Q-value: $\begin{array} { r } { \hat { \boldsymbol { Q } } ^ { \prime } ( s _ { k } ^ { \prime } , \hat { \pmb { a } } _ { k } ^ { \prime } ) ^ { \wedge } = \hat { \mathcal { C } } _ { k } ^ { \prime } ( s _ { k } ^ { \prime } , \pmb { a } _ { k } ^ { \prime } ; \pmb { \phi } _ { k } ^ { \prime } ) ; } \end{array}$   
5: Calculate ${ \hat { y } } _ { k } \ b y \ \operatorname { E q . } ( 2 1 ) ;$   
6: Calculate $\ell _ { \mathcal { C } _ { k } } ( \pmb { \phi } _ { k } ) , \ell _ { \mathcal { A } _ { k } } ( \pmb { \theta } _ { k } )$ by Eq.(20), Eq.(22);   
7: C A Update network parameters using SGD optimizer:   
$\pmb { \phi } _ { k _ { \perp } } ^ { t + 1 }  \pmb { \phi } _ { k } ^ { t } - \eta _ { \mathcal { C } } \nabla _ { \phi } \tilde { \ell } _ { \mathcal { C } _ { k } } ( \pmb { \phi } _ { k } ^ { t } ) ,$   
$\pmb { \theta } _ { k } ^ { i + 1 }  \pmb { \theta } _ { k } ^ { i } - \eta _ { \mathcal { A } } \nabla _ { \theta } \tilde { \ell } _ { \mathcal { A } _ { k } } ( \pmb { \theta } _ { k } ^ { i } ) .$   
8: end for

Then, we summarize the training procedure at tth epoch as Algorithm 1.

4) Edge-Federated Mode: Note that it is a cooperative model and the penalties of edge agents are identical. Commonly speaking, so as to reach the global optimum, the cross-communication is required in such multiagent learning scenarios to share the knowledge of different agents. However, in the MEC system of interests, the encoding of the multimodal input states is hard to design. What is more, the transmission and the processing of the observation will cost excessive communication and computation resources. Hence, to overcome these difficulties, inspired by the concept of FL [57], we propose an edge-federated mode for the above framework, where every $E _ { f }$ learning epoch, all edge agents share their actor net parameters and carry out the federated updating. Under the proposed updating rule, each edge agent preserves the parameters with weight ω and mixes the others’ parameters, which can be formulated by

$$
\pmb { \theta } ^ { t + 1 } = \pmb { \theta } ^ { t } \cdot \pmb { \Omega }\tag{23}
$$

where $\pmb { \theta } ^ { t } = [ \pmb { \theta } _ { 1 } ^ { t } , \dots , \pmb { \theta } _ { N _ { e } } ^ { t } ]$ denotes the vector of all edge actor nets at the tth learning epoch and  denotes the federated updating matrix

$$
\pmb { \Omega } = \left[ \begin{array} { c c c c } { \omega } & { \frac { 1 - \omega } { N _ { e } - 1 } } & { \cdot \cdot \cdot } & { \frac { 1 - \omega } { N _ { e } - 1 } } \\ { \frac { 1 - \omega } { N _ { e } - 1 } } & { \omega } & { \cdot \cdot \cdot } & { \frac { 1 - \omega } { N _ { e } - 1 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \frac { 1 - \omega } { N _ { e } - 1 } } & { \frac { 1 - \omega } { N _ { e } - 1 } } & { \cdot \cdot \cdot } & { \omega } \end{array} \right] .\tag{24}
$$

On the one hand, edge-federated provide the model-wise communication for each edge agent. Instead of sharing the input states, only the parameters of the lightweight actor nets are transmitted, which improves the communication efficiency of the system [58] and such communication costs can be neglected compared to the data volume in the considered MEC system. On the other hand, the edge-federated mode performs better learning convergence which will be discussed in Section IV.

5) Exploitation-Exploration: Online learning suffers from the exploitation-exploration dilemma, namely, the agents tend to repeat the previous actions which may cause trap at some position and the loss of exploration in the MEC system. To avoid such phenomenon during the online interactions with the MEC environment, we adopt an 
-exploration approach [16] which enforces random actions with the probability 
.

## C. Online Learning Algorithms for Multiagent Cooperation

Based on the proposed learning framework, we develop the corresponding online multiagent collaboration as Algorithm 2 where the agents learn and update the optimal strategies while the MEC system works continuously. More specifically, each epoch comprises four procedures: 1) lines 1–11 are the acting procedure with 
-exploration, where agents choose whether to act randomly or to follow the actor net strategies; 2) lines 13–17 are the replay training procedure of the networks which will be skipped if the count of samples in B is insufficient; 3) lines 18–20 are the periodical target net updating procedure; and 4) lines 21–23 are the edge-federated updating procedure.

## IV. CONVERGENCE ANALYSIS

In this section, we shall investigate the convergence of the collaboration algorithms and show that the edge-FL mode has a better performance in terms of convergence rate and stability, compared to the original H-MAAC.

First, consider the convergence of actor nets, the objective function is defined by the average loss of all edge agents’ actor

```latex
Algorithm 2 EdgeFed H-MAAC Online Collaboration
Initialization: Initialize system parameters and hyper parameters for
learning.
Initialization: Initialize net parameters $\pmb { \theta } _ { k } , \ \pmb { \phi } _ { k }$ and set target nets:
$\begin{array} { r } { \pmb { \theta } _ { k } ^ { \prime }  \pmb { \theta } _ { k } , \pmb { \phi } _ { k } ^ { \prime }  \pmb { \phi } _ { k } . } \end{array}$
1: for epoch t = 1 to MAX_EPOCH do
2: Randomly generate $q \in [ 0 , 1 ] ;$
3: for each agent k in $\{ 1 , \ldots , N _ { e } , \mathbb { C } \}$ do
4: if $p < \epsilon \ \mathrm { o r } \ | \pmb { \mathcal { B } } [ k ] | < B$ then
5: Randomly choose actions ${ \pmb a } _ { k } ( t ) ;$
6: else
7: Ensemble local observation and states: $s _ { k } ( t ) ;$
8: Set actions: $\pmb { a } _ { k } ( t ) = \mathcal { A } _ { k } ( s _ { k } ( t ) ; \pmb { \theta } _ { k } ) ;$
9: end if
10: end for
11: Interact with environment and obtain ${ \overline { { \Delta } } } ( t ) , s ^ { \prime } ( t + 1 ) ;$
12: Add s, a, , s  into ;
13: for each agent k in $\{ 1 , \ldots , N _ { e } , \mathbb { C } \}$ do
14: if | [k]| ≥ B then
15: Run replay procedure, Algorithm 1;
16: end if
17: end for
18: if t mod $T _ { u } = = 1$ then
19: Update target nets, as Eq.(19);
20: end if
21: if t mod $E _ { f } = = 1$ then
22: Run edge-federated updating, as Eq.(23);
23: end if
24: end for
```

nets, i.e.,

$$
\mathcal { L } \Big ( \overline { { \pmb { \theta } } } \Big ) = \frac { 1 } { N _ { e } } \sum _ { k = 1 } ^ { N _ { e } } \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \pmb { \theta } } } _ { k } \Big )\tag{25}
$$

where $\ell _ { \mathcal { A } _ { k } } ( \overline { { \pmb \theta } } _ { k } ) : = \mathcal { C } _ { k } ( s _ { k } , \mathcal { A } _ { k } ( s _ { k } ; \overline { { \pmb \theta } } _ { k } ) ; \pmb \phi _ { k } )$ is the loss of $\mathcal { A } _ { k }$ for edge agent $E _ { k }$ and $\overline { { \pmb { \theta } } } _ { k }$ represents the updated model parameters of $\boldsymbol { \mathcal { A } } _ { k }$ under the federated rule, (23).

Second, since the proposed algorithm is online and the environment is dynamic, the training may not guarantee either the global optimal or the penalty stability. Hence, we investigate the learning convergence from an alternative perspective, the gradients’ 2-norm time average of all edge actor nets

$$
\frac { 1 } { N _ { e } T } \sum _ { t = T _ { 0 } + 1 } ^ { T _ { 0 } + T } \sum _ { k = 1 } ^ { N _ { e } } \left. \nabla \ell _ { \mathcal { A } _ { k } } \left( \overline { { \pmb { \theta } } } _ { k } ^ { t } \right) \right. ^ { 2 }\tag{26}
$$

where T is the time horizon length after $T _ { 0 } \mathrm { { t h } }$ learning epoch. Additionally, we denote the training interaction sets of $E _ { k }$ from epoch $T _ { 0 } + 1$ to $T _ { 0 } + T$ as

$$
\{ \mathcal { T } _ { k } \} ^ { T _ { 0 } } : = \{ \mathcal { T } _ { k } ( T _ { 0 } + 1 ) , \ldots , \mathcal { T } _ { k } ( T _ { 0 } + T ) \}\tag{27}
$$

where $\mathcal { T } _ { k } ( t ) ~ = ~ \{ s _ { k } , \pmb { a } _ { k } , \overline { { \Delta } } , s _ { k } ^ { \prime } \} ^ { t }$ represents the interaction records used to train $E _ { k } \mathrm { ^ { \circ } s }$ neural networks at tth learning epoch.

## A. Assumptions

Before the analysis, let us illustrate some assumptions similar to [59], under which the convergence theorem can be carried out.

1) Existence of Optimal Loss: While the environment state sets are given, we assume that the optimal net parameters, $\pmb { \theta } _ { k } ^ { \ast } ,$ and the corresponding actor loss, $\ell _ { A _ { k } } ^ { * }$ , of each edge agent exist. This assumption is intuitive because the action space is finite and the loss is related to system penalty.

2) Independence of Center Agent Learning: Since we consider the FL mode of edge agents, to eliminate the influence of center agent, we assume that the training process of center agent is independent with the edge agents’. Consequently, the learning of center actor–critic parameters, $\pmb { \theta } _ { \mathbb { C } }$ and $\phi _ { \mathbb { C } }$ can always reach the optimal parameters and be out of account while discussing the edge learning convergence.

3) Fine-Fitness of Critic Nets: Note that each critic net is trained to predict the common penalty value Q related to the system average age, -(t). Owing to the large learning capacity of neural networks, we assume that after T0th epoch, for given training set $\{ \mathcal { T } _ { k } \} ^ { T _ { 0 } }$ , critic nets converge and fit the Q values well, which refers to that the critic nets’ parameters $\phi _ { k }$ keep the consistent values of $\pmb { \phi } _ { k } ^ { T _ { 0 } * }$ and the loss $\ell _ { \mathcal { A } _ { k } } ( \pmb \theta _ { k } ) = \mathcal { C } _ { k } ( s _ { k } , \mathcal { A } _ { k } ( \pmb s _ { k } ; \pmb \theta _ { k } ) ; \pmb \phi _ { k } ^ { * } )$ can be regarded as a determined function with fixed $\phi _ { k } ^ { * }$

4) Conditional L-Smoothness for Given Environment States: In general, neural networks are neither smooth nor convex. However, for given training set $\{ \mathcal { T } _ { k } \} ^ { T _ { 0 } }$ , the Lipschitz constant of a model consisting of MLP or CNN can be estimated [60]. Besides, the activation functions used in proposed net, such as ReLU and Softmax are Lipschitz continuous and differentiable [61]. Therefore, we can denote the Lipschitz constant of each actor net as $L \tau _ { k }$ which is related to its inputs $\{ \mathcal { T } _ { k } \} ^ { T _ { 0 } }$ from $T _ { 0 } + 1 \mathrm { t h }$ to T0 + Tth training epoch. Then, the conditional L-smoothness of each actor net’s loss function can be expressed as

$$
\left\| \nabla \ell _ { \mathcal { A } _ { k } } ( \pmb \theta _ { k } ) - \nabla \ell _ { \mathcal { A } _ { k } } \big ( \pmb \theta _ { k } ^ { \prime } \big ) \right\| \leq L \tau _ { k } \left\| \pmb \theta _ { k } - \pmb \theta _ { k } ^ { \prime } \right\| .\tag{28}
$$

5) Unbiased Bounded SGD: In each epoch, each agent samples a mini batch with size B from the experience memory. Consider an stochastic gradient descent (SGD)-based optimizer is applied for back propagation, the unbiased stochastic gradient $\tilde { \pmb { g } } _ { k } ^ { \prime } \tilde { \pmb { \jmath } } _ { k } ^ { \prime }$ s variance is bounded by

$$
E \Big [ \big \| \tilde { \pmb { g } } _ { k } - \pmb { g } _ { k } \big \| ^ { 2 } \Big ] \leq C \big \| \pmb { g } _ { k } \big \| ^ { 2 } + \frac { \sigma _ { \mathscr { T } _ { k } } ^ { 2 } } { B }\tag{29}
$$

where $\pmb { \mathrm { g } } _ { k } = E [ \tilde { \pmb { g } } _ { k } ] = \nabla \ell _ { A _ { k } } ( \pmb { \theta } _ { k } )$ is the average stochastic gradient for given training input states $\{ \mathcal { T } _ { k } \} ^ { T _ { 0 } }$ and $\sigma \tau _ { k }$ is the related variance constant. Moreover, C is the non-negative constant for all edge agents.

Briefly, assumptions 1)–3) are reasonable to the MEC environment while 4) and 5) are assumptions for the learning progress.

## B. Convergence Theorem for EdgeFed H-MAAC

Under the assumptions above, the learning convergence can be conducted and the results can be summarized as the following theorem.

Theorem 1: For the proposed EdgeFed H-MAAC algorithm with the update period $E _ { f }$ , and under assumptions 1)–5), if the

learning rates of all edge agents are set to be η which satisfies

$$
\begin{array} { r l r } {  { [ \frac { 2 C E _ { f } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \xi ^ { 2 } } + \frac { E _ { f } ^ { 2 } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \xi } ( \frac { 2 \zeta } { 1 + \zeta } + \frac { 2 \zeta } { 1 - \zeta } + \frac { E _ { f } - 1 } { E _ { f } } ) ] \eta ^ { 2 } } } \\ & { } & { + L _ { \mathrm { m a x } } ( C + 1 ) \eta - 1 \le 0 \qquad ( 3 } \end{array}\tag{0}
$$

where $\zeta ~ = ~ [ ( N _ { e } \omega - 1 ) / ( N _ { e } - 1 ) ]$ is the second maximum eigenvalue of the updating matrix  and $L _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } ~ L _ { s _ { k } }$ Then, the time average squared gradient norm after T0th epoch is bounded by

$$
\begin{array} { r l } & { E \Bigg [ \frac { 1 } { N _ { e } T } \underset { t = T _ { 0 } + 1 } { \overset { T _ { 0 } + T } { \sum } } \underset { k = 1 } { \overset { N _ { e } } { \sum } } \Big \lVert \nabla \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { t } \Big ) \Big \rVert ^ { 2 } \Bigg ] } \\ & { \leq \frac { 2 \sum _ { k = 1 } ^ { N _ { e } } \Big [ \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { T _ { 0 } } \Big ) - \ell _ { \mathcal { A } _ { k } } ^ { * } \Big ] } { \eta N _ { e } T } + \ \frac { \eta } { N _ { e } B } \underset { k = 1 } { \overset { N _ { e } } { \sum } } L _ { \mathcal { T } _ { k } } \sigma _ { \mathcal { T } _ { k } } ^ { 2 } + } \\ & { \quad \frac { \eta ^ { 2 } \sigma _ { \operatorname* { m a x } } ^ { 2 } L _ { \operatorname* { m a x } } ^ { 2 } } { B } \Big ( \frac { 1 + \zeta ^ { 2 } } { 1 - \zeta ^ { 2 } } E _ { f } - 1 \Big ) . } \end{array}\tag{31}
$$

Proof: We present the proof of the theorem in the way similar to [62, Appendix D]. Consider the difference of the average loss

$$
\mathcal { L } \Big ( \overline { { { \pmb { \theta } } } } ^ { t + 1 } \Big ) - \mathcal { L } \Big ( \overline { { { \pmb { \theta } } } } ^ { t } \Big ) = \frac { 1 } { N _ { e } } \sum _ { k = 1 } ^ { N _ { e } } \Big [ \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { { \pmb { \theta } } } } _ { k } ^ { t + 1 } \Big ) - \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { { \pmb { \theta } } } } _ { k } ^ { t } \Big ) \Big ] .\tag{32}
$$

As an application of the L-smoothness gradient assumption, we obtain

$$
\ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \pmb { \theta } } } _ { k } ^ { t + 1 } \Big ) - \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \pmb { \theta } } } _ { k } ^ { t } \Big ) \leq \frac { \eta ^ { 2 } L _ { \mathcal { T } _ { k } } } { 2 } \left\| \tilde { \pmb { g } } _ { k } ^ { t } \right\| ^ { 2 } - \eta \Big \langle \nabla \ell _ { \mathcal { A } _ { k } } \Big ( \overline { { \pmb { \theta } } } _ { k } ^ { t } \Big ) , \tilde { \pmb { g } } _ { k } ^ { t } \Big \rangle .\tag{33}
$$

By assumption 5), the expected value of the second term can be bounded by

$$
\begin{array} { r l r } {  { E \Big [ \| \tilde { \pmb { g } } _ { k } ^ { t } \| ^ { 2 } \Big ] = E \Big [ \Big \| \tilde { \pmb { g } } _ { k } ^ { t } - \pmb { g } _ { k } ^ { t } \Big \| ^ { 2 } \Big ] + \Big \| \pmb { g } _ { k } ^ { t } \Big \| ^ { 2 } } } \\ & { } & { \leq ( C + 1 ) \Big \| \pmb { g } _ { k } ^ { t } \Big \| + \frac { \sigma _ { T _ { k } } ^ { 2 } } { B } . } \end{array}\tag{34}
$$

For the mean of the first term, by (28) in assumption 4), we have

$$
\begin{array} { r l } { E \Big [ - \eta \Big \langle \nabla \ell _ { A _ { k } } \big ( \overline { { \theta } } _ { k } ^ { t } \big ) , \bar { g } _ { k } ^ { t } \Big \rangle \Big ] = - \eta \Big \langle \nabla \ell _ { A _ { k } } ( \overline { { \theta } } _ { k } ^ { t } ) , E \big [ \bar { g } _ { k } ^ { t } \big ] \Big \rangle } & { } \\ { = - \eta \Big \langle \nabla \ell _ { A _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { t } \Big ) , g _ { k } ^ { t } \Big \rangle } & { } \\ { = \frac { - \eta } { 2 } \Bigg [ \Big \| \nabla \ell _ { A _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { t } \Big ) \Big \| ^ { 2 } + \big \| g _ { k } ^ { t } \big \| ^ { 2 } } & { } \\ { \quad \quad \quad \quad \quad \quad - \| \nabla \ell _ { A _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { t } \Big ) - g _ { k } ^ { t } \Big \| \Bigg ] } & { } \\ { \leq \frac { \eta } { 2 } \Bigg [ - \Big \| \nabla \ell _ { A _ { k } } \Big ( \overline { { \theta } } _ { k } ^ { t } \Big ) \Big \| ^ { 2 } - \big \| g _ { k } ^ { t } \big \| ^ { 2 } } & { } \\ { \quad \quad \quad \quad \quad + { \cal L } _ { T _ { k } } ^ { 2 } \Big \| \overline { { \theta } } _ { k } ^ { t } - \theta _ { k } ^ { t } \Big \| ^ { 2 } \Bigg ] . } & { } \end{array}\tag{35}
$$

According to [59, eq. (136), Appendix D.2.4], we get the average bound for the last term in (35) as

$$
\frac { 1 } { N _ { e } T } \sum _ { t , k } L _ { \mathcal { T } _ { k } } ^ { 2 } \left. \overline { { \theta } } _ { k } ^ { t } - \theta _ { k } ^ { t } \right. ^ { 2 } \leq \frac { \eta ^ { 2 } \sigma _ { \operatorname* { m a x } } ^ { 2 } L _ { \operatorname* { m a x } } ^ { 2 } } { B } \bigg ( \frac { 1 + \zeta ^ { 2 } } { 1 - \zeta ^ { 2 } } E _ { f } - 1 \bigg )
$$

$$
\begin{array} { r } { + \left[ \frac { \eta ^ { 2 } E _ { f } ^ { 2 } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \zeta } \bigg ( \frac { 2 \zeta } { 1 + \zeta } + \frac { 2 \zeta } { 1 - \zeta } + \frac { E _ { f } - 1 } { E _ { f } } \bigg ) \right. } \\ { + \left. \frac { 2 \eta ^ { 2 } C E _ { f } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \zeta ^ { 2 } } \right] \frac { 1 } { N _ { e } T } \sum _ { t } \sum _ { k } \| g _ { k } ^ { t } \| ^ { 2 } . } \end{array}\tag{36}
$$

Then, by taking the average on both sides of (32) and combining (33)–(36), we obtain

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = T _ { 0 } + 1 } ^ { T _ { 0 } + T } E \Big [ \mathcal { L } \big ( \overline { { \theta } } ^ { t + 1 } \big ) - \mathcal { L } \big ( \overline { { \theta } } ^ { t } \big ) \Big ] \leq \frac { 1 } { N _ { e } T } \sum _ { t } \sum _ { k } \frac { \eta ^ { 2 } L _ { T _ { k } } } { 2 } \| \tilde { g } _ { k } ^ { t } \| ^ { 2 } } \\ & { \quad \quad + \frac { 1 } { N _ { e } T } \displaystyle \sum _ { t } \sum _ { k } - \eta \Big \langle \nabla \ell _ { A _ { k } } \big ( \overline { { \theta } } _ { k } ^ { t } \big ) , \tilde { g } _ { k } ^ { t } \Big \rangle } \\ & { \leq \frac { - \eta } { 2 N _ { e } T } \displaystyle \sum _ { t } \sum _ { k } \Big \| \nabla \ell _ { A _ { k } } \big ( \overline { { \theta } } _ { k } ^ { t } \big ) \Big \| ^ { 2 } + \mathbf { r } \cdot \frac { \eta } { 2 N _ { e } T } \displaystyle \sum _ { t } \sum _ { k } \| g _ { k } ^ { t } \| ^ { 2 } } \\ & { \quad \quad + \frac { \eta ^ { 3 } \sigma _ { \operatorname* { m a x } } ^ { 2 } L _ { \operatorname* { m a x } } ^ { 2 } } { 2 B } \Big ( \frac { 1 + \zeta ^ { 2 } } { 1 - \zeta ^ { 2 } } E _ { f } - 1 \Big ) + \frac { \eta ^ { 2 } } { 2 N _ { e } B } \displaystyle \sum _ { k } L _ { T _ { k } } \sigma _ { T _ { k } } ^ { 2 } } \end{array}\tag{37}
$$

where  is the left-hand side of (30)

$$
\begin{array} { l } { \Gamma = \displaystyle { \bigg [ \frac { 2 C E _ { f } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \zeta ^ { 2 } } + \frac { E _ { f } ^ { 2 } L _ { \mathrm { m a x } } ^ { 2 } } { 1 - \zeta } \bigg ( \frac { 4 \zeta } { 1 - \zeta ^ { 2 } } + \frac { E _ { f } - 1 } { E _ { f } } \bigg ) \bigg ] \eta ^ { 2 } } } \\ { + L _ { \mathrm { m a x } } ( C + 1 ) \eta - 1 . } \end{array}\tag{38}
$$

Particularly, if the learning rate η is set properly to make $\Gamma \leq 0 .$ the terms related with $\| \pmb { g } _ { k } ^ { t } \| ^ { 2 }$ can be eliminated and the conclusion (31) can be obtained.

Further, we summarize the following remarks to interpret some insights observed from Theorem 1.

Remark 1 (Convergence of Gradients): The theorem investigates the learning convergence from the perspective of gradients. When the average 2-norm of $\ell _ { \mathcal { A } _ { k } } ( \overline { { \pmb { \theta } } } _ { k } ) \mathrm { { ' s } }$ gradients are upper bounded, one can deem that the loss functions are stable and the learning process converges.

Remark 2 (Generalization to Other Metrics): Note that in the above theorem, we consider the actor loss without regard to the specific metric and the goal is to bound the deviation of the gradients. Therefore, if one hopes to maintain any metric (or metric combination) which converges to certain value, the theorem always holds.

Remark 3 (Interpretation of the Bounds): In the theorem, the right-hand side upper bound $\Lambda ( \omega , E _ { f } , \mathcal { T } _ { k } ^ { T _ { 0 } } )$ can be interpreted by the following three separated terms:

$$
\begin{array} { r l } { \Lambda \Bigl ( \omega , E _ { f } , T _ { k } ^ { T _ { 0 } } \Bigr ) = \underbrace { \frac { 2 \sum _ { k = 1 } ^ { N _ { e } } \Bigl [ \ell _ { A _ { k } } \Bigl ( \overline { { \theta } } _ { k } ^ { T _ { 0 } } \Bigr ) - \ell _ { A _ { k } } ^ { * } \Bigr ] } { \eta N _ { e } T } } _ { \mathrm { l n i t i a l ~ o v e r ~ a l u i t i o n } } } & { } \\ { + \underbrace { \frac { \eta \sum _ { k = 1 } ^ { N _ { e } } L _ { T _ { k } } \sigma _ { T _ { k } } ^ { 2 } } { N _ { e } B } } _ { \mathrm { S e u t w o ~ L e n s i t i o n } } } & { } \\ { + \underbrace { \frac { \eta ^ { 2 } \sigma _ { m a x } ^ { 2 } L _ { m a x } ^ { 2 } } { B } \Bigl ( \frac { 1 + \xi ^ { 2 } } { 1 - \xi ^ { 2 } } E _ { f } - 1 \Bigr ) } _ { \mathrm { F o l e c a t e d ~ U p d a t i o n } } . } \end{array}\tag{39}
$$

<!-- image-->  
Fig. 4. Sketch for the impact of ω.

The first term is the initial deviation caused by the training before $T _ { 0 }$ . The second term, related to the interaction after $T _ { 0 }$ is called the sequel deviation. The third term is the gradients’ deviation when the edge federated updating mode is carried.

Remark 4 (Minimum Bounds): For the federated update matrix in (24),  is positive-definite and symmetrical which has a 1-order eigenvalue $\Lambda _ { 1 } = 1$ and $N _ { e } - 1$ order repeated eigenvalue, $\Lambda _ { 2 } ~ = ~ \cdots ~ = ~ \Lambda _ { N _ { e } - 1 } ~ = ~ [ ( N _ { e } \omega - 1 ) / ( N _ { e } - 1 ) ]$ Then, $\zeta ~ = ~ [ ( N _ { e } \omega - 1 ) / ( N _ { e } - 1 ) ] ~ \in ~ [ ( - 1 / [ N _ { e } - 1 ] ) , 1 ]$ Besides, one can rewrite the last term of the right-hand side in (31) as

$$
\begin{array} { r l } {  { \frac { \eta ^ { 2 } \sigma _ { \mathrm { m a x } } ^ { 2 } L _ { \mathrm { m a x } } ^ { 2 } } { B } \bigg ( \frac { 1 + \zeta ^ { 2 } } { 1 - \zeta ^ { 2 } } E _ { f } - 1 \bigg ) } \quad } & { } \\ & { = \frac { \eta ^ { 2 } \sigma _ { \mathrm { m a x } } ^ { 2 } L _ { \mathrm { m a x } } ^ { 2 } } { B } \bigg ( \frac { 2 } { 1 - \zeta ^ { 2 } } E _ { f } - E _ { f } - 1 \bigg ) . } \end{array}\tag{40}
$$

Thus, with other hyper parameters fixed, the upper bound of the gradients’ 2-norm in (31) meets its minimum if $\zeta ^ { * } = 0 ,$ i.e., $\omega ^ { * } = ( 1 / N _ { e } )$ . This implies that when the edge-federated updating is carried out with uniform weights, namely, all elements in  are equal to $( 1 / N _ { e } )$ , the gradients’ mean-square time average is bounded tightly, which leads to best training convergence.

Remark 5 (Impact of ω): Intuitively, if one sets $\omega < ( 1 / N _ { e } )$ the diagonal elements of  in (24) are smaller than the others. We regard these cases as chaos because each agent keeps less knowledge of the policies learned from its own observations. Also, in particular, when $\omega = 1$ ,  is an identity matrix which refers to the original mixed H-MAAC, where each agent learns the policies individually and no parameter sharing occurs. In this case, $\zeta ~ = ~ 1$ and the right-hand side of (31) becomes infinity, which means that the gradients are unbounded, and therefore, the convergence of original H-MAAC is not guaranteed under the above deductions. Thus, the effective interval of ω lies on $[ ( 1 / N _ { e } ) , 1 )$ . Overall, we elucidate the impact of ω as the following Fig. 4. When $\omega  ( 1 / N _ { e } )$ , the federated parameter sharing becomes uniform and the edge agents tend to be homogeneous. When $\omega  1$ , the diagonal elements of  dominate and each edge agents keeps more individuality by preserving the most of its own policies.

Remark 6 (Convergence Versus Efficiency): Note that in assumption 3)–5), we require the preconditions that the interaction sets are given and fixed. Nevertheless, in the online MEC collaboration, the environment is stochastic where the dynamics of data arrival and 
-exploration are time variant. In addition, the random sampling in experience replay may also lead to variant training sets, $\{ \mathcal { T } _ { k } \} ^ { t }$ . For reasons above, the optimal critic parameters $\phi _ { k } ^ { * }$ , the Lipschitz constant $L \tau _ { k }$ in (28), the gradient variance $\sigma \tau _ { k }$ as well as the coefficient C in (29) are likely to be unstable and of large variance at different learning epochs. Furthermore, to the contrary, the online collaboration system might benefit from the fluctuation of the gradients because the actor parameters θ can be modified to adapt the change of the environment. Thus, the best training convergence in the ideal scenarios may not always guarantee the best system performance. We reckon this as a learning convergence versus system efficiency tradeoff in practice and the details will be discussed in Section V.

<!-- image-->  
(a)

<!-- image-->  
（c）

<!-- image-->  
(b)

<!-- image-->  
(d)  
Fig. 5. Comparison of several RL-based MEC collaboration methods. (4 edge servers, 30 data sources on a 200 × 200 map.). (a) Evolution of MEC system’s average age, (t). (b) Worst AoI of all data sources, i.e., {max $\Delta _ { k } ( t ) \}$ . (c) Data volume received at cloud center. (d) Count of the data packets received by cloud center.

## V. PERFORMANCE EVALUATION

In this section, we will test the proposed MEC collaboration framework and evaluate its performance through simulation. Besides, the comparisons with popular RL approaches and other insights are investigated.

## A. Simulation Settings

First, we implement the age sensitive MEC system presented in Section II as a universal gym [63] module in Python. For simplicity of the following discussion and simulation, we suppose that the packets independently arrive at each data source $S _ { n }$ with the probability $p _ { g } ^ { n }$ at the beginning of each time slot and the data size follows a Poisson distributed with the arrival rate, $\lambda _ { n }$ [64]. Thus, the data generation process can be described as a switch Poisson process

$$
P \big \{ d _ { n , i } ( t ) = m \big \} = \mathbb { 1 } _ { G _ { n } } \cdot \frac { e ^ { - \lambda _ { n } } \lambda _ { n } ^ { m } } { m ! }\tag{41}
$$

<!-- image-->  
(a)

<!-- image-->

<!-- image-->  
（c）

<!-- image-->  
(d)  
Fig. 6. Training loss of all agents’ actor–critic nets. (a) Edge agent actor loss. (b) Edge agent critic loss. (c) Center agent actor loss. (d) Center agent critic loss.

where the boolean random variable $G _ { n } \sim$ Bernoulli $( p _ { g } ^ { n } )$ , indicates the arrival of the packets.

Second, let us claim some basic environment settings of simulations. The main parameter settings are listed in Table II.

TABLE II  
MAIN PARAMETER SETTINGS FOR SIMULATIONS
<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td> $\overline { { ( \lambda _ { n } , p _ { g } ^ { n } ) } }$ </td><td>Datarate and arrival probability of data generation at  ${ \overline { { S _ { n } } } } .$ </td><td>(1Kb/slot, 0.3)</td></tr><tr><td> $( r _ { m o v e } ^ { k } , r _ { o b s } ^ { k } , r _ { c o l l e c t } ^ { k } )$ </td><td>Edge devices’radius of movement,observation,collection.</td><td>(6,60,40)</td></tr><tr><td> $B _ { c o l } ^ { k } , B _ { e x t } ^ { k }$  0</td><td>Maximum buffer size for collected data and executed data.</td><td>5</td></tr><tr><td> $\smash { \dot { f } _ { c } ^ { k } }$ </td><td>Computation rate of of  $\operatorname { E } _ { k } .$ </td><td>20Kb/slot</td></tr><tr><td> $\breve { W }$ </td><td>Total bandwidth for offloading communication.</td><td>100MHz</td></tr><tr><td> $f$ </td><td>The carrier frequency in Eq.(2.8).</td><td>2.5GHz</td></tr><tr><td> $( a , b , \eta _ { L o S } , \eta _ { N L o S } )$ </td><td>Coefficients of A2G path loss.</td><td>(9.61,0.16,1, 20)</td></tr><tr><td> $N _ { 0 }$ </td><td>The noise power spectral density of A2G channel.</td><td>-130dB</td></tr><tr><td> $P _ { t r , m a x } ^ { k }$ </td><td>Maximum power for offloading communication.</td><td>0.2W</td></tr><tr><td> $\gamma$ </td><td>Penalty decay.</td><td>0.85</td></tr><tr><td> $\tau$ </td><td>Target updating weight.</td><td>0.8</td></tr><tr><td>E</td><td>Probability of random exploration.</td><td>0.2</td></tr><tr><td> $T _ { u } , E _ { f }$ </td><td>Period for target updating and federated updating.</td><td>8</td></tr><tr><td> $( \eta _ { A } , \eta _ { C } )$ </td><td>Learning rates for actor/critic nets.</td><td></td></tr><tr><td> $B$ </td><td>Batch size of experience replay.</td><td> $( 1 \times 1 0 ^ { - 3 } , 2 \times 1 0 ^ { - 3 } )$ </td></tr></table>

As for data sources, we set the arrival probability and the generating rate as 0.3 and 1 kb/slot. Then, for the attributes of edge devices, we set $r _ { \mathrm { m o v e } } ^ { k } , r _ { \mathrm { o b s } } ^ { k } , r _ { \mathrm { c o l l e c t } } ^ { k }$ to be 6, 60, 40 in measure of the grid map and the height is fixed. The computing rate of edge process is 20 kb/slot and the maximum buffer lengths for caching collected data and executed data are both set to be 5 packet pieces. For edge-cloud communication channel, we set the total offloading bandwidth as 100 MHz, the carrier frequency f in (8) as 2.5 GHz, the noise power spectral density $N _ { 0 } ~ \mathrm { a s } \ - 1 3 0$ dB, and the max transmission power $P _ { \mathrm { t r } } ^ { k }$ r,max as 0.2 W. The coefficients $( a , b , \eta _ { L o S } , \eta _ { N L o S } )$ in (8) and (10) are selected to be (9.61, 0.16, 1, 20), which refers to the urban scenarios mentioned in [47]. Particularly, the transmission rate of edge-source collection is fixed to be 8 kb/slot due to the limit of collection cover.

With regard to learning configurations, the decay coefficient $\gamma$ of the system penalty is set to be 0.85. As for hyper parameters, the learning rates of actor and critic are 1e-3, 2e-3, respectively, and the batch size of each epoch is 128. As proposed in Section III, the target updating period $T _ { u }$ and the reserving weight τ are 8 and 0.8. In addition, we exploit the -exploration with $\epsilon = 0 . 2$ and edge-federated mode where parameters are shared every 8 learning epochs.

As for the metrics, we also adopt peak AoI (PAoI) and worst AoI in comparisons. PAoI of data source $S _ { n } , \Delta _ { p , n } ,$ is defined as the average peak value of $S _ { n } \mathrm { \ ' } _ { \mathrm { s } }$ AoIs, which represents the maximum AoI before a new update is received [30]. We denote

$$
\overline { { \Delta } } _ { p } = \frac { 1 } { N _ { s } } \sum _ { n } \Delta _ { p , n }\tag{42}
$$

as the average PAoI for all data sources. Besides, the worst AoI, defined as the maximum AoI of data sources at each time slot, is considered to evaluate the AoI performance in worst cases.

## B. Evaluation Results

We implement EdgeFed H-MAAC collaboration algorithm, by building the CNN-MLPs mixed networks for edge actor– critic nets and MLP-based networks as center agent with TensorFlow. To compare the performance of the proposed frameworks with the other RL approaches, we select two actor–critic-based algorithms, DDPG (centralized) and the popular MADDPG (multiagent) as the baselines where dual neural networks are also exploited to learn the mixed policies. For fairness, we set the same random seeds of the MEC environment for all methods and instead of spending much effort on network tuning, we also fix the random seeds of training processes. Consequently, the results are of generality and can be easily reproduced.

<!-- image-->  
Fig. 7. EdgeFed H-MAAC’s (t) performance under different environment settings on a 300 × 300 map.  
TABLE III

NUMERICAL COMPARISON ON AOI METRICS
<table><tr><td rowspan="2">Approach</td><td colspan="2">Average age  $\overline { { \Delta ( t ) } }$ </td><td colspan="3">Peak AoI</td></tr><tr><td>mean</td><td>std</td><td>peak count</td><td> ${ \overline { { \Delta } } } _ { p }$ </td><td> $v a r ( \Delta _ { p } )$ </td></tr><tr><td>Mixed DDPG</td><td>473.3</td><td>174.4</td><td>10121</td><td>26.6</td><td>156.0</td></tr><tr><td>Mixed MADDPG</td><td>39.3</td><td>16.7</td><td>12842</td><td>24.9</td><td>93.0</td></tr><tr><td>EdgeFed H-MAAC (proposed)</td><td>21.2</td><td>3.6</td><td>13782</td><td>21.7</td><td>23.1</td></tr></table>

Fig. 5 and Table III show the comparison results of proposed algorithm (set $\omega = 0 . 5 )$ and the baselines under the environment with 4 edge devices and 30 data sources on a $2 0 0 \times 2 0 0$ map where the positions of edge devices and data sources are initialized randomly. The average age $\overline { { \Delta } } ( t )$ during the online interaction is shown in Fig. 5(a) where one can find that the mixed DDPG results in highest $\overline { { \Delta } } ( t )$ and the curve is not stable till 5K epoch. However, under mixed MADDPG and EdgeFed H-MAAC, the average age maintains at a lower value after sufficient iterations. Specifically, the statistics of $\overline { { \Delta } } ( t )$ are listed in Table III. EdgeFed H-MAAC attains not only the lowest average age, but also the lowest variance, which means the edge-federated approach outperforms DDPG as well as MADDPG on both system penalty and learning stability. The right-hand part of Table III presents the comparison on PAoI. Evidently, the proposed collaboration algorithm reaps most peak updates and lowest average PAoI, ${ \overline { { \Delta } } } _ { p }$ . This implies that the efficiency of data processing in MEC can be promoted by EdgeFed H-MAAC. The lowest variance of PAoI also demonstrate that all data sources are updated frequently and fairly. Fig. 5(b) displays the worst AoI of three approaches. EdgeFed H-MAAC also performs the best. One can find that under the centralized DDPG, some sources are ignored for a long time. We explain this as the fact that the centralized collaboration algorithms require larger neural network models with complex structure to extract the relations between the excessive global input states and the local policies of each individual agent, which also leads to the difficulties for training. Besides, Fig. 5(c) and (d) presents the volume and the count of the aggregated packets received at the cloud center, namely, the final hop of the MEC system. The curves illustrate that EdgeFed H-MAAC collaboration algorithm also improves the data utility of the MEC system by finishing more data processing within same time.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 8. EdgeFed H-MAAC performance with different ω. (8 edge devices, 60 data sources on a 300×300 map). (a) (t) performance under different ω. The vertical doted lines represent the convergence time of each ω. (b) Box plots of (t) under different ω.

Additionally, we present the training loss of mixed MADDPG and EdgeFed H-MAAC in Fig. 6 where the actor loss and critic loss of the first edge agent and the center agent are displayed. Similarly, EdgeFed H-MAAC leads to lower and more stable loss. What is more, the critic loss shown in Fig. 6(b) and (d) demonstrates that it is acceptable to assume that the center agent training can be out of consideration and the critic nets fit the Q values well, which refers to the assumption 2) and 3) of the convergence discussion in Section IV. Beyond our expectation, although the federated parameter sharing only works on edge agents, this scheme also promotes the center agent training significantly.

Then, we investigate the performance under different environment settings. We change the edge number $N _ { e }$ as well as the source number $N _ { s }$ and evaluate the EdgeFed H-MAAC in these cases. As presented in Fig. 7, low average ages are well maintained under different environment settings through EdgeFed H-MAAC collaboration. Consistent with the common sense, the more edge servers or fewer data sources both lead to better timeliness of the MEC systems.

Furthermore, to investigate the impact of the federated factor ω, we set different ω in the scene with 8 edge devices and 60 data sources on a 300 × 300 map. Likewise, the settings and the randomness of the MEC environment are identical for all the simulations. Note that the federated factor ω denote the weight with which each agent retains its model during the parameter sharing, agents will lose their own parameters if ω becomes too small. Hence, we only explore the cases where $\omega \geq ( 1 / N _ { e } )$ . The results are presented as $\overline { { \Delta } } ( t )$ evolution curves in Fig. 8(a) and box plots of $\overline { { \Delta } } ( t )$ in Fig. 8(b). Intuitively, one can find that while $\omega = 1 , \mathrm { i . e . }$ ., the original H-MAAC, the system gets the worst performance and the learning process seems not to converge after 5K iterations, which is consistent with our theoretical analysis in Remark 5. Additionally, from Theorem 1 and Remark 4, the deviation of the gradients gets minimum when $\omega ^ { * } = 0 . 1 2 5$ . This is verified by the experiment where the vertical doted lines imply that the average age gets the rapidest convergence when $\omega = \omega ^ { * }$ and the larger ω leads to longer convergence time. However, the simulation outcomes show that $\omega = 0 . 2 5$ and $\omega = 0 . 5$ also perform well as they reach low average ages and even lower variances with similarly rapid convergence. As discussed in Remark 6, due to the dynamics of the environment and the randomness of the online learning, the constants in (28) and (29) are time variant. Besides, the fine-fitness of edge critic nets are not exactly guaranteed, as shown in Fig. 6(b) where the edge critic loss of EdgeFed H-MAAC does not strictly decrease to 0. For above reasons, one can find the gap between the convergence theorem and the simulation results. We explain such phenomenon as a tradeoff between the ideal learning convergence and the system robustness to environment variation. In practice, the fluctuation of the gradients may contribute to learning the features of the stochastic environment. Meanwhile, smaller ω implies that the model parameters of each edge agent itself are preserved with lower proportion. Particularly, if $\omega = \omega ^ { * }$ , all edge agents learn the same parameters after federated updating operation. Therefore, all edge agents tend to make same responses to the input states and lose their individuality small ω. For larger federated factor $\omega ,$ , though the gradients are not bounded tightly, it reserves the individuality of each edge agents to counter the stochastic environment in MEC collaboration systems. Thus, as elaborated in Fig. 4, in such multiagent cooperative learning framework, ω with mediate values may balance the learning convergence and the system efficiency. In addition, the simulation results in Fig. 8 can be utilized to design the proper edge-FL mode for the proposed multiagent collaboration MEC algorithm. Approximately, through the above results and the discussions, the recommended interval of the federated factor ω lies on $[ ( 1 / N _ { e } ) , 0 . 5 ]$

## VI. CONCLUSION

We investigated the age sensitive MEC systems and proposed a policy-based MARL framework, H-MAAC, for agent intelligent control of the trajectory planning, data scheduling as well as bandwidth allocation. By adopting FL mode, we developed the corresponding edge federated online joint collaboration algorithms whose convergence were theoretically proved. We implemented the MEC simulation system and evaluated the proposed algorithms. The outcomes showed that our method has lower average age and better learning stability compared to classical centralized actor–critic RL approaches. Moreover, some other advantages and the inspirations for edge federated design are also discussed according to the simulation results.

For further work, based on the proposed H-MAAC framework, more operations of agents can be expanded, such as the power allocation, multitask scheduling and multitask offloading. In addition, the penalty/reward of the system can be flexibly defined which may bring more applications for cooperative MEC systems.

## REFERENCES

[1] W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, “Edge computing: Vision and challenges,” IEEE Internet Things J., vol. 3, no. 5, pp. 637–646, Oct. 2016.

[2] X. Sun and N. Ansari, “EdgeIoT: Mobile edge computing for the Internet of Things,” IEEE Commun. Mag., vol. 54, no. 12, pp. 22–29, Dec. 2016.

[3] Y. C. Hu, M. Patel, D. Sabella, N. Sprecher, and V. Young, “Mobile edge computing—A key technology towards 5G,” Sophia Antipolis, France, ETSI, White Paper, 2015.

[4] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, “A survey on mobile edge computing: The communication perspective,” IEEE Commun. Surveys Tuts., vol. 19, no. 4, pp. 2322–2358, 4th Quart., 2017.

[5] M. A. Abd-Elmagid, N. Pappas, and H. S. Dhillon, “On the role of age of information in the Internet of Things,” IEEE Commun. Mag., vol. 57, no. 12, pp. 72–77, Dec. 2019.

[6] N. Abbas, Y. Zhang, A. Taherkordi, and T. Skeie, “Mobile edge computing: A survey,” IEEE Internet Things J., vol. 5, no. 1, pp. 450–465, Feb. 2018.

[7] B. Liu, C. Liu, and M. Peng, “Resource allocation for energy-efficient MEC in NOMA-enabled massive IoT networks,” IEEE J. Sel. Areas Commun., vol. 39, no. 4, pp. 1015–1027, Apr. 2020.

[8] Y. Du, K. Wang, K. Yang, and G. Zhang, “Energy-efficient resource allocation in UAV based MEC system for IoT devices,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), 2018, pp. 1–6.

[9] X. Hu, K.-K. Wong, K. Yang, and Z. Zheng, “UAV-assisted relaying and edge computing: Scheduling and trajectory optimization,” IEEE Trans. Wireless Commun., vol. 18, no. 10, pp. 4738–4752, Oct. 2019.

[10] P. Mach and Z. Becvar, “Mobile edge computing: A survey on architecture and computation offloading,” IEEE Commun. Surveys Tuts., vol. 19, no. 3, pp. 1628–1656, 3rd Quart., 2017.

[11] Z. Zhao et al., “A novel framework of three-hierarchical offloading optimization for MEC in industrial IoT networks,” IEEE Trans. Ind. Informat., vol. 16, no. 8, pp. 5424–5434, Aug. 2020.

[12] J. Zhao, Q. Li, Y. Gong, and K. Zhang, “Computation offloading and resource allocation for cloud assisted mobile edge computing in vehicular networks,” IEEE Trans. Veh. Technol., vol. 68, no. 8, pp. 7944–7956, Aug. 2019.

[13] L. Lovén et al., “EdgeAI: A vision for distributed, edgenative artificial intelligence in future 6G networks,” in Proc. 1st 6G Wireless Summit, 2019, pp. 1–2.

[14] V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[15] D. Ye et al., “Mastering complex control in MOBA games with deep reinforcement learning,” in Proc. AAAI, 2020, pp. 6672–6679.

[16] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. Cambridge, MA, USA: MIT Press, 2018.

[17] T. P. Lillicrap et al., “Continuous control with deep reinforcement learning,” 2015. [Online]. Available: arXiv:1509.02971.

[18] L. Busoniu, R. Babuska, and B. De Schutter, “A comprehensive survey of multiagent reinforcement learning,” IEEE Trans. Syst., Man, Cybern. C, Appl. Rev., vol. 38, no. 2, pp. 156–172, Mar. 2008.

[19] H. H. Zhuo, W. Feng, Q. Xu, Q. Yang, and Y. Lin, “Federated reinforcement learning,” 2019. [Online]. Available: arXiv:1901.08277.

[20] A. Ndikumana et al., “Joint communication, computation, caching, and control in big data multi-access edge computing,” IEEE Trans. Mobile Comput., vol. 19, no. 6, pp. 1359–1374, Jun. 2020.

[21] A. Merwaday and I. Guvenc, “UAV assisted heterogeneous networks for public safety communications,” in Proc. IEEE Wireless Commun. Netw. Conf. Workshops (WCNCW), 2015, pp. 329–334.

[22] V. Sharma, M. Bennis, and R. Kumar, “UAV-assisted heterogeneous networks for capacity enhancement,” IEEE Commun. Lett., vol. 20, no. 6, pp. 1207–1210, Jun. 2016.

[23] M. Emara, H. El Sawy, M. C. Filippou, and G. Bauch, “Spatiotemporal dependable task execution services in MEC-enabled wireless systems,” IEEE Wireless Commun. Lett., vol. 10, no. 2, pp. 211–215, Feb. 2021.

[24] B. Cao, L. Zhang, Y. Li, D. Feng, and W. Cao, “Intelligent offloading in multi-access edge computing: A state-of-the-art review and framework,” IEEE Commun. Mag., vol. 57, no. 3, pp. 56–62, Mar. 2019.

[25] Z. Ning et al., “Partial computation offloading and adaptive task scheduling for 5G-enabled vehicular networks,” IEEE Trans. Mobile Comput., early access, Sep. 18, 2020, doi: 10.1109/TMC.2020.3025116.

[26] D. W. Matolak and R. Sun, “Unmanned aircraft systems: Air-ground channel characterization for future applications,” IEEE Veh. Technol. Mag., vol. 10, no. 2, pp. 79–85, Jun. 2015.

[27] F. Zhou, Y. Wu, H. Sun, and Z. Chu, “UAV-enabled mobile edge computing: Offloading optimization and trajectory design,” in Proc. IEEE Int. Conf. Commun. (ICC), 2018, pp. 1–6.

[28] Y. Liu, K. Xiong, Q. Ni, P. Fan, and K. B. Letaief, “UAV-assisted wireless powered cooperative mobile edge computing: Joint offloading, CPU control, and trajectory optimization,” IEEE Internet Things J., vol. 7, no. 4, pp. 2777–2790, Apr. 2020.

[29] C.-F. Liu, M. Bennis, and H. V. Poor, “Latency and reliability-aware task offloading and resource allocation for mobile edge computing,” in Proc. IEEE Globecom Workshops (GC Wkshps), 2017, pp. 1–7.

[30] M. Costa, M. Codreanu, and A. Ephremides, “Age of information with packet management,” in Proc. IEEE Int. Symp. Inf. Theory, 2014, pp. 1583–1587.

[31] X. Wang, Z. Ning, S. Guo, M. Wen, and V. Poor, “Minimizing the ageof-critical-information: An imitation learning-based scheduling approach under partial observations,” IEEE Trans. Mobile Comput., early access, Jan. 21, 2021, doi: 10.1109/TMC.2021.3053136.

[32] X. Chen et al., “Age of information aware radio resource management in vehicular networks: A proactive deep reinforcement learning perspective,” IEEE Trans. Wireless Commun., vol. 19, no. 4, pp. 2268–2281, Apr. 2020.

[33] J. Liu, X. Wang, B. Bai, and H. Dai, “Age-optimal trajectory planning for UAV-assisted data collection,” in Proc. IEEE INFOCOM Conf. Comput. Commun. Workshops (INFOCOM WKSHPS), 2018, pp. 553–558.

[34] H. Hu, K. Xiong, G. Qu, Q. Ni, P. Fan, and K. B. Letaief, “AoI-minimal trajectory planning and data collection in UAV-assisted wireless powered IoT networks,” IEEE Internet Things J., vol. 8, no. 2, pp. 1211–1223, Jan. 2021.

[35] P. Tong, J. Liu, X. Wang, B. Bai, and H. Dai, “Deep reinforcement learning for efficient data collection in UAV-aided Internet of Things,” in Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops), 2020, pp. 1–6.

[36] J. Wang, L. Zhao, J. Liu, and N. Kato, “Smart resource allocation for mobile edge computing: A deep reinforcement learning approach,” IEEE Trans. Emerg. Topics Comput., early access, Mar. 4, 2019, doi: 10.1109/TETC.2019.2902661.

[37] J. Li, H. Gao, T. Lv, and Y. Lu, “Deep reinforcement learning based computation offloading and resource allocation for MEC,” in Proc. IEEE Wireless Commun. Netw. Conf. (WCNC), 2018, pp. 1–6.

[38] S. Wan, J. Lu, P. Fan, and K. B. Letaief, “Towards big data processing in Iot: Path planning and resource management of UAV base stations in mobile-edge computing system,” IEEE Internet Things J., vol. 7, no. 7, pp. 5995–6009, Jul. 2020.

[39] X. Chen, H. Zhang, C. Wu, S. Mao, Y. Ji, and M. Bennis, “Optimized computation offloading performance in virtual edge computing systems via deep reinforcement learning,” IEEE Internet Things J., vol. 6, no. 3, pp. 4005–4018, Jun. 2019.

[40] X. Wang, Z. Ning, and S. Guo, “Multi-agent imitation learning for pervasive edge computing: A decentralized computation offloading algorithm,” IEEE Trans. Parallel Distrib. Syst., vol. 32, no. 2, pp. 411–425, Feb. 2021.

[41] H. Peng and X. Shen, “Multi-agent reinforcement learning based resource management in MEC-and UAV-assisted vehicular networks,” IEEE J. Sel. Areas Commun., vol. 39, no. 1, pp. 131–141, Jan. 2021.

[42] L. Wang, K. Wang, C. Pan, W. Xu, N. Aslam, and L. Hanzo, “Multiagent deep reinforcement learning based trajectory planning for multi-UAV assisted mobile edge computing,” IEEE Trans. Cogn. Commun. Netw., vol. 7, no. 1, pp. 73–84, Mar. 2021.

[43] Y. Zhang, Z. Mou, F. Gao, J. Jiang, R. Ding, and Z. Han, “UAV-enabled secure communications by multi-agent deep reinforcement learning,” IEEE Trans. Veh. Technol., vol. 69, no. 10, pp. 11599–11611, Oct. 2020.

[44] S. Kumar, P. Shah, D. Hakkani-Tur, and L. Heck, “Federated control with hierarchical multi-agent deep reinforcement learning,” 2017. [Online]. Available: arXiv:1712.08266.

[45] X. Wang, C. Wang, X. Li, V. C. M. Leung, and T. Taleb, “Federated deep reinforcement learning for Internet of Things with decentralized cooperative edge caching,” IEEE Internet Things J., vol. 7, no. 10, pp. 9441–9455, Oct. 2020.

[46] J. Liu, Y. Mao, J. Zhang, and K. B. Letaief, “Delay-optimal computation task scheduling for mobile-edge computing systems,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), 2016, pp. 1451–1455.

[47] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP altitude for maximum coverage,” IEEE Wireless Commun. Lett., vol. 3, no. 6, pp. 569–572, Dec. 2014.

[48] A. Kosta, N. Pappas, and V. Angelakis, “Age of information: A new concept, metric, and tool,” Found. Trends Netw., vol. 12, no. 3, pp. 162–259, 2017.

[49] M. Costa, M. Codreanu, and A. Ephremides, “On the age of information in status update systems with packet management,” IEEE Trans. Inf. Theory, vol. 62, no. 4, pp. 1897–1910, Apr. 2016.

[50] M. Wiering and M. Van Otterlo, Reinforcement Learning, vol. 12. Heidelberg, Germany: Springer, 2012,

[51] M. Van Otterlo and M. Wiering, “Reinforcement learning and Markov decision processes,” in Reinforcement Learning. Heidelberg, Germany: Springer, 2012, pp. 3–42.

[52] M. L. Littman, “Markov games as a framework for multi-agent reinforcement learning,” in Proc. Mach. learn., 1994, pp. 157–163.

[53] L. Panait and S. Luke, “Cooperative multi-agent learning: The state of the art,” Auton. Agents Multi Agent Syst., vol. 11, no. 3, pp. 387–434, 2005.

[54] M. L. Puterman, Markov Decision Processes: Discrete Stochastic Dynamic Programming. New York, NY, USA: Wiley, 2014.

[55] R. Lowe, Y. I. Wu, A. Tamar, J. Harb, O. P. Abbeel, and I. Mordatch, “Multi-agent actor-critic for mixed cooperative-competitive environments,” in Proc. Adv. Neural Inf. Process. Syst., 2017, pp. 6379–6390.

[56] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath, “A brief survey of deep reinforcement learning,” 2017. [Online]. Available: arXiv:1708.05866.

[57] Q. Yang, Y. Liu, T. Chen, and Y. Tong, “Federated machine learning: Concept and applications,” ACM Trans. Intell. Syst. Technol., vol. 10, no. 2, pp. 1–19, 2019.

[58] J. Konecnˇ y, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh, and \` D. Bacon, “Federated learning: Strategies for improving communication efficiency,” 2016. [Online]. Available: arXiv:1610.05492.

[59] J. Wang and G. Joshi, “Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms,” 2018. [Online]. Available: arXiv:1808.07576.

[60] R. Balan, M. Singh, and D. Zou, “Lipschitz properties for deep convolutional networks,” 2017. [Online]. Available: arXiv:1701.05217.

[61] F. Latorre, P. Rolland, and V. Cevher, “Lipschitz constant estimation of neural networks via sparse polynomial optimization,” 2020. [Online]. Available: arXiv:2004.08688.

[62] F. Haddadpour and M. Mahdavi, “On the convergence of local descent methods in federated learning,” 2019. [Online]. Available: arXiv:1910.14425.

[63] G. Brockman et al., “OpenAI gym,” 2016. [Online]. Available: arXiv:1606.01540.

[64] Q. Fan and N. Ansari, “Application aware workload allocation for edge computing-based IoT,” IEEE Internet Things J., vol. 5, no. 3, pp. 2146–2153, Jun. 2018.