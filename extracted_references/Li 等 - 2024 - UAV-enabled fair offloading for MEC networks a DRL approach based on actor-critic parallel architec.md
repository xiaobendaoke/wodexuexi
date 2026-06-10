<!-- image-->

# UAV-enabled fair offloading for MEC networks: a DRL approach based on actor-critic parallel architecture

Wei Li1,2 .Si Li1,2 . Huaguang Shi1,2 .Wenhao Yan1,2 .Yi Zhou1,2

Accepted: 12 February 2024 / Published online: 29 February 2024

© The Author(s), under exclusive licence to Springer Science+Business Media, LLC, part of Springer Nature 2024

## Abstract

Data processing is a key challenge for computationally limited Ground Users (GUs) in various applications. Unmanned Aerial Vehicles (UAVs) equipped with Multi-access Edge Computing (MEC) servers can assist GUs by offloading their computing tasks. However, existing work ignores fairness when multiple GUs compete for limited computing resources, which may result in UAV underserving certain GUs. In this paper, we investigate a flight trajectory optimization based on reinforcement learning for UAV selection of target GUs for task computation, which provides low latency and fair offloading computing services for GUs by jointly training UAV flight trajectories and task offloading decisions. We formulate UAV flight and offloading as a mixed integer non-convex optimization problem with high-dimensional state and action spaces. The problem is then transformed into a Markov Decision Processes (MDPs) problem and the Maximizing Service Efficiency Proximal Policy Optimization (MSE-PPO) algorithm is proposed to find the optimal solution. The algorithm adopts an actor-criticbased parallel architecture to handle the parameterized action space. Specifically, the UAV position sequence is updated while ensuring an optimal offloading policy between the UAV and the GUs. Simulation results verify that the average system rewards including computational energy efficiency and fairness index are improved by 35.06% and 12.10% respectively compared to DDPG and PPO algorithms.

Keywords Unmanned aerial vehicle (UAV) · Reinforcement learning · Flight trajectory · Task offloading

## 1 Introduction

As Internet of Things (IoT) advances, there has been an escalating trend in Ground Users (GUs), fueling a dramatic expansion in computationally taxing functions such as video conferencing, sensory information gathering, and virtual augmentation infused into everyday life [1–3]. However, constraints imposed by scarce computing facilities and power supplies have posed substantial challenges in delivering an optimal user experience for local GUs [4]. In response, Multi-access Edge Computing (MEC) has emerged as a groundbreaking solution, outsourcing computationally burdensome functions to an edge server positioned proximate to the mobile device before transmitting the processed results. MEC is capable of substantially curtailing computational energy utilization and the demand for on-site memory provision on mobile device while maintaining the advantages of low latency and compatibility across diverse network configurations [5].

Unmanned Aerial Vehicles (UAVs) have received wide attention from researchers due to their characteristics of mobility, flexibility and rapid deployment [6–9]. UAVenabled MEC networks can provide temporary data collection, task computing, wireless charging, and other services when ground-based base station is damaged and unable to provide network coverage [10–13]. UAV-enabled MEC networks can leverage the good line-of-sight between UAVs and GUs to provide communication services. Moreover, they can offer additional computational resources to GUs to enhance their battery life and mission computation, and ease the communication and computational congestion in the core network.

UAV-enabled MEC is an innovative technology capable of delivering data collection and task offloading services to GUs using edge computing nodes integrated into UAVs [14, 15]. Nevertheless, the UAV-assisted GU task offloading process faces substantial challenges. Primarily, given the limited energy capacity of both UAV and GUs, the overall service effectiveness of assisting GUs requires optimal consideration of their energy consumption to reduce the processing latency of GUs [16]. Secondly, UAVs effectively limit the scope of communication computing services, thus presenting a challenge of unfair service to GUs where some GUs are allocated for multiple offloading services, while others remain unprovided. It is crucial to contemplate fairness concerns when multiple GUs vie for limited resources [17]. Presently, several studies have focused on energy consumption optimization, trajectory design, resource allocation, and other aspects of UAV-assisted GU edge computing, but these studies have not considered the three dimensions of energy consumption, latency, and fairness simultaneously. Therefore, this article attempts to solve this comprehensive optimization problem.

Addressing the benefits of UAV-enabled MEC systems and the shortcomings of existing work, this paper introduces a new reinforcement learning algorithm architectu101re to optimize UAV trajectories and task offloading strategies. This work aims to investigate the joint computation and flight problem in the two-layer paradigm of Air-to-Ground (A2G) communication, thus providing new ideas for future integrated A2G network architectures. Our research is relevant in practical scenarios, such as natural disasters causing damage to communications facilities or temporary offloading requirements for data volumes in hotspots, where flexible UAVs can be rapidly deployed. Therefore, it is necessary and promising to study flight planning and task offloading strategies. In this paper, we present a new approach to hybrid actor-critic architecture based on proximal policy optimization. The proposed algorithm takes into account the time-varying communication conditions to achieve the optimal task offloading decisions for each GU. This paper offers several novel contributions:

• We establish a UAV-enabled MEC model incorporating communication, offloading computation, and energy consumption, while considering service fairness within the UAV’s finite computing capabilities. Flight and offloading issues are formalized as Markov Decision Processes (MDPs), encompassing the state space of the

UAV’s constrained power and GUs’ locations, alongside the action space of flight control and offloading decisions.

• To ensure fairness in the system, we set lower and upper limits for compute bits each GU per time slot, thus preventing overloading the UAV by excessive computing tasks from a single GU. In addition, a service fairness index is designed to assess the collective service fairness of all GUs, indicating the overall UAV service scenario. This index is synced with incremental computing efficiency in the MDP reward function to further incentivize higher fairness actions by UAV.

• To solve the control problem in the combined flight and offloading action domain, we propose an algorithm to maximize serviceefficiency (MSE-PPO). Thisapproachemploys an actor-critic-based, parametrized parallel action space design, subsequently reframing its loss function accordingly, updating task offloading and flight hybrid policy parameters separately per time interval.

The rest of the paper is structured as follows: We review the related work on fixed base station MEC and UAV-enabled MEC in Section 2. We introduce the UAV flight and offloading computation related models in Section 3. Additionally, we formulate the optimization problem as a markov decision process and describe the problem setting in Section 4. We also introduce the deep reinforcement learning algorithm and analyze its complexity. In Section 5, we conduct experiments and discuss the results. In Section 6, we identify the limitations of our study and the future research directions. We summarize this paper in Section 7. Finally, we briefly describe future work in Section 8.

## 2 Related work

Technical studies of fixed MEC infrastructures [18–23] aim to minimize the processing latency and energy consumption of GUs by optimizing the task resource allocation scheme for this end-to-end interaction mode between GUs and MEC servers. The excellent mobility and cost-effectiveness of UAVs has brought UAV-enabled MEC into focus. One common approach to optimize UAV trajectories and task offloading strategies for better system performance is to formulate the problem as an optimization problem with constraints on transmission rate, delay, and energy efficiency [24–36]. Solutions to such non-linear, multi-constraint, mixed-integer optimization problems can be classified into three categories: convex optimization, heuristic-based approach, and DRL.

Successive convex approximation optimization algorithms have been used widely [24–27]. Wu et al. [25] proposed a resource management scheme to jointly optimize the scheduling of GUs, UAV trajectories, and energy controls to maximize the throughput of task offloading. The literature [26] optimized the UAV trajectory to minimize the energy consumption of all GUs but ignored the task of offloading throughput requirements of GUs. Unlike [25] and [26], which consider only single-link, the UAV in [24] acts as a full-duplex base station and jointly optimized the UAV trajectory, bidirectional link GU scheduling selection, and uplink GUs transmit power. The literature [27] aims to achieve the maximum minimum average throughput by jointly optimized UAV trajectory and Orthogonal Frequency Division Multiple Access (OFDMA) resource allocation. Although convex functions can yield closed-loop solutions to related research problems, convex functions are difficult to find globally optimal solutions to nonconvex problems. However, these approaches [24–27] are computationally intensive and poorly adaptable in the dynamic MEC environment.

As shown in the literature [28–30], heuristic-based approaches can provide feasible solutions for complex optimization problems. A new genetic algorithm-based offloading scheduling method is proposed in the literature [29] to maximize UAV coverage under GU random distribution conditions but ignoring the energy consumption generated by the transmission process. Xu et al. [30] proposed an online resource allocation and trajectory optimization algorithm with external and internal structures. The external structure transforms the task offloading problem into a deterministic problem by applying a Lyapunov-based optimization framework. The internal structure suppresses UAV energy consumption by designing a velocity-triggered penalty term based on a block coordinate descent framework using Lagrangian dyadic method and continuous convex approximation technique. Unlike the literature [29, 30], Wang et al. [28] proposed a genetic algorithm-based heuristic joint optimization energy and UAV trajectory algorithm to solve such mixed integer nonlinear programming problems, which improves the GUs transmission efficiency and reduces the complexity of multiple optimization objectives. Although the heuristic algorithm can obtain the global optimal solution in a short time for many practical problems, it may fall into the local optimal solution and there is a lack of a complete theoretical system of heuristic algorithms.

The UAV-enabled MEC environment is non-fully observable and highly complex, so it is hard to model it as a regular optimization problem. Therefore, the conventional algorithms based on optimization theory or game theory are not suitable for this scenario. Deep Reinforcement Learning (DRL) has attracted attention due to its model-free features and high learning capability for nonlinear hybrid optimization problems. DRL uses deep neural networks to capture the complex state of UAV-enabled MEC environment and makes decision through reinforcement learning to learn and optimize UAV flight trajectories, task resource allocation without training data [31]. Li et al. [32] proposed a triple learner-based reinforcement learning approach to model the computational task scheduling problem as a stochastic game of coupled intelligences involving trajectory planning, energy updating, and application placement of UAVs. Similarly, Song et al. [33] proposed an application of a multi-objective RL algorithm to solve the trajectory control and task offloading problems, achieving minimization of task latency and UAV energy consumption, and maximizing the number of tasks collected by the UAV. However, the above literature does not consider the user task equalization problem of UAV during the offloading process. In [34], resource allocation, terminal association, and power are jointly optimized, and a Deep Q-Network (DQN) based reinforcement learning approach is proposed to achieve total power minimization and reduced runtime compared to centralized algorithms. Wang et al. [35] considered the time-varying channel state for UAV-assisted terminal task offloading and used the Deep Deterministic Policy Gradient (DDPG) algorithm to jointly optimize GU scheduling, UAV mobility, and task assignment to minimize the maximum processing delay. Unlike the above studies, Zhou et al. [36] considered the fairness among different terminals and proposed a Soft Actor-Critic (SAC) based UAV trajectory planning and resource allocation algorithm to maximize the terminal computation bits. Compared to traditional optimization algorithms, DRL-based models generally achieve better results in dealing with decision problems. For example, DQN [37] and Asynchronous Advantage Actor-Critic (A3C) [38] have better performance in discrete action spaces. DDPG [39] is an offline and deterministic policy gradient approach, which does not explore the action space well if the actions are not noise-laden. Proximal Policy Optimization (PPO) [40] adopted a truncated alternative target approach to ensure monotonic boosting in the direction of policy optimization, which is faster compared to the Trust Region Policy Optimization (TRPO) approach [41]. PPO inherits all the advantages of TRPO, but has much lower implementation complexity than this algorithm. We propose a parameterized action space architecture based on the PPO algorithm with AC architecture, and optimize the discrete continuous action space considered in this paper with multiple types of parameters to achieve offloading computation services. To provide a comprehensive overview of the existing studies, we summarized the surveys mentioned above in Table 1.

## 3 System model

As shown in Fig. 1, we consider a square area scenario with the UAV-enabled MEC system. The UAV provides communication and offloading computation services to GUs with random movement. The whole service period T is divided into N equal time periods and the set of GUs is denoted as ${ \mathcal { K } } \triangleq \{ 1 , 2 , . . . , K \}$ . The UAV needs to optimize trajectory and

<table><tr><td>Strategy</td><td colspan="6">Optimization objective</td><td colspan="2">Offloading type</td><td rowspan="2">Reference</td></tr><tr><td></td><td>Energy consumption Latency</td><td>Fairness Throughput Coverage percentage Computation bits</td><td></td><td></td><td>Local device Edge Cloud</td><td></td><td>Binary Partial</td><td></td></tr><tr><td>Game-theoretic approach</td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>Ref. [18]</td></tr><tr><td>Federal gradient descent algorithm</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td>Ref. [19]</td></tr><tr><td>DRL-based online algorithm</td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td>Ref. [20]</td></tr><tr><td>Convex optimization</td><td></td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [21]</td></tr><tr><td>Relaxation-based generalized</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td>√</td><td>Ref. [22]</td></tr><tr><td>benders algorithm</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> Deep learning</td><td></td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [23]</td></tr><tr><td>Convex optimization</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [24]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [25]</td></tr><tr><td></td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td>Ref. [26]</td></tr><tr><td>Heuristic-based approach</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td>Ref. [27]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [28]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [29]</td></tr><tr><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td></td><td></td><td>Ref. [30]</td></tr><tr><td>DRL</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>Ref. [31]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [32]</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td>Ref. [33]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td>Ref. [34]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>√</td><td>Ref. [35]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>Ref. [36]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td><td>Our work</td></tr></table>

Fig. 1 UAV-enabled MEC system scenario

offloading scheduling to provide better service to GUs. Due to the high energy consumption generated in the flight and elevation of the UAV, we assume that the UAV flies at a fixed altitude H. The horizontal position of the UAV at time slot n can be expressed as $q ( n ) = [ x ( n ) , y ( n ) ] ^ { T } \in \mathbb { R } ^ { 2 }$ , where the time slot $n \in \{ 1 , 2 , . . . , N \}$ . The GU at the time slot n level position can be denoted as $P _ { k } ( n ) = [ x _ { k } ( n ) , y _ { k } ( n ) ] ^ { T } \in \mathbb { R } ^ { 2 }$ ， where $k \in \mathcal { K }$

## 3.1 Communication model

Full-duplex relay communication is employed in the task of offloading UAV-assisted GUs, and the residual selfinterference channel model can be set up as a quasi-static fading channel model with a known constant self-interference power [42]. The A2G communication link can be modeled as a probabilistic Path Loss (PL) model. The Line-of-Sight (LoS) link connection probability $P _ { U A V , k } ^ { L o S } ( n )$ of the UAV and GU k at time slot n can be expressed as

$$
P _ { U A V , k } ^ { L o S } ( n ) = \frac { 1 } { 1 + a \exp \left( - b \mathrm { s i n } ^ { - 1 } ( \frac { H } { d _ { U A V , k } } ) - a \right) } ,\tag{1}
$$

where $d _ { U A V , k } = \sqrt { ( x ( n ) - x _ { k } ( n ) ) ^ { 2 } + ( y ( n ) - y _ { k } ( n ) ) ^ { 2 } + H ^ { 2 } }$ denotes the Cartesian 3D coordinate distance between the UAV and GU k. a and b are environment-dependent constants. Therefore, the Non-Line-of-Sight (NLoS) link connection probability is $P ^ { N L o S } ( n ) = 1 - \overline { { P _ { U A V . k } ^ { L o S } ( n ) } }$ . The path loss of the LoS link and NLoS link between the UAV and GU k is

$$
P L _ { U A V , k } ^ { L o S } ( n ) = 2 0 \log \left( \frac { 4 \pi d _ { U A V , k } f } { \overline { { c } } } \right) + \eta ^ { L o S } ,\tag{2}
$$

<!-- image-->

$$
P L _ { U A V , k } ^ { N L o S } ( n ) = 2 0 \log \left( \frac { 4 \pi d _ { U A V , k } f } { \overline { { c } } } \right) + \eta ^ { N L o S } ,\tag{3}
$$

where f is the carrier frequency, c is the speed of light, $\eta ^ { L o S }$ and $\eta ^ { N L o S }$ denote the additional path loss of the LoS link and NLoS link, respectively. Therefore, the average path loss $\overline { { L } } _ { U A V , k } ( n )$ of the UAV and GU k at time slot n can be expressed as

$$
\begin{array} { r l } & { \overline { { L } } _ { U A V , k } ( n ) = P _ { U A V , k } ^ { L o S } ( n ) \times P L _ { U A V , k } ^ { L o S } ( n ) } \\ & { \qquad + P _ { U A V , k } ^ { N L o S } ( n ) \times P L _ { U A V , k } ^ { N L o S } ( n ) . } \end{array}\tag{4}
$$

The GU k uplink transmission rate $r _ { k } ( n )$ at time slot n can be expressed as

$$
r _ { k } ( n ) = B \mathrm { l o g } _ { 2 } ( 1 + \frac { P _ { u p } 1 0 ^ { - \overline { { L } } _ { U A V , k } ( n ) / 1 0 } } { \sigma ^ { 2 } } ) ,\tag{5}
$$

where B denotes the communication bandwidth, $P _ { u p }$ denotes the transmitted power of GU k in the uplink at time slot n, and $\sigma ^ { 2 }$ is the power of Gaussian white noise.

## 3.2 Fair service model

Each GU in this UAV-enabled MEC system generates offloading tasks with different content and each offloading task has three configurations: $\{ D _ { k } ( n ) , s , { \overline { { T } } } ( n ) \}$ . Where $D _ { k } ( n )$ denotes the size of the computation bits generated by GU k at time slot n, s denotes the number of CPU cycles required to process the computation bits, and $\overline { { T } } ( n )$ denotes the maximum latency tolerated during task processing. To ensure network fairness and stability, each GU has a lower and an upper bound for the computation bits in each time slot, i.e.,

$$
D _ { k } ^ { \operatorname* { m i n } } \leq D _ { k } ( n ) \leq D _ { k } ^ { \operatorname* { m a x } } ,\tag{6}
$$

where $D _ { k } ^ { \mathrm { m i n } }$ represents the minimum generated bits to be processed to ensure the fairness among GUs, and $D _ { k } ^ { \mathrm { m a x } }$ indicates the maximal reasonable workload to be scheduled according to the computational capability at the UAV shared among GUs. Task offloading decision scheduling was denoted by binary variable $\delta _ { k } ( n ) \ \in \ \{ 0 , 1 \}$ , which could be defined as

$$
\delta _ { k } ( n ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ t h e ~ U A V ~ s e l e c t s ~ G U } } k } \\ { 0 } & { { \mathrm { ~ O t h e r w i s e } } } \end{array} \right. } ,\tag{7}
$$

where $\delta _ { k } ( n ) ~ = ~ 1$ means that the GU itself processes the task beyond the maximum allowable delay and needs UAV assistance to offload the computation. $\delta _ { k } ( n ) ~ = ~ 0$ means that the UAV is in the exploration period, or the GU can compute on its own. The UAV has limited processing capacity to ensure that only one GU satisfying the offloading condition can be provided with the association service in a time slot, which can be expressed as

$$
\sum _ { k = 1 } ^ { K } \delta _ { k } ( n ) = 1 .\tag{8}
$$

The UAV may have unfair service provisioning for GUs, with some GUs are served during multiple time slots while others are not served at all. Based on the Jain Fairness Index [43], we design a service fairness index $f ^ { s } ( n )$ to reflect the overall service fairness of the UAV, which can be expressed as

$$
f ^ { s } ( n ) = { \frac { \left( \sum _ { k = 1 } ^ { K } \sum _ { n \cdot = 1 } ^ { n } \delta _ { k } ( n \cdot ) \right) ^ { 2 } } { K \sum _ { k = 1 } ^ { K } \left( \sum _ { n \cdot = 1 } ^ { n } \delta _ { k } ( n \cdot ) \right) ^ { 2 } } } ,\tag{9}
$$

where $1 / K ~ \le ~ f ^ { s } ( n ) ~ \le ~ 1$ indicates the service fairness among GUs. A higher value of $f ^ { s } ( n )$ means more similar UAV offloading computation services for all GUs from the initial time slot $n ^ { \prime } = 1$ to the time slot n.

## 3.3 Offloading computation model

When GU k decides to offload the task to the UAV at time slot n, GU k must be in the coverage area of the UAV, which can be expressed as

$$
\delta _ { k } ( n ) R _ { U A V , k } ( n ) \leq R ^ { \operatorname* { m a x } } ,\tag{10}
$$

where $\begin{array} { r l r } { R _ { U A V , k } ( n ) } & { { } = } & { \sqrt { ( x ( n ) - x _ { k } ( n ) ) ^ { 2 } + ( y ( n ) - y _ { k } ( n ) ) ^ { 2 } } } \end{array}$ denotes the horizontal distance between GU k and the UAV, and $R ^ { \mathrm { m a x } }$ is the maximum horizontal coverage radius of the UAV. After establishing the communication association between GU k and the UAV, the transmission communication time $t _ { k } ^ { c o m } ( n )$ can be expressed as

$$
t _ { k } ^ { c o m } ( n ) = \frac { \delta _ { k } ( n ) D _ { k } ( n ) R _ { k } ( n ) } { r _ { k } ( n ) } ,\tag{11}
$$

where $D _ { k } ( n )$ denotes the generation task size of GU k. we set $R _ { k } ( n ) \in [ 0 , 1 ]$ to be the task ratio of GU k offloading to the UAV at time slot n. After the transmission communication is completed, the UAV can perform the offloading task computation and the offloading computation time $t _ { U A V } ^ { o f \breve { f } } ( n )$ can be expressed as

$$
t _ { U A V } ^ { o f f } ( n ) = \frac { D _ { k } ( n ) R _ { k } ( n ) s } { f _ { U A V } } ,\tag{12}
$$

where $f _ { U A V }$ denotes the CPU processing power of the UAV. According to the above expression, the computation time $t _ { k } ^ { l o c } ( n )$ of local task is expressed as

$$
t _ { k } ^ { l o c } ( n ) = \frac { D _ { k } ( n ) \left( 1 - R _ { k } ( n ) \right) s } { f _ { k } } ,\tag{13}
$$

where $f _ { k }$ denotes the CPU processing power of GU k. Since the amount of computation result data from the MEC server is much smaller than the offloading task itself, the backhaul time to send the result back to the GU is ignored. The total time cost is the maximum of the transmission and offloading times or the local computation time. The total time (denoted as $t _ { c o m p l e t e d } ( n ) \rangle$ it takes for a task to be generated, processed and completed by the GU is called the delay cost, which is calculated as

$$
t _ { c o m p l e t e d } ( n ) = \operatorname* { m a x } \left( t _ { k } ^ { c o m } ( n ) + t _ { U A V } ^ { o f f } ( n ) , t _ { k } ^ { l o c } ( n ) \right) .\tag{14}
$$

## 3.4 System energy model

At time slot n, the UAV flies from position $q ( n )$ to the new hovering position

$$
\begin{array} { c } { { q ( n + 1 ) = [ x ( n ) + v ( n ) t _ { f l y } \cos \alpha ( n ) , y ( n ) } } \\ { { \nonumber } } \\ { { + v ( n ) t _ { f l y } \sin \alpha ( n ) ] ^ { T } , } } \end{array}\tag{15}
$$

where $\alpha ( n ) \in [ 0 , 2 \pi ]$ ] and $t _ { f l y }$ are the angle and time of flight, respectively. Due to the high mobile energy consumption, we assume that the UAV has the ability to reach the designated charging location after the assisted GUs have completed the offloading task with the following flight speed v(n) [44]

$$
v ( n ) = \frac { \parallel q ( n + 1 ) - q ( n ) \parallel } { T } .\tag{16}
$$

We assume that the flight power $P _ { f }$ is determined and the UAV flight energy consumption is only determined by

the distance moved, then the current time slot flight energy consumption $e _ { f } ( n )$ can be expressed as

$$
e _ { f } ( n ) = P _ { f } { \frac { \parallel q ( n + 1 ) - q ( n ) \parallel } { v ( n ) } } .\tag{17}
$$

The transmission communication energy consumption $e _ { c o m } ( n )$ generated by the UAV during task offloading period of GU k can be expressed as

$$
e _ { c o m } ( n ) = P _ { u p } t _ { k } ^ { c o m } ( n ) .\tag{18}
$$

The computational energy consumption $e _ { o f f } ( n )$ generated by the UAV can be expressed as

$$
e _ { o f f } ( n ) = \kappa D _ { k } ( n ) R _ { k } ( n ) s f _ { U A V } ^ { 2 } ,\tag{19}
$$

where $\kappa ~ = ~ 1 0 ^ { - 2 6 }$ is the impact factor of hardware construction on CPU processing, the power consumption of the processor is $\kappa f _ { U A V } ^ { 3 }$ . The local computational energy consumption (denoted as $e _ { k } ^ { l o c } ( n ) )$ of GU k can be expressed as

$$
e _ { k } ^ { l o c } ( n ) = \kappa D _ { k } ( n ) \left( 1 - R _ { k } ( n ) \right) s f _ { k } ^ { 2 } .\tag{20}
$$

Then the total energy cost (denoted as $e _ { t o t a l } ( n ) )$ generated in the current time slot n is

$$
e _ { t o t a l } ( n ) = e _ { f } ( n ) + \delta _ { k } ( n ) \left( e _ { c o m } ( n ) + e _ { o f f } ( n ) + e _ { k } ^ { l o c } ( n ) \right) .\tag{21}
$$

## 4 Optimization UAV flight trajectory and offloading decision

We apply reinforcement learning to achieve task offloading in a UAV-assisted GUs environment with a large number of continuous states and hybrid action spaces. This section first presents the problem formulation for the UAV-enabled MEC system, and then MSE-PPO DRL algorithm is proposed to achieve offloading computation service.

## 4.1 Problem formulation

In the proposed UAV-enabled MEC system, the goal is to maximize computational efficiency by training UAV flight trajectory and task offloading decision to provide low-latency and fair offloading computation services to GUs. Thus, the optimization problem can be expressed as

$$
\operatorname* { m a x } _ { \delta _ { k } , \alpha _ { u } , v _ { u } , R _ { k } } \sum _ { n = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \frac { r _ { k } ( n ) f ^ { s } ( n ) } { e _ { t o t a l } ( n ) } ,\tag{22}
$$

s.t . C 1 $D _ { k } ^ { \operatorname* { m i n } } \leq D _ { k } [ n ] \leq D _ { k } ^ { \operatorname* { m a x } } , \forall n \in \{ 1 , 2 , . . . , N \} , k \in \mathcal { K }$

$$
\begin{array} { r l } & { C ^ { 2 } : \mathcal { E } _ { k } ( n ) \in \{ 1 , 1 , \forall k \in \{ 1 , 2 , \ldots , N \} , k \in \mathcal { K } \} } \\ & { \bullet : \mathcal { K } } \\ & { C : \sum _ { k = 1 } ^ { N } \delta ( n ) = 1 , \forall k \in \{ 1 , 2 , \ldots , N \} } \\ & { C : \mathcal { K } : \mathcal { K } _ { k } ( n ) \leq 1 , \forall k \in \{ 1 , 2 , \ldots , N \} , k \in \mathcal { K } } \\ & { C : \mathcal { K } : \mathcal { Q } ( n ) \leq \{ 1 , ( \forall k ( n ) ) , \forall \theta \} \leq \{ 0 , 1 \} , } \\ & { \forall k \in \{ 1 , 2 , \ldots , N \} } \\ & { C : \mathcal { K } : \mathcal { P } ( n ) \in \{ 1 , ( \mathtt { s c } ( n ) , \forall \theta ) \} \leq \{ 0 , 1 \} , } \\ &  \forall k \in \{ 1 , 2 , \ldots , N \} , k \in \mathcal { K } \} \\ & { C : \mathcal { K } : \mathcal { Q } ( n ) \leq \{ 1 , ( \forall k ( n ) , \forall \theta ) \in \{ 1 , 2 , \ldots , N \} \} } \\ & { C : \mathcal { K } : \mathcal { K } : \mathcal { K } = \{ 1 , ( \forall k ( n ) ) \leq \{ 1 , 2 , \ldots , N \} \} } \\ & { C : \mathcal { K } : \mathcal { K } : \mathcal { K } = \{ 1 , ( \forall k ( n ) ) \leq \{ 1 , 2 , \ldots , N \} \} } \\ & { C : \Delta _ { k } ( n ) : \mathcal { K } : \mathcal { K } = \{ 1 , ( \forall k ( n ) , \forall \theta \in \{ 1 , 2 , \ldots , N \} ) \} } \\ & { C : \Delta _ { k } ( n ) : \mathcal { K } : \mathcal { K } = \{ 1 , ( \forall k ( n ) , \forall \theta \in \{ 1 , 2 , \ldots , N \} ) \} } \\ & { C : \Delta _ { k } ( n ) : \mathcal { K } : \mathcal { K } = \{ 1 , ( \forall k ( n ) ) \leq \{ 1 , 2 , \ldots , N \} \} } \\ &  C : \Delta _ { k } ( n ) : \mathcal { K } : \mathcal \end{array}
$$

where constraint C1 indicates the limit of task bits generated by the GU within the time slot. Constraints C2 and C3 ensure that the UAV can only offload task processing for one GU within the time slot. Constraint C4 indicates the range of values of task offloading ratio. Constraints C5 and C6 ensure that the UAV and GU cannot move beyond the set region. Constraint C7 ensures that the total time cost between task generation and processing completion back to the GU cannot exceed the maximum allowable delay. Constraint C8 is the range of UAV flight speed. Constraint C9 is that the GU needs to be within the UAV coverage area to perform offloading computation services, and constraint C10 specifies that total task D need to be completed within the whole service cycle.

## 4.2 Problem transformation

In existing reinforcement learning settings, a MDP includes a state space S and an action space A [33]. In an MDP, an agent observes the current state $s _ { t }$ of the environment and chooses an action $a _ { t }$ that is feasible. The agent receives a reward $r _ { t }$ for taking the action and transitions to the next state $s _ { t } + 1$ . During the interaction with the environment, the agent aims to choose the best strategy $\pi _ { \theta } ( a _ { t } | s _ { t } )$ to maximize the cumulative reward $\mathbb { E } _ { ( s _ { 0 } , a _ { 0 } , \ldots ) } [ \sum _ { t = 0 } ^ { \infty } \gamma t r ( s _ { t } ) ]$ , where $( s _ { 0 } , a _ { 0 } , . . . )$ denotes the agent’s historical information, Et ... represents the empirical average of the sample and $\gamma \in [ 0 , 1 ]$ is a discount factor indicating the effect of future rewards on the current agent’s behavior. $\gamma = 1$ means that the reward value of the future state has a large effect on the current time slot action state value function, while $\gamma = 0$ means that the reward value of the future state has almost no effect on the current time slot action state value function.

## A. State Space

In UAV-enabled MEC system, the state space is jointly determined by the K GUs, the UAV and the environment in which they are located. The state of the system at time slot n can be defined as

$$
\begin{array} { r } { s ( n ) = \{ E _ { b a t t e r y } ( n ) , q ( n ) , P _ { 1 } ( n ) , . . . , P _ { K } ( n ) , D _ { r e m a i n } ( n ) , D _ { 1 } ( n ) , . . . , D _ { K } ( n ) \} , } \end{array}\tag{23}
$$

where $E _ { b a t t e r y } ( n )$ denotes the remaining power of the UAV in time slot $n , q ( n )$ denotes the location information of the UAV, $P _ { 1 } ( n ) , . . . P _ { K } ( n )$ denotes the location information of the UAV serving GU K , $D _ { r e m a i n } ( n )$ ) denotes the size of the remaining tasks to be completed by the UAV-enabled MEC system in the whole service cycle, and $D _ { K } ( n )$ denotes the size of the remaining tasks to be completed by the UAVenabled MEC system in the whole service cycle, and $D _ { K } ( n )$ denotes the size of the tasks randomly generated by GU K in time slot n. Initially, time slot $n = 1 , E _ { b a t t e r y } ( n ) = E _ { b }$ and $D _ { r e m a i n } ( n ) = D$

## B. Action Space

The UAV selects the action including the GU K to be served, the flight angle, the flight speed, and the task offloading ratio to be executed at time slot n according to the current state of the system and the observation environment. The action a(n) can be expressed as

$$
a ( n ) = \{ \delta _ { k } ( n ) , \alpha ( n ) , \upsilon ( n ) , R _ { k } ( n ) \} ,\tag{24}
$$

it is worth noting that the action space executed by the UAV consists of discrete and continuous actions. The discrete actions include indicator function $\delta _ { k } ( n ) \ \in \ \{ 0 , 1 \}$ to guide whether the UAV provides offloading computation services to the GU K under the above constraints, with $\delta _ { k } ( n ) = 1$ if offloading is provided and 0 otherwise. The continuous actions include the flight angle and speed of the UAV, and the offloading ratio of the task to be decided based on the CPU processing power of GU K and the corresponding task configuration generated in each time slot, where $\alpha ( n ) \in [ 0 , 2 \pi ]$ $v ( n ) \in [ 0 , v _ { \operatorname* { m a x } } ] , R _ { k } ( n ) \in [ 0 , 1 ]$ . The above four variables are jointly and precisely optimized to maximize the computational efficiency to provide low-latency and fair offloading computation services for GUs.

## C. Reward Function

We design a reward-based optimization problem to achieve low-latency and fair computation offloading services. The UAV actions depend on the reward, which reflects the effect of an agent’s action in the current state. On the one hand, incremental computational efficiency $\scriptstyle \sum _ { k = 1 } ^ { K } { \frac { r _ { k } ( n ) } { e _ { t o t a l } ( n ) } }$ is included to encourage improvement. On the other hand, to measure the cumulative service fairness among GUs, the service fairness index $f ^ { s } \left( n \right)$ is combined with incremental computational efficiency to encourage the agent to achieve higher fairness actions. The reward elements in time slot n can be expressed as

$$
r \left( s \left( n \right) , a ( n ) \right) = \log _ { 2 } \left( f ^ { s } \left( n \right) . \sum _ { k = 1 } ^ { K } \frac { r _ { k } ( n ) } { e _ { t o t a l } \left( n \right) } + \xi \right) - P _ { 0 } ,\tag{25}
$$

where a penalty $P _ { 0 }$ is given if the transmitted data rate in time slot n is below the average, as in $\begin{array} { r } { r _ { k } ( n ) t _ { k } ^ { c o m } \left( n \right) < \frac { D _ { k } } { N } } \end{array}$ To avoid drastic fluctuations in the reward function, we use the main reward by the logarithmic function and restrict the minimum value of the main reward element to log2(ξ ).

## 4.3 The MSE-PPO algorithm

The conventional policy gradient algorithm can handle many complex decision-making problems. However, it also has many challenges. The agent system is very sensitive to the changes in the training environment, cannot explore the state space in real time, and has a high implementation complexity. The PPO-based algorithm works well for single-agent training environments, where agents can make independent decisions using their policies and observations to improve network performance in dynamic scenarios. We propose a novel robust algorithm MSE-PPO, which adopts a parallel actor-critic architecture to simultaneously handle discrete task offloading decisions and continuous flight trajectory control in dynamic environments. It can better enables UAVs to assist GUs in task computation. Figure 2 shows the overall architecture of the MSE-PPO algorithm.

In this architecture, the UAV acts as an agent that selects an action to execute based on the policy π and the current environmental state. The environment then responds with a new state and a reward based on the UAV’s action. The UAV then chooses a new action based on the new state. The policy π is composed of a neural network with the parameter θ, which can be described as: $\pi _ { \boldsymbol { \theta } } ( s , a ) = P ( a | s , \boldsymbol { \theta } ) \approx \pi ( a | s )$ In general, the environment model is unknown, therefore the actual expectation value can be obtained from the statistical data trajectory value, and the gradient ascent method is used to optimize the expectation value to achieve the goal of optimizing the policy parameters. In the policy gradient algorithm, the optimal objective of the strategy is defined as

$$
J ( \theta ) = \mathbb { E } _ { r ( s , a ) \sim \pi _ { \theta } } [ \sum _ { t = 0 } ^ { T } r ( s _ { t } , a _ { t } ) ] = \mathbb { E } _ { r ( s , a ) \sim \pi _ { \theta } } [ r ( \tau ) ] ,\tag{26}
$$

where τ denotes the data trajectory. The probability of occurrence of the data trajectory τ is

$$
P ( \tau | \theta ) = \rho _ { 0 } ( s _ { 0 } ) \prod _ { t = 0 } ^ { T } P ( s _ { t + 1 } | s _ { t } , a _ { t } ) \pi _ { \theta } ( a _ { t } | s _ { t } ) ,\tag{27}
$$

Fig. 2 The parallel architecture of the MSE-PPO algorithm  
<!-- image-->

where state $s _ { 0 }$ as the starting point, and $\rho _ { 0 } ( s _ { 0 } )$ denotes the probability of occurrence of state $s _ { 0 }$ . Based on the above policy gradient optimization objective, the gradient value of the objective function is

$$
\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } ) = \mathbb { E } _ { \pi _ { \boldsymbol { \theta } } } \left[ \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { s } , \boldsymbol { a } ) r ( \tau ) \right] .\tag{28}
$$

For the PPO algorithm based on policy gradient, we need to optimize the parameter set at each time step. In the policy gradient, the main estimator is defined as $g _ { t } ( \theta ) =$ $\mathbb { E } _ { t } \left[ \nabla _ { \theta } \log \pi _ { \theta } \left( \alpha _ { t } \vert s _ { t } \right) A _ { t } \right]$ , where $A _ { t }$ is the estimated value of the model’s advantage function in time step t. The estimator is obtained from the loss function

$$
L ^ { P G } ( \theta ) = \mathbb { E } _ { t } \left[ \log \pi _ { \theta } \left( \alpha _ { t } \vert s _ { t } \right) A _ { t } \right] .\tag{29}
$$

As show in Fig. 3, considering the hybrid optimization problem of solving UAV flight planning and task offloading decision, we proposed a parameterized action space parallel architecture based on AC style. The parallel actor network executes corresponding action parameter selection respectively: discrete actor network learns random policy $\pi _ { \theta _ { d } }$ to select discrete action $\delta _ { k } ,$ , continuous actor network learns random policy $\pi _ { \theta _ { c } }$ to select flight control and task offloading rate. Critic network with evaluation function provides the estimation of the advantage function, and updates the random policy of parallel actor network with the advantage function provided by the critic network. The complete action to be executed is to pair the selected discrete action $\delta _ { k }$ with the selected continuous action parameters corresponding to $\delta _ { k }$ . The parallel actor networks share the first few layers to encode state information. Due to the existence of both discrete and continuous actors in this architecture, we refer to the proposed architecture as a hybrid AC architecture.

According to this parallel action space, we reconstruct its loss function to adopt different update ways to accelerate the convergence of reward function in hybrid policy optimization problem. The algorithm combines policy surrogate with the loss function of value function (entropy reward), which can be expressed as

$$
L _ { t } ^ { h y b r i d + V F + S } ( \theta ) = \mathbb { E } \left[ L _ { t } ^ { h y b r i d } ( \theta ) - c _ { 1 } L _ { t } ^ { V F } ( \omega ) + c _ { 2 } S [ \pi _ { \theta } ] ( s _ { t } ) \right] ,\tag{30}
$$

where $c _ { 1 }$ and $c _ { 2 }$ are coefficients, S is entropy reward, and $L _ { t } ^ { V F }$ is square-error loss. The main objective is given as follows

$$
L _ { t } ^ { h y b r i d } ( \theta ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( g _ { t } ^ { h y b r i d } ( \theta ) A _ { t } , c l i p \left( g _ { t } ^ { h y b r i d } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \right) A _ { t } \right) \right] ,\tag{31}
$$

<!-- image-->  
Fig. 3 The parallel architecture for parameterized action space

where ε is a hyperparameter, usually equal to 0.2. This goal is explained as follows. The first term inside the min is $L _ { t } ^ { h y b r i d } ( \theta )$ . The second item, cli p $\left( g _ { t } ^ { h y b r i d } , 1 - \varepsilon , 1 + \varepsilon \right)$ modifies the surrogate objective by clipping the probability ratio, which eliminates the possibility of $\dot { \boldsymbol g } _ { t } ^ { h y \bar { b } r i d }$ moving outside the interval $\left[ 1 - \varepsilon , 1 + \varepsilon \right] \left[ 4 0 \right]$ . Finally, the algorithm takes the minimum of the clipped and unclipped objectives.

$$
g _ { t } ^ { h y b r i d } ( \theta ) = c _ { 3 } \frac { \pi _ { \theta } ( a _ { c } | \hat { s } _ { t } ) } { \pi _ { \theta o l d } ( a _ { c } | \hat { s } _ { t } ) } + ( 1 - c _ { 3 } ) \frac { \pi _ { \theta } ( a _ { d } | \hat { s } _ { t } ) } { \pi _ { \theta o l d } ( a _ { d } | \hat { s } _ { t } ) } ,\tag{32}
$$

where $\theta _ { o l d }$ represents the old vector of the policy parameter, $c _ { 3 } ~ \in ~ ( 0 , 1 )$ is a weighting factor, and its optimum value will be studied in Section 5. Specifically, as shown in (Eqn.32), hybrid probability ratio ghybrid $g _ { t } ^ { h \bar { y } b r i d } ( \theta )$ is the combination of continuous action probability ratio $\alpha , v , R _ { k }$ , and discrete action probability ratio $\delta _ { k }$ . According to [40], if the probability ratio is not constrained, the update of policy parameters will be too large. The hybrid actor network takes the normalized $\hat { s } _ { t }$ as the input and generates the action distribution. The UAV obtains the action through sampling in the distribution. Actor networks generate policies, and critic networks evaluate the current policies and modify them by estimating the advantage function $A _ { t } .$ . The loss function of the critic network is

$$
L _ { t } ^ { V F } ( \omega ) = \mathbb { E } \left[ A _ { t } ^ { 2 } \right] ,\tag{33}
$$

where $A _ { t } = y _ { t } - \mathscr { Q } _ { \omega } ( s _ { t } , a _ { t } ) , y _ { t } = r _ { t } + \gamma \mathscr { Q } _ { \omega } \left( s _ { t + 1 } , \pi _ { \theta } \left( s _ { t + 1 } \right) \right)$ Critic network parameters are updated in gradient descent through the above loss function $\bar { L } _ { t } ^ { V F } ( \omega )$

## 4.4 Implementation of the MSE-PPO algorithm

## 4.4.1 Normalization of the UAV observations

During the training of actor and critic deep neural networks (DNNs), the input distribution of each layer dynamically changes with the upper layer network parameters, which requires fine parameter initialization and thus slows down the training. We use a state normalization algorithm to preprocess the states observed by the UAV. This helps us train the DNNs of actor and critic more efficiently and obtain more accurate normal distributions and Q values. Specifically, the algorithm uses the difference between the maximum and minimum values of each observed variable as a scaling factor, by which the input state space size discrepancy problem can be solved and the optimization target state input space efficiency can be enhanced. As shown in Algorithm 1, we use five scaling factors in the normalization operation, and each scaling factor is explained below. The scaling factor $\varphi _ { b }$ is used to scale down the UAV battery capacity. Consider a scenario where the UAV is at a fixed height and has the same range as the GU in the coordinate system, then use $\varphi _ { x }$ and $\varphi _ { y }$ to scale down the corresponding X and Y coordinates, respectively. We use $\varphi D _ { r e m a i n }$ to scale down the number of tasks remaining in the whole service cycle, and $\varphi _ { D _ { k } }$ to scale down the size of the tasks generated by the GU in time slot n.

Algorithm 1 State normalization.   
Input: All state observation variables that require to be normalized:   
$1 \colon s = \{ E _ { b a t t e r y } ( n ) , q ( n ) , P _ { 1 } ( n ) , . . . , P _ { K } ( n ) , D _ { r e m a i n } ( n ) , D _ { 1 } ( n ) , . . . , D _ { K } ( n ) \} ;$   
2: Scales factors: $\varphi _ { b } , \varphi _ { x } , \varphi _ { y } , \varphi _ { D _ { r m } } , \varphi _ { D _ { G U } } ;$   
Output: Normalized variables:   
3: $\bar { E } _ { b a t t e r y } ^ { \prime } ( n ) = E _ { b a t t e r y } ( n ) / \varphi _ { b } ;$   
4: $x ^ { \prime } ( n ) = x ( n ) / \varphi _ { x } ;$   
5: $y ^ { \prime } ( n ) = y ( n ) / \varphi _ { y } ;$   
6: $x _ { k } ^ { \prime } ( n ) = x _ { k } ( n ) / { \overset { } { \varphi _ { x } } } ;$   
7: $y _ { k } ^ { \prime } ( n ) = y _ { k } ( n ) / \varphi _ { y } ;$   
8: $\ddot { D _ { r e m a i n } ^ { \prime } } ( n ) = D _ { r e m a i n } ( n ) / { \varphi _ { D _ { r e m a i n } } } ;$   
9: $D _ { k } ^ { \prime } ( n ) = D _ { k } ( n ) / \varphi _ { D _ { k } } ;$   
10: return $\widehat { s } = \{ E _ { b a t t e r y } ^ { \prime } ( n ) , q ^ { \prime } ( n ) , P _ { 1 } ^ { \prime } ( n ) , . . . , P _ { K } ^ { \prime } ( n ) , D _ { r e m a i n } ^ { \prime } ( i ) , D _ { 1 } ^ { \prime } ( n ) , . . . , D _ { K } ^ { \prime } ( n ) \}$

## 4.4.2 Training of the MSE-PPO algorithm

Algorithm 2 represents the training process of the MSE-PPO algorithm, which is described as follows:

Algorithm 2 MSE-PPO algorithm.   
Input: Observation from UAV-enabled MEC environment:   
1: $s _ { t } = \{ E _ { b a t t e r y } , q , P _ { 1 } , . . . , P _ { K } , D _ { r e m a i n } , D _ { 1 } , . . . , D _ { K } \} ;$   
Output: Decision of the UAV: $a _ { t } = \{ \delta _ { k } , \alpha , v , R _ { k } \} ;$   
2:  Parameter initialization   
3: Initial actor network parameters $\theta ^ { 0 } ,$ , critic network parameter $\omega ^ { 0 } ;$   
4: for each episode: $1 , { \overset { \cdot } { 2 } } , . . . , N$ do   
5: Reset simulation parameters of the system and obtain initial   
observation state s1;   
6: for $e p o c h t { = } 1 , 2 , . . . , T$ do   
7: Normalize state s to ${ \widehat { s } } ;$   
8:  Action generation   
9: Obtain action distribution samples from hybrid actor network;   
10: UAV take selected action $a _ { t }$ in current environment and get   
reward r ;   
11: if the UAV moves out of the border, then   
12: Cancel this action;   
13: end if   
14:  Experience storage   
15: Collect $( s _ { t + 1 } , a _ { t } , r )$ and store it in Buffer;   
16: Updated the state $s _ { t } \gets s _ { t + 1 } ;$   
17: Use $r _ { t } + \gamma Q _ { \omega } ( s _ { t + 1 } , \pi _ { \theta } ( s _ { t + 1 } ) ) - Q _ { \omega } ( s _ { t } , a _ { t } )$ to compute   
advantage estimates $A _ { t } ;$   
18: After every L steps;   
19:  Parameter updating   
20: Update θ by a gradient method (Eqn.27) $L _ { t } ^ { h y b r i d + V F + S } ( \theta )$   
with K epochs with random minibatch size M $\leq \ N ( T - k )$ and   
learning rate lr ;   
21: Update ω by a gradient method (Eqn.30) $L _ { t } ^ { V F } ( \omega ) ;$   
22: $\theta _ { o l d }  \theta ;$   
23: end for   
24: end for

Initialization (lines 2) In the initial stage of training, the actor network and critic network parameters are randomly initialized.

Action generation (lines 3-12) After the training begins, we normalize the state observations and use them as inputs to the hybrid action network, and derive discrete and continuous action distributions. To explore the distribution, we sample actions and obtain information about GUs. The UAV then executes the policy and receives a reward. If the UAV flies out of the set area, an operation like this case is aborted and this processing is subject to C5 and C6.

Experience storage (lines 14-17) The experience of UAV can be represented as tuple $( s _ { t + 1 } , a _ { t } , r )$ and the new state $s _ { t } +$ 1 will be updated to the current state $s _ { t }$ at the next moment. After the UAV performs some operations, it will be stored in the buffer.

Parameter update (lines 19-21) After every L step, the parameters of the actor and critic networks are updated by the loss function, where in the optimization process, we randomly select M minibatch at time step t to optimize surrogate $L _ { t } ^ { h y b r i d + V F + S }$ . This improvement not only allows accurate Q-value evaluation for discrete and continuous actions, but also speeds up the whole training process. Finally, we replace the parameter $\theta _ { o l d }$ with θ and proceed to the next iteration.

## 4.5 Complexity analysis

We evaluate the algorithm efficiency of UAV-assisted GUs offloading computation through complexity analysis. The DNNs of J -th layer actor and F -th layer critic realizes the nonlinear mapping between state and action. The time complexity of MSE-PPO algorithm (denoted as $T _ { c } )$ is

$$
\begin{array} { r } { T _ { c } = 4 \times \underset { j = 1 } { \overset { J } { \sum } } u _ { a c t o r , j } \cdot u _ { a c t o r , j + 1 } + \underset { f = 1 } { \overset { F } { \sum } } u _ { c r i t i c , f } \cdot u _ { c r i t i c , f + 1 } } \\ { = O ( \underset { j = 1 } { \overset { J } { \sum } } u _ { a c t o r , j } \cdot u _ { a c t o r , j + 1 } + \underset { f = 1 } { \overset { F } { \sum } } u _ { c r i t i c , f } \cdot u _ { c r i t i c , f + 1 } ) , } \end{array}\tag{34}
$$

where $u _ { a c t o r , j }$ represents the number of neurons in the j -th layer of actor network, and $u _ { c r i t i c , f }$ represents the number of neurons in the f -th layer of the critic network.

There is matrix ${ \cal P } \cdot { \cal Q }$ and bias Q in the fully connected layer. The number of storage units required by the fully connected neural network is (P + 1) · Q, therefore the space complexity is $O ( G )$ . In addition, it is necessary to allocate storage space for the buffer to store the information in the training process, and the space complexity is $O ( M _ { b } )$ . Therefore, the space complexity of MSE-PPO algorithm (denoted as $S _ { c } )$ is

$$
\begin{array} { r l } {  { S _ { c } = 2 \times \sum _ { j = 1 } ^ { J } ( u _ { a c t o r , j } + 1 ) \cdot u _ { a c t o r , j + 1 } } \quad } & { } \\ & { + 4 \times \sum _ { f = 1 } ^ { F } ( u _ { c r i t i c , f } + 1 ) \cdot u _ { c r i t i c , f + 1 } + M _ { b } } \\ & { = \underbrace { O ( \sum _ { j = 1 } ^ { J } u _ { a c t o r , j } \cdot u _ { a c t o r , j + 1 } + \sum _ { f = 1 } ^ { F } u _ { c r i t i c , f } \cdot u _ { c r i t i c , f + 1 } ) } _ { O ( G ) } + O ( M _ { b } ) . } \end{array}\tag{35}
$$

## 5 Results and analysis

In this section, we illustrate the RL-based task offloading framework of the UAV-enabled MEC system through numerical simulations. Then, the performance of the MSE-PPO-based framework is verified in different scenarios and compared with other baseline schemes. We implemented MSE-PPO algorithm based on the experimental platforms of Intel Corei9-11900H, NVIDIA GeForce RTX3090, and Tensorflow-CPU-1.12, respectively from the selection of the optimal value of the hyperparameter and the comparison of four baselines. The main simulation parameters for this environment are shown in Table 2.

Table 2 Simulation settings
<table><tr><td>Parameters</td><td>Values</td></tr><tr><td>Number of GUs (K)</td><td>{5,10,15}</td></tr><tr><td>Discount factoer (y)</td><td>0.9</td></tr><tr><td>Noise power  $( \sigma ^ { 2 } )$ </td><td>-90dBm</td></tr><tr><td>Channel bandwidth (B)</td><td>1MHz</td></tr><tr><td>CPU cycles (s)</td><td>1200cycles/bit</td></tr><tr><td>Minibatch (M)</td><td>256</td></tr><tr><td>Service cycle (T)</td><td>600s</td></tr><tr><td>Time slot number (N)</td><td>30</td></tr><tr><td>Maximum energy of UAV (Emax)</td><td>500 kJ</td></tr><tr><td>The parameter of path loss model (a,b)</td><td>4.8,0.33</td></tr><tr><td>Additional path loss for LoS and NLoS  $( n _ { L o S } , n _ { L o S } , )$ </td><td>1.6,2.1</td></tr><tr><td>UAV CPU processing power  $\left( f _ { U A V } \right)$ </td><td>0.6GHz</td></tr><tr><td>UAV hover altitude (H)</td><td>100m</td></tr><tr><td>UAV flight power  $( P _ { f } )$ </td><td>0.3 Watt</td></tr><tr><td>Maximal horizontal coverage radius (Rmax)</td><td>15m</td></tr><tr><td>UAV maximum flight speed (Umax)</td><td>10 m/s</td></tr><tr><td>Buffer size(Mb)</td><td>30000</td></tr></table>

Table 3 Parametric settings
<table><tr><td></td><td colspan="8">Weight coefficient  $\left( c _ { 3 } \right)$ </td></tr><tr><td>Indicator</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td></tr><tr><td> $\overline { { r } } _ { k } ^ { \prime }$ </td><td>0.74</td><td>0.65</td><td>0.77</td><td>0.84</td><td>0.87</td><td>0.90</td><td>0.91</td><td>0.76</td><td>0.80</td></tr><tr><td> ${ \overline { { f } } } ^ { s \prime }$ </td><td>0.77</td><td>0.78</td><td>0.78</td><td>0.75</td><td>0.80</td><td>0.74</td><td>0.82</td><td>0.76</td><td>0.70</td></tr><tr><td> $\overline { { e } } _ { t a t a l } ^ { \prime }$ </td><td>0.75</td><td>0.76</td><td>0.74</td><td>0.79</td><td>0.83</td><td>0.87</td><td>0.74</td><td>0.76</td><td>0.81</td></tr></table>

For comparison purposes, the four baseline methods are described as follows.

• Local-only: In the case that the UAV does not carry the MEC, all computational tasks generated by the GU are performed locally.

• Offload-only: We adopt the traditional traveler approach-Traveling Salesman Problem (TSP) as the full offloading computation services mechanism, where the GU offloads all its tasks to the UAV that cruises periodically in a fixed region [45].

• PPO: The PPO algorithm, which is based on stochastic policy gradient and TRPO as described in the literature [40], uses a trust region for exploration in the continuous action space of the UAV. The action noise variance is a trainable vector that is not output by the network, which enhances its robustness during the training process. Thus, it also serves as a starting point for our design algorithm in dealing with continuous action and discrete action spaces.

DDPG: The deterministic gradient algorithm DDPG in the literature [35] is used as a benchmark method, which is used to design flight planning and task offloading decisions for the UAV, and to increase its good performance, Ornstein Uhlenbeck (OU)-noise is added during the training process to increase the exploration space to avoid falling into local optimization.

## 5.1 Parametric analysis

We start with a comparison experiment to determine the best value of the important hyperparameter $c _ { 3 }$ used in the MSE-PPO algorithm. By investigating the literature [40] we found that when the loss coefficient $c _ { 1 }$ or the entropy reward coefficient c2 in (Eqn.30) are too large, then they will dominate the loss function. When their weight is too small, they will be meaningless in the action space at this moment. Therefore, according to the above literature, we take the values of $c _ { 1 }$ and $c _ { 2 }$ in this experiment as 0.1 and 0.01, respectively. Similarly, the hybrid probability ratio of the weight coefficient $c _ { 3 }$ for continuous and discrete actions plays a very important role in (Eqn.32). The unbalanced combination of action decision probability ratio will lead to unstable strategies and low computational efficiency. In our scenario, three average normalized indexes after training were compared through parameter adjustment, i.e., transmission rate, overall energy consumption and service equity index. Table 3 shows that the MSE-PPO algorithm we implement performs well with the choice of $c _ { 3 } = 0 . 7$

## 5.2 Performance analysis

In this section, we evaluate the MSE-PPO algorithm by contrasting its cumulative rewards and performance metrics (such as average delay, average transmission rate, average offloading ratio, overall energy consumption, and average fairness) with those of other baseline algorithms.

## 5.2.1 Comparison of training rewards

Figure 4 shows the cumulative rewards obtained by UAV after 2000 training rounds of iterations between different algorithms, since the two non-RL algorithms i.e., Localonly and Offload-only, do not change with training, so no comparative analysis is performed. Initially during the time segment, the GU is away from the UAV, allowing local task computations to be performed exclusively, thus significantly escalating computational energy expenditure. Moreover, the channel correlation between the UAV and the GU may not be optimal, leading to fewer and more erratic rewards in each round. However, as the training progresses, the UAV continues to fly closer to the GU, mastering improved flight trajectories and offloading strategies, gradually increasing the cumulative rewards achieved by the UAV. And in later training cycles, the curve can reach satiation at about 300, indicating a less consistent cumulative reward, but converges faster than the baseline algorithm.

<!-- image-->  
Fig. 4 The parallel architecture of the MSE-PPO algorithm

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)  
Fig. 5 (a) Average delay of different algorithms with D = 120 Mbit. (b) Average delay with different task sizes. (c) Average transmission rate performance of different algorithms with k = 10

## 5.2.2 Performance evaluation of MSE-PPO algorithm

Figure 5 shows the comparison of the training results of different algorithms regarding the average delay and average transmission rate for the same and different task volumes, respectively.

• Figure 5a shows the average latency training curves for different algorithms with a total task of D = 120 Mbit in the service cycle. The average latency of Offloadonly single offload decision for Local-only and TSP does not converge with training. In contrast, Q-based update DDPG has the dual Critic network and Target network structure, which can eliminate historical data correlation to find the optimal policy action. The policy gradient-based PPO with Critic network estimator can adjust the action probability according to the trajectory return in buffer, therefore it has better performance in terms of average delay. The MSE-PPO algorithm encompassing a parallel actor network and a critic network architecture possesses a marked advantage in offloading decisions, substantially reducing the average delay by 33.55%, 33.41%, and 20.31% in comparison to TSP, DDPG, and PPO, respectively.

• Figure 5b shows the average latency comparison of different algorithms after 2000 rounds of training with the same total task size. The MSE-PPO algorithm consistently has the lowest average latency among several algorithms. The Local-only and Offload-only latencies are higher compared to RL. Due to the exploration of discrete and continuous non-negligible spaces, DDPG and

PPO cannot obtain the best action strategy accurately. And MSE-PPO has a significantly slower processing latency growth than other algorithms as the total task size increases.

• Figure 5c shows the average transmission rate training curves of different algorithms for k = 10. TSP uses an iterative range GU and builds the communication computation service vertically, which does not increase the transmission rate with training. Other RL-based algorithms adjust the flight and offloading strategies with iterations, which means more states can be explored. The average transmission rate of MSE-PPO is 13.24 Mbit/s after 2000 rounds of training, which is 41.45%, 38.78%, and 7.55% higher than that of TSP, DDPG, and PPO, respectively.

Figure 6 shows the comparison of the average latency and average offloading rate of the MSE-PPO algorithm for GU with different computing power, and the comparison of the average latency of different algorithms for GU with the same computing power.

• Figure 6a shows the average latency training curve of MSE-PPO algorithm when the computing power of the GU is 0.2GHz, 0.4GHz, 0.6GHz, and 0.8GHz, respectively. The system offloads fewer tasks when the GU has a high computing capacity, and the GU prefers to execute tasks locally. The lower the computing capacity of the GU, and the slower the data processing speed of the system, the longer the maximum delay between local execution and offloading. After 2000 rounds of training, the average latency of the four different computing capabilities converge to 9.49s, 8.22s, 6.71s, and 5.75s, respectively.

• Figure 6b shows the task offloading rate performance comparison of the MSE-PPO algorithm when the GU

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)  
Fig. 6 (a) Average delay of MSE-PPO under different computing capabilities of GU. (b) Average offloading ratio of MSE-PPO under different computing capabilities of GU. (c) Comparison of average delay under different algorithms

CPU computing power is 0.2 GHz, 0.4 GHz, 0.6 GHz, and 0.8 GHz, respectively. The average offloading rate of the UAV-enabled MEC system is smaller when the GU computing power is strong, therefore the GU prefers to perform task computation locally. Similarly, when the computing power is smaller, the task computation delay and task offloading rate increase.

• Figure 6c shows the comparison of the optimized delay between the proposed scheme and other algorithms under the same CPU computing power of the GU. We can see that as the CPU computing power of the GU increases, the average system latency decreases and the proposed MSE-PPO scheme achieves the lowest average latency. This is due to the fact that the scheme can find the most favorable influence on the UAV-enabled MEC system in a dynamic environment, which is the task offloading rate.

Figure 7 shows the overall system energy consumption, fairness comparison and fairness performance of each algorithm for different number of GUs when the UAV provides offloading computation services at different time periods.

• Figure 7a shows a comparison of the overall system energy consumption of different algorithms over the service cycle. The energy cost increases with service time, and the MSE-PPO algorithm has better performance than other baseline algorithms with a relatively slow growth rate of overall energy consumption. Overall energy consumption is reduced by 29.61%, 20.44%, 10.80%, and 6.68% compared to Local-only, TSP, DDPG, and PPO, respectively.

• Figure 7b shows a comparison of the fairness obtained by different algorithms for UAV in serving GUs with offloading computation. It is clear that fairness increases with service time, and the MSE-PPO algorithm provides the highest degree of fairness by employing a parameterized hybrid actor spatial action selection strategy during its training phase. This strategy involves randomly sampling discrete offloading decision actions from softmax( f ) and generating continuous flight and offloading rate actions through the mean and variance of the Gaussian distribution produced by the sub-actor network. A parameterized hybrid loss function is used to speed up the update process, therefore that UAV exploration of more states is rewarded with higher fairness.

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)  
Fig. 7 (a) Comparison of overall energy consumption under different algorithms. (b) Comparison of average fairness under different number of time slots. (c) Comparison of average fairness under different number of GUs

Figure 7c verifies the impact of different number of GUs on the performance of the UAV-enabled MEC system. We can observe that the fairness of all four methods decreases as the number of GUs increases, which is due to the fact that a UAV provides limited offloading computation services and must return when the UAV is about to run out of power. The MSE-PPO algorithm achieves higher fairness than the other three baseline algorithms. This is because our hybrid loss function approach improves fairness by minimizing the clipped surrogate objective for discrete and continuous policies separately and updating them accordingly. This allows more offloading tasks to be computed during the hybrid action space selection process.

## 6 Discussion

This section aims to discuss the inherent difficulties encountered in the current paper. The paper explores the multifaceted challenges in this field: (1) We model the challenging problem of single UAV-enabled MEC as a complex and non-convex optimization problem. This problem shows the characteristics of being NP-hard. Therefore, it poses a significant challenge in finding an analytical solution. Therefore, the use of a DRL algorithm is considered necessary to effectively address this problem. Although the DRL algorithm may not guarantee an optimal solution, it can effectively learn an approximate optimal solution through iterative trial-anderror. (2) We make the implicit assumption that the UAV is capable of consistently establishing a reliable wireless link with the GU for offloading task data. However, in some complex environments (such as urban scenarios), the wireless link may be vulnerable to interference, fading, or blockage. This phenomenon can significantly hinder the performance of edge computing, leading to an increase in communication overhead. These considerations have not been incorporated into our model and algorithm as comprehensively detailed in this paper.

## 7 Conclusions

In this paper, we studied the optimization problem of flight trajectory and task offloading decisions in a UAV-enabled MEC system, where the UAV provides communication and computation services to GUs. We have considered the fairness of all GU services, transmission communication between UAV and GUs, and overall system energy consumption, with the goal of maximizing computational efficiency and providing low latency and fair services for GUs. We aim to achieve efficient task offloading for UAV-assisted GUs under time-varying conditions. To do this, we design a reasonable reward function based on UAV and GU states, formulate the problem as an MDP, and propose the MSE-PPO algorithm based on proximal policy optimization. First, we normalized observations across different ranges and decomposed the AC-based hybrid action space into simpler sub-actor networks. Secondly, we have reconstructed its loss function to update the parameters of the policy network, and improve the training speed and decision-making ability of the UAV in the dynamic environment. Although this paper focuses mainly on the parameterized action space to achieve better learning in single UAV assisted offloading, more experiments are needed to test the performance of this architecture in multi-UAV cooperative offloading. In addition to this, multi-UAVs need to balance load fairness among UAVs in addition to considering the issue of GU offloading resource fairness, and we leave these studies for future work.

## 8 Future work

This section investigates the potential future prospects of this academic endeavor. In our forthcoming research endeavors, our objective is to address the aforementioned limitations and refine our approach in several pivotal domains:(1) We aim to incorporate prior knowledge or expert guidance within the DRL framework to expedite the learning process and enhance overall performance. (2) We shall consider more intricate wireless channel models that encompass interference, fading, and blocking phenomena. Furthermore, we shall investigate methodologies to optimize the communication parameters (e.g., transmit power, bandwidth allocation, and modulation scheme) between Unmanned Aerial Vehicles (UAVs) and Ground Users (GUs), with the aim of enhancing communication quality and minimizing communication overhead.

Acknowledgements This work is supported in part by the National Natural Science Foundation of China (62176088, 62303159), International Strategic Innovative Project of National Key Research & Development Program of China (2023YFE0112500), Key Project of Science and Technology Research of the Education Department of Henan Province (22A120001), China Postdoctoral Science Foundation Funded Project (2023M741008).

Author Contributions All authors contributed to the study conception and design. Material preparation, data collection and analysis were performed by Wei Li, Si Li, Huaguang Shi, Wenhao Yan and Yi Zhou. The first draft of the manuscript was written by Huaguang Shi and all authors commented on previous versions of the manuscript. All authors read and approved the final manuscript.

Data Availability The datasets generated during and/or analysed during the current study are not publicly available due to [REASON(S) WHY DATA ARE NOT PUBLIC] but are available from the corresponding author on reasonable request.

## Declarations

Conflict of Interest We declare that we have no financial and personal relationships with other people or organizations that can inappropriately influence our work, there is no professional or other personal interest of any nature or kind in any product, service and/or company that could be construed as influencing the position presented in, or the review of, the manuscript entitled, “UAV-enabled Fair Offloading for MEC Networks: A DRL Approach based on Actor-Critic Parallel Architecture”.

Ethical and informed consent for data used The data used did not involve human participants and animal studies.

## References

1. Cui J, Wei L, Zhong H, Zhang J, Xu Y, Liu L (2020) Edge computing in VANETs-An efficient and privacy-preserving cooperative downloading scheme. IEEE J Sel Areas Commun 38:1191– 1204

2. Sharma MK, Kumar M, Saini J (2021) Design and analysis of a compact UWB-MIMO antenna with improved isolation for UWB/WLAN applications. Wirel Pers Commun 119:2913–2928

3. Senthilkumar G, Tamilarasi K, Kaviarasan S, Arun M (2022) Trusty authentication of devices using blockchain-cloud of things (B-CoT) for fulfilling commercial services. Int J Syst Assur Eng 1–11

4. Anwar MR, Wang S, Akram MF, Raza S, Mahmood S (2022) 5G-Enabled MEC: a distributed traffic steering for seamless service migration of internet of vehicles. IEEE Internet Things J 9:648– 661

5. Ju Y, Chen Y, Cao Z, Liu L, Pei Q, Xiao M, Ota K, Dong M, Leung V (2023) Joint secure offloading and resource allocation for vehicular edge computing network: a multi-agent deep reinforcement learning approach. IEEE Trans Intell Transp Syst , in press

6. Huang S, Zhang J, Wu Y (2022) Altitude optimization and task allocation of UAV-Assisted MEC communication system. Sensors 22:8061

7. Liu Y, Xiong K, Ni Q, Fan P, Letaief KB (2020) UAV-Assisted wireless powered cooperative mobile edge computing: joint offloading, CPU control, and trajectory optimization. IEEE Internet Things J 7:2777–2790

8. Li H, Wu S, Jiao J, Lin XH, Zhang N, Zhang Q (2023) Energyefficient task offloading of edge-aided maritime UAV systems. IEEE Trans Veh Technol 72:1116–1126

9. Zhang K, Gui X, Ren D, Li D (2021) Energy-latency tradeoff for computation offloading in uav-assisted multiaccess edge computing system. IEEE Internet Things J 8:6709–6719

10. Zhang P, Wang C, Jiang C, Benslimane A (2021) UAV-Assisted multi-access edge computing: technologies and challenges. IEEE Internet Things Mag 4:12–17

11. Akbari M, Syed A, Kennedy W, Erol-Kantarci M (2023) Constrained federated learning for AoI-limited SFC in UAV-Aided MEC for smart agriculture. IEEE Trans Mach Learn Commun Network 1:277–295

12. Zhang T, Xu Y, Loo J, Yang D, Xiao L (2020) Joint computation and communication design for UAV-assisted mobile edge computing in IoT. IEEE Trans Ind Inform 16:5505–5516

13. Li Q, Shi L, Zhang Z, Zheng G (2023) Resource Allocation in UAVenabled wireless-powered MEC networks with hybrid passive and active communications. IEEE Internet Things J 10:2574–2588

14. Ye W, Luo J, Shan F, Wu W, Yang M (2020) Offspeeding: optimal energy-efficient flight speed scheduling for UAV-assisted edge computing. Comput. Networks 183:107577

15. Zhang T, Xu Y, Loo JK, Yang D, Xiao L (2019) Joint computation and communication design for UAV-assisted mobile edge computing in IoT. IEEE Trans Industr Inform 16:5505–5516

16. Sun H, Zhou F, Hu RQ (2019) Joint offloading and computation energy efficiency maximization in a mobile edge computing system. IEEE Trans Veh Technol 68:3052–3056

17. Lin X, Bi S, Cheng N, Dai M, Wang H (2022) An α-fairness approach to balancing the energy consumption among sensors for UAV-IoT systems. IEEE Internet Things J 9:17965–17978

18. Ding Z, Xu D, Schober R, Poor HV (2022) Hybrid NOMA offloading in multi-user MEC networks. IEEE Trans Wirel Commun 21:5377–5391

19. Chi HR, Radwan A (2023) Fully-decentralized fairness-aware federated MEC small-cell peer-offloading for enterprise management networks. IEEE Trans Industr Inform 19:644–652

20. Ren D, Gui X, Zhang K (2022) Adaptive request scheduling and service caching for MEC-assisted IoT networks: an online learning approach. IEEE Internet Things J 9:17372–17386

21. Sun Y, Xu J, Cui S (2022) User association and resource allocation for MEC-enabled IoT networks. IEEE Trans Wirel Commun 21:8051–8062

22. Liu M, Feng G, Sun Y, Chen N, Tan W (2023) A network function parallelism-enabled MEC framework for supporting low-latency services. IEEE Trans Serv Comput 16:40–52

23. Lekharu A, Jain M, Sur A, Sarkar A (2022) Deep learning model for content aware caching at MEC servers. IEEE Trans Netw Service Manag 19:1413–1425

24. Hua M, Yang L, Pan C, Nallanathan A (2019) Throughput maximization for full-duplex UAV aided small cell wireless systems. IEEE Wireless Commun Lett 9:475–479

25. Wu Q, Zeng Y, Zhang R (2017) Joint trajectory and communication design for multi-UAV enabled wireless networks. IEEE Trans Wirel Commun 17:2109–2121

26. Zhan C, Zeng Y, Zhang R (2018) Energy-efficient data collection in UAV enabled wireless sensor network. IEEE Wireless Commun Lett 7:328–331

27. Wu Q, Zhang R (2018) Common throughput maximization in UAVenabled OFDMA systems with delay consideration. IEEE Trans Commun 66:6614–6627

28. Wang Q, Gao A, Hu Y (2021) Joint power and QoE optimization scheme for multi-UAV assisted offloading in mobile computing. IEEE Access 9:21206–21217

29. Zhong X, Huo Y, Dong X, Liang Z (2020) QoS-compliant 3-D deployment optimization strategy for UAV base stations. IEEE Syst J 15:1795–1803

30. Xu Y, Zhang T, Liu Y et al (2022) Cellular-connected multi-UAV MEC networks: an online stochastic optimization approach. IEEE Trans Commun 70:6630–6647

31. Gu B, Zhang X, Lin Z, Alazab M (2021) Deep multi-agent reinforcement-learning-based resource allocation for internet of controllable things. IEEE Internet Things J 8:3066–3074

32. Li J, Yi C, Chen J, Zhu K, Cai J (2023) Joint trajectory planning, application placement, and energy renewal for UAV-assisted MEC: A triple-learner-based approach. IEEE Internet Things J 10:13622– 13636

33. Song F et al (2023) Evolutionary multi-objective reinforcement learning based trajectory control and task offloading in UAVassisted mobile edge computing. IEEE Trans Mobile Comput 22:7387–7405

34. Nie Y, Zhao J, Gao F, Yu FR (2021) Semi-distributed resource management in UAV-aided MEC systems: a multi-agent federated reinforcement learning approach. IEEE Trans Veh Technol 70:13162–13173

35. Wang Y, Fang W, Ding Y, Xiong NN (2021) Computation offloading optimization for UAV-assisted mobile edge computing: a deep deterministic policy gradient approach. Wirel Networks 27:2991– 3006

36. Zhou X, Huang L, Ye T, Sun W (2022) Computation bits maximization in UAV-assisted MEC networks with fairness constraint. IEEE Internet Things J 9:20997–21009

37. Mnih V, Kavukcuoglu K, Silver D, Rusu A, Veness J, Bellemare MG et al (2015) Human-level control through deep reinforcement learning. Nat 518:529–533

38. Tuli S, Ilager S, Ramamohanarao K, Buyya R (2022) Dynamic scheduling for stochastic edge-cloud computing environments using A3C learning and residual recurrent neural networks. IEEE Trans Mobile Comput 21:940–954

39. Lu S, Liu S, Zhu Y, Liang W, Li K, Lu Y (2023) A DRL-based decentralized computation offloading method: an example of an intelligent manufacturing scenario. IEEE Trans Industr Inform 19:9631–9641

40. Chen Z, Yin B, Zhu H, Li Y, Tao M, Zhang W (2022) Mobile communications, computing, and caching resources allocation for diverse services via multi-objective proximal policy optimization. IEEE Trans Commun 70:4498–4512

41. Ho Tai, Nguyen K, Cheriet M (2021) UAV control for wireless service provisioning in critical demand areas: a deep reinforcement learning approach. IEEE Trans Veh Technol 70:7138–7152

42. Mao K, Zhu Q, Qiu Y, Liu X, Song M, Fan W, Kokkeler A, Miao Y (2023) A UAV-aided real-time channel sounder for highly dynamic nonstationary A2G scenarios. IEEE Trans Instrum Meas 72:1–15

43. Liu C, Dai Z, Zhao Y, Crowcroft J, Wu D, Leung K (2021) Distributed and energy-efficient mobile crowdsensing with charging stations by deep reinforcement learning. IEEE Trans. Mobile Comput 20:130–146

44. Zhao M, Li W, Bao L, Luo J, He Z, Liu D (2021) Fairnessaware task scheduling and resource allocation in UAV-enabled mobile edge computing networks. IEEE Trans Green Commun Netw 5:2174–2187

45. Wang Y, Chen M, Pan C, Wang K, Pan Y (2022) Joint optimization of UAV trajectory and sensor uploading powers for UAV-assisted data collection in wireless sensor networks. IEEE Internet Things J 9:11214–11226

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

<!-- image-->

Wei Li received his Ph.D. degree in Cartography and GIS from Henan University in 2022. She is currently an associate professor at the School of Artificial Intelligence, Henan University. His research interests include cooperative control of multi-intelligent body systems, vehicular edge computing and intelligent transportation systems.

<!-- image-->  
Si Li is currently pursuing the M.S. degree with Henan University, Zhengzhou, China. His research interests include UAV-assisted edge computing, and multi-agent collaboration control.

<!-- image-->  
and multiagent learning.

Huaguang Shi received the B.S. degree in electronic science and technology from Zhengzhou University, Zhengzhou, China, in 2014, and the Ph.D. degree in measurement technique and automation equipment from the University of Chinese Academy of Sciences, Beijing, China, in 2021. He is currently a Lecturer with the School of Artificial Intelligence, Henan University, Zhengzhou. His current research interests include industrial Internet of Things, wireless networks,

<!-- image-->  
Wenhao Yan is currently pursuing the M.S. degree with Henan University, Zhengzhou, China. His research interests include UAV-assisted data collection, and air-to-ground cooperative control.

<!-- image-->

Yi Zhou received the B.S. degree in electronic engineering from the First Aeronautic Institute of Air Force, Changchun, China, in 2002, and the Ph.D. degree in control system and theory from Tongji University, Shanghai, China, in 2011. He is currently a Full Professor and the Deputy Dean of the School of Artificial Intelligence, Henan University, Kaifeng, China, and also the Director of the International Joint Research Laboratory for Cooperative Vehicular Networks. His

research interests include vehicular cyber-physical systems and multiagent collaboration.