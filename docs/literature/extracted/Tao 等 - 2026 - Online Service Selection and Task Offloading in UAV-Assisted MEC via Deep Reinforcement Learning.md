# Online Service Selection and Task Offloading in UAV-Assisted MEC via Deep Reinforcement Learning

Youwei Tao, Zhufang Kuang , Member, IEEE, Fen Hou , Member, IEEE, and Anfeng Liu

Abstract— Unmanned Aerial Vehicle (UAV)-assisted Mobile Edge Computing (MEC) combines UAV technology and MEC paradigms. However, it is challenging to guarantee the User Terminals’ (UTs) Quality of Experience (QoE) simply by offloading tasks to the UAV for assisted computation. This paper studies a UAV-assisted heterogeneous edge computing network, where a cache-enabled UAV dynamically responds to online service requests from MEC servers. The QoE is defined as a weighted sum of latency reduction and energy savings at UTs. To maximize UAV energy efficiency, a joint optimization problem is formulated, involving online service selection, task offloading, CPU frequency allocation, UAV trajectory, and hovering scheduling. The problem is modeled as a multi-stage stochastic mixedinteger nonlinear program. A hierarchical framework combining Deep Reinforcement Learning (DRL) and Lyapunov optimization is proposed. The Lyapunov method analyzes task queues and balances queue lengths with utility, while Deep Deterministic Policy Gradient (DDPG) derives online services decision policies. UAV trajectory and hovering are further optimized based on the decisions. Simulation results show that the proposed method significantly outperforms benchmark schemes in terms of UAV energy efficiency.

Index Terms— Online service selection, heterogeneous networks, Lyapunov optimization, deep reinforcement learning, UAV.

## I. INTRODUCTION

WITH the advancement of network infrastructure, hetero-geneous networks are becoming increasingly prevalent. geneous networks are becoming increasingly prevalent. Addressing the diverse requests from UTs within heterogeneous networks is consistently recognized as a challenge [1].

In heterogeneous networks, UAV-assisted computation plays a role in addressing the requests, due to the high mobility and line-of-sight (LoS) links [2], [3]. However, as the complexity and volume of requests grow, relying solely on UAV for simple auxiliary computation fails to meet the QoE for UTs [4]. To improve QoE, frequently accessed services are placed at the MEC server. This maximizes the overall efficiency and performance of the heterogeneous network [5].

Currently, many computationally intensive applications are enabled by the fifth-Generation (5G) networks, such as intelligent recognition and Virtual Reality (VR) [6]. By placing the corresponding services on MEC servers, latency and energy consumption are reduced through the offloading of these complex application tasks to MEC, with the aim of maximizing the QoE of UTs [7], [8], [9], [10]. However, unlike cloud infrastructures, MEC servers are constrained by storage limitations, preventing them from placing all services [11]. In heterogeneous edge networks with cloud infrastructures, the problem of how to choose the best services to place on the MEC server is a challenge [12]. However, in extreme network environments lacking cloud facilities [13], the UAV-assisted communication is considered crucial for delivering the requested services. Some studies on UAV-assisted heterogeneous edge networks emphasize the optimization of the UAV trajectory, communication, and energy [14], but the problem of online service selections has not been considered. Moreover, uncertainties in the heterogeneous network, such as diverse service demands and varying task arrival rates, are critical factors influencing online service selection.

The service content is determined in advance by offline service placement in a network environment [15]. However, offline service placement schemes lack adaptability to dynamic and time-varying task demands. In contrast, online service selection offers significant advantages in handling complex and heterogeneous network environments, where user terminals generate diverse and unpredictable requests. By responding to the random arrival of tasks and varying service demands, online methods overcome the inherent rigidity of traditional offline approaches. Furthermore, in resourceconstrained MEC servers, services can be adaptively selected based on the characteristics of incoming tasks. This flexibility enables more efficient resource utilization and enhances the QoE for user terminals [16].

TABLE I  
COMPARISON OF EXISTING WORK ON HETEROGENEOUS EDGE NETWORKS
<table><tr><td rowspan=1 colspan=1>Work</td><td rowspan=1 colspan=1>Objective</td><td rowspan=1 colspan=1>Offloading Decision</td><td rowspan=1 colspan=1>Resources Allocation</td><td rowspan=1 colspan=1>service Caching</td><td rowspan=1 colspan=1>Online Service Selection</td><td rowspan=1 colspan=1>Optimization Trajectory</td></tr><tr><td rowspan=1 colspan=1>[7]</td><td rowspan=1 colspan=1>QoEofUTs</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[19]</td><td rowspan=1 colspan=1>Energy Consumption</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>[22]</td><td rowspan=1 colspan=1>Number of Offloading Tasks</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[24]</td><td rowspan=1 colspan=1>EnergyEfficiency of the UAV</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>[26]</td><td rowspan=1 colspan=1>EnergyEfficiency of the UAV</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>[33]</td><td rowspan=1 colspan=1>Total Delay</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[35]</td><td rowspan=1 colspan=1>Service Experience Ratio</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>（</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Ourwork</td><td rowspan=1 colspan=1>EnergyEfficiency of theUAV</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td></tr></table>

Various scenarios in UAV-assisted heterogeneous edge networks are studied. In these scenarios, objectives such as minimizing system energy consumption and processing delay, or maximizing QoE and UAV’s energy efficiency are considered. However, the optimization problem aimed at maximizing energy efficiency of the UAV by jointly considering online service selection, task offloading, CPU frequency allocation, UAV trajectory, and hover scheduling deserves to be considered. To address the dynamic and uncertain service requests and task arrivals in resource-constrained MEC environments, our study introduces an online service selection strategy. Unlike traditional offline caching methods, this approach enables realtime adaptability, ensuring timely response and improved QoE for UTs in UAV-assisted networks. This flexibility is especially crucial in extreme scenarios where cloud infrastructure is unavailable. The contributions of this paper are as follows:

• We investigate the online service selection and task offloading problem in UAV-assisted heterogeneous edge networks, aiming to maximum energy efficiency of the UAV. The joint optimization of online service selection, task offloading, CPU frequency allocation, UAV trajectory and hovering schedule is formulated as a multi-stage mixed-integer nonlinear programming problem.

• We divide the problem into two subproblems, and an iterative method framework is proposed based on Lyapunov theory and Deep Reinforcement Learning (Ly-DRL). This framework combines the advantages of two approaches, offering good convergence for solving problems with mixed discrete and continuous variables, and enables queue length analysis.

• We propose an online service selection algorithm based on Lyapunov optimization and DDPG (LyOSS-PG), which extends the Ly-DRL framework. In the first subproblem, Lyapunov theory is employed for queue stability analysis, while the DDPG algorithm jointly optimizes online service selection, task offloading, and CPU frequency allocation. In the second subproblem, given the service selection decisions, UAV trajectory planning and hovering scheduling are jointly optimized using a separate DDPG agent.

The remainder of the paper is organized as follows. In Section II, the related work is discussed. The network model and problem are formulated in Section III. An online algorithm for solving the problem is proposed in Section IV. The simulation results are shown in Section V, and the conclusions are summarized in Section VI.

## II. RELATED WORK

In this section, existing studies are summarized, including the optimization goal of processing latency, energy consumption, energy efficiency, QoE of UTs and UAV trajectory, etc. in heterogeneous edge networks.

Various UAV-assisted heterogeneous edge networks are investigated to address the complexities of heterogeneous networks. In [17], a UAV and D2D-assisted MEC communication scenario is presented, and an optimization algorithm based on block descent and potential game is proposed to minimize task processing latency. In [18], a UAV-assisted MEC network is considered, and an optimization algorithm is proposed to minimize long-term energy consumption while maintaining task backlog queue stability for heterogeneous UTs’ demands. In [19], a UAV-assisted edge computing scenario is studied, and an iterative method based on the block coordinate descent is proposed to jointly solve the task offloading and UAV trajectory, aiming to reduce energy consumption.

In [20], a 3D urban environment is considered, and a DRL solution based on multi-step DDQN is proposed to jointly optimize UAV trajectory and communication scheduling, with the goal of minimizing the total task processing delay. In [21], a 3D UAV-assisted communication scenario within MEC networks is studied, where a DRL-based algorithm is developed to optimize UAV trajectories for reducing task execution latency. Furthermore, [22] investigates a multi-UAVassisted MEC system and proposes a two-layer optimization framework that aims to maximize the number of offloaded tasks while ensuring task success rates under both energy and latency constraints.

In the above studies, only energy consumption of system and processing latency are considered as objectives, while energy efficiency of the UAV is not taken into account. In [23], to maximize energy efficiency of the UAV, a UAV-assisted IoT system scenario is presented, and a DRL-based method is proposed to optimize UAV trajectory and bandwidth allocation. In [24], a UAV-assisted NOMA-MEC communication network is studied, and an iterative optimization algorithm is proposed to maximize energy efficiency by jointly optimizing communication scheduling and UAV trajectory. Similarly, in [25], a heterogeneous UAV-assisted task offloading scheme is explored, where a hybrid algorithm combining DRL and federated reinforcement learning is proposed to maximize the average energy efficiency across all UAVs. In [26], the problem of energy-efficient scheduling in a multi-UAV-assisted MEC system is investigated, and a novel triple-learner-based reinforcement learning approach is introduced, aiming to optimize the long-term energy efficiency of all UAVs.

Although the aforementioned studies consider objectives like UAV energy efficiency, system energy consumption, and delay of task processing, none of them investigate service placement. However, considering service placement is regarded as an effective solution to improve the QoE of UTs [27]. Given the storage constraints in the heterogeneous edge networks, the optimization of service selection strategies is considered by researchers. In [28], a hierarchical scalable video service placement system is studied. UAVs are employed as placement terminals to reduce video access latency for UTs. In [29], the adaptive cache partitioning and placement strategy are proposed for MEC networks to reduce service access costs, using a two-layer placement model.

In [30], a multi-user cache-assisted MEC scenario with location-aware user terminal preferences is considered, where a joint optimization strategy for service placement and bandwidth allocation is proposed to minimize energy consumption. In [31], a competitive MEC environment involving multiple service providers is studied, and a game-theoretic mechanism for service placement with virtual machine (VM) sharing is developed to reduce the overall economic cost. Furthermore, [32] introduces a dual time-scale optimization framework that jointly addresses adaptive request scheduling and collaborative service selection in MEC networks, aiming to minimize processing latency.

In the studies mentioned above, the system performance is improved through service placement. However, to address the requests of heterogeneous networks, online service selection need to be considered. In [33], online service placement and computation resource allocation in edge computing is investigated. An optimization method based on Deep Q Networks (DQN) with the goal of reducing task latency is proposed. In [34], the study of online service placement and task scheduling is considered. A joint optimization scheme through spatial-temporal collaboration is presented to minimize the overall cost of the system under communication interference. In [35], a UAV-assisted MEC scenario is studied, and a cooperative placement framework is proposed. The goal is to maximize the QoE of UTs while optimizing task offloading and trajectory planning. Yet, this framework is unable to meet sudden service requests and does not include queue analysis. In [36], a scenario of computation offloading in dynamic MEC networks is conceived, and an online offloading algorithm is presented, which combines Lyapunov optimization and DRL framework. Although this is a task online offloading algorithm, it has the potential to be extended to solve the problem of online service selection in UAV-assisted heterogeneous edge networks. Compared with previous studies, as summarized in Table I, our work proposes a unified framework that integrates online service selection, dynamic task offloading, and UAV trajectory optimization. This joint formulation addresses both the service caching challenge and energy-constrained UAV operations under stochastic demands. The novelty lies in integrating Lyapunov optimization for queue stability with DDPG-based decision-making, enabling real-time adaptation in MEC systems with high-dimensional state and action spaces.

<!-- image-->  
Fig. 1. Network model.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

In this section, the system model is described, including the network model, the task processing model, the task queuing model and the UAV communication model. Subsequently, the problem of jointly optimizing online service selections, task offloading, CPU frequency allocation, UAV trajectory and hovering schedule is formulated, aiming at the energy efficiency of the UAV while ensuring the high QoE of UTs. In this paper, online service selection refers to a real-time decision-making process where the service assignment is dynamically determined in each time slot based solely on the currently observed system state, including incoming task requests, resource availability, and network conditions. Unlike offline approaches, which assume prior knowledge of all future tasks and states to compute fixed service allocations, the online scheme operates without such foresight and continuously adapts to changes as they occur.

## A. Network Model

Fig. 1 shows a UAV-assisted heterogeneous edge network. Specifically, the network contains Q clusters denoted as $\mathcal { Q } ~ = ~ \{ 1 , 2 , \dots , q , \dots , Q \}$ and one UAV, where the q-th cluster has an MEC server and $I _ { q }$ UTs denoted as $\begin{array} { l } { { \mathcal { T } } _ { q } } \end{array} =$ $\{ 1 , 2 , \ldots , i , \ldots , I _ { q } \}$ . In the q-th cluster, different types of tasks generated by UTs arrive at the MEC server. The tasks generated by UT need to be handled by special service programs, while MEC server may not cache such services. As a service provider, UAV provides the required service programs for ground MEC to handle some UT application tasks. The UAV with caching functionality, flies to each UT cluster and places the service programs on the MEC server. The MEC hosts these services and allocates a portion of CPU frequency to handle types of tasks. According to characteristics of tasks, types of tasks are denoted as $\mathcal { I } = \{ 1 , 2 , . . , j , . . , J \}$ The task generated by the UT is represented as a tuple $A _ { i , j , q } = ( L _ { i , j , q } , C _ { i , j , q } ) . A _ { i , j , q }$ indicates that in the q-th cluster, the task of j-type generated by the i-th UT. $L _ { i , j , q }$ denotes the data size of the task and $C _ { i , j , q }$ denotes the computational intensity of the task. The set of services cached by the UAV is expressed as $\mathcal { Y } = \{ 1 , 2 , \ldots , y , . . . Y \}$ . To represent the oneto-one correspondence between task of j-type and the service y, the mapping function is defined as $\phi : \mathcal { I }  \mathcal { Y } .$ where $Y _ { j } ~ = ~ \phi ( j ) , j ~ \in ~ \mathcal { I } , Y _ { j } ~ \in ~ \mathcal { V } _ { }$ , indicating that tasks of type j can only be handled by service program $Y _ { j } , \ L _ { y }$ denotes the data size of the service $Y _ { j } ,$ , corresponding to the choice of processing j-type task. $y _ { j }$ denotes the service selection decision, which is a binary variable, and $y _ { j }$ equals 1 means that the service $Y _ { j }$ is selected. Due to the limited computational and storage resources of MEC server, $M _ { q }$ denotes the maximum number of services that can be hosted at the q-th MEC server. $\rho _ { j , q }$ denotes the certain proportion of CPU frequency, which is allocated to the service $y _ { j }$ hosted at the $q \mathrm { . }$ -th MEC. In the network, the location of the q-th MEC server is denoted by $c _ { q } ^ { M E C } ~ = ~ ( x _ { q } ^ { M E C } , y _ { q } ^ { M E C } , 0 )$ and the coordinate of the UT is indicated as $\vec { c _ { i , j , q } ^ { U T } } ^ { \textbf { \forall } } = ( x _ { i , j , q } ^ { U T } , y _ { i , j , q } ^ { U T } , 0 )$ . The whole process which the UAV completes delivery of all cluster services is completed in period ${ \bar { T } } ^ { 0 }$

<!-- image-->  
Fig. 2. Time slot division.

Specifically, the period is discretized into unequal time frame $\mathcal { T } ~ = ~ \{ 1 , 2 , \ldots t , \ldots T \}$ . It is stipulated that in two adjacent time frames, the UAV arrives at hovering points of two ground clusters respectively. One time frame $t \in \tau$ consists of time slots $\mathcal { K } ~ = ~ \{ 1 , 2 , \ldots , k , \ldots K _ { t } \}$ , and the duration of time slot k is δ, referred to as the frame-timeslot structure in [37]. The time slot k of the time frame t is donated by $\delta _ { t } [ k ]$ . In the the q-th cluster, the number of time slot for UAV hovering is denoted by $\eta _ { q }$ . It is assumed that each time slot is sufficiently short, such that the UAV’s position can be considered quasi-static within a single slot. The total service time of the UAV in the q-th cluster comprises three components: hovering, service migration, and flying to the next cluster. The time slot structure is illustrated in Fig. 2. Notably, the UT, MEC, and UAV operate on independent time slot schedules. To maintain timeline synchronization, the start and end points of each process are explicitly marked.

## B. Task Processing Model

In the q-th cluster, an MEC server is deployed alongside $I _ { q }$ UTs. Each UT is capable of generating a type-j task $A _ { i , j , q } ,$ which can be processed locally but at the cost of considerable energy consumption and latency. Alternatively, the task $A _ { i , j , q }$ can be offloaded to the MEC server for execution. The offloading decision for a type-j task is denoted by a binary variable $x _ { i , j , q }$ , where $x _ { i , j , q } ~ = ~ 1$ indicates that the task is offloaded to the MEC, and $x _ { i , j , q } = 0$ implies local processing. When a task is generated, its data characteristics are first transmitted to the MEC server for pre-processing. To guide the offloading decision, an offloading gain function $G _ { i } ( \rho _ { j } )$ is defined, which quantifies the benefit of offloading the task to the MEC server. A positive value of $G _ { i } ( \rho _ { j } )$ indicates that offloading yields performance gain, while a non-positive value suggests that local processing is more advantageous.

When task $A _ { i , j , q }$ is computed locally, the processing time is denoted by

$$
T _ { i , j , q } ^ { L o c a l } = \frac { C _ { i , j , q } \times L _ { i , j , q } } { f _ { i , j , q } } , \quad \forall i \in \mathcal { I } , \ \forall j \in \mathcal { I } , \ \forall q \in \mathcal { Q } ,\tag{1}
$$

where $f _ { i , j , q }$ represents the CPU frequency of the UT. Energy consumption $E _ { i , j , q } ^ { L o c a l }$ from local computation can be denoted by

$$
E _ { i , j , q } ^ { L o c a l } = \kappa \times C _ { i , j , q } \times L _ { i , j , q } \times f _ { i , j , q } ^ { 2 } ,\tag{2}
$$

where κ is the energy coefficient of the UT.

Orthogonal channels are considered for offloading tasks to the MEC. The transmission rate $w _ { i , j , q } ^ { t r }$ at which the UT transfers task to the MEC server is expressed as

$$
w _ { i , j , q } ^ { t r } = { { B } _ { 1 } } { { \log } _ { 2 } } \left( 1 + \frac { { { \rho } _ { o } } p _ { i , j , q } ^ { t r } } { { \xi } ^ { 2 } \big | \big | c _ { i , j , q } ^ { U T } - c _ { q } ^ { M E C } \big | \big | ^ { 2 } } \right) ,\tag{3}
$$

where $B _ { 1 }$ is the channel bandwidth of ground, the Path Loss at a reference distance of 1 meter is denoted by $\rho _ { o } . \ p _ { i , j , q } ^ { t r }$ denotes the transmit power of the UT. The white noise power is expressed by $\xi ^ { 2 }$ and $\left| \left| c _ { i , j , q } ^ { U T } - c _ { q } ^ { M E C } \right| \right|$ denotes the distance of the UT from the q-th MEC server.

When a task is offloaded to the MEC server, the task transmission delay can be represented by

$$
T _ { i , j , q } ^ { T r } = \frac { L _ { i , j , q } } { w _ { i , j , q } ^ { t r } } .\tag{4}
$$

The processing time at the q-th MEC can be denoted as

$$
T _ { i , j , q } ^ { M E C } = \frac { C _ { i , j , q } \times L _ { i , j , q } } { \rho _ { j , q } \times f ^ { m a x } } ,\tag{5}
$$

where $f ^ { m a x }$ is the maximum CPU frequency of the MEC server. Since the unloading gain of the UT is the focus, the energy consumption of the MEC is ignored as in the [7].

$$
E _ { i , j , q } ^ { M E C } = p _ { i , j , q } ^ { t r } \times T _ { i , j , q } ^ { T r } .\tag{6}
$$

## C. Task Queuing Model

In the q-th cluster, tasks generated by UTs are arrived at the MEC randomly. $O _ { j } [ k ]$ denotes the number of tasks of type $j$ generated by all UTs in time slot $k ,$ which are independently distributed random variables at each time slot. $o _ { j } [ k ]$ denotes the number of tasks of type j arriving at the MEC in time slot k, where $0 \leq o _ { j } [ k ] \leq O _ { j } [ k ]$ and $0 \leq O _ { j } [ k ] \leq O _ { i } ^ { m a x }$ . Not all tasks arriving at MEC are offloaded to the MEC for execution, depending on task offloading and service selection decisions. $r _ { j } ^ { m a x }$ denotes the maximum number of j-type tasks that can be offloaded to the MEC. The number of tasks of type j that can be offloaded at the MEC in time slot k as

$$
r _ { j } [ k ] = \sum _ { i = 1 } ^ { \mathcal { T } } x _ { i , j } [ k ] , \quad \forall k \in \mathcal { K } .\tag{7}
$$

The queue backlog of j-type task at time slot k is denoted by $Q _ { j } [ k ]$ . The queue backlog vector at k + 1 time slot obeys

$$
Q _ { j } [ k + 1 ] = m a x \{ Q _ { j } [ k ] - r _ { j } [ k ] , 0 \} + o _ { j } [ k ] ,\tag{8}
$$

where $Q _ { j } [ 0 ] \ = \ 0$ and $Q _ { j } [ k ] ~ \geq ~ 0$ Under the premise of ensuring the stability of the system, a maximum queue length constraint $Q _ { j } ^ { m a x }$ is set to prevent task queue congestion.

$$
\operatorname* { l i m } _ { K \to \infty } \frac { \operatorname { E } \left\{ Q _ { j } [ k ] \right\} } { K } \leq Q _ { j } ^ { \operatorname* { m a x } } .\tag{9}
$$

## D. UAV Communication Model

Since the length of each time slot is small enough, each UAV can be considered static within each time slot. The trajectory of the UAV is temporally discretized to convert it into time-series segments, consisting of the coordinates of the UAV at each moment in time. The UAV flies at a fixed altitude H, and the position of the UAV at each time slot can be represented as $c _ { t } [ k ] = ( x _ { t } ^ { u } [ k ] , y _ { t } ^ { u } [ k ] , H )$ . In the flight period, the UAV is assumed to fly from a predetermined initial position $c _ { u } ^ { I }$ to the endpoint $\overset { \cdot } { c } _ { u } ^ { F } .$ , and V max represents the UAV’s maximum speed. A multi-wing hovering UAV is utilized to support service migration. The UAV delivers services with a specific data size $L _ { q } ^ { m }$ to the q-th MEC server. The flight of the UAV in two neighboring time slot should satisfy $\| c _ { t } [ k + 1 ] - c _ { t } [ k ] \| \leq V ^ { \operatorname* { m a x } } \delta$ . The spatial distance from UAV to q-th MEC can be denoted by

$$
d _ { q } [ k ] = \sqrt { ( x _ { t } ^ { u } [ k ] - x _ { q } ^ { M E C } ) ^ { 2 } + ( y _ { t } ^ { u } [ k ] - y _ { q } ^ { M E C } ) ^ { 2 } + H ^ { 2 } } .\tag{10}
$$

It is assumed that the communication channel between the UAV and the ground MEC server follows a Line-of-Sight (LoS) model. When the UAV communicates with the MEC server in the q-th cluster, the corresponding path loss is expressed as

$$
P L ( d _ { q } [ k ] ) = ( \frac { 4 \pi f _ { c } d _ { q } [ k ] } { c } ) ^ { 2 } \eta L o s ,\tag{11}
$$

where $f _ { c }$ denotes the carrier frequency, c is the speed of light, and ηLos is the excessive path losses of the LoS $( \eta L o s > 1 )$ . The achievable data rate from UAV to MEC at time slot k can be expressed as

$$
W _ { q } ^ { U M } [ k ] = B _ { 2 } \log _ { 2 } ( 1 + \frac { p ^ { U M } \rho _ { 0 } } { P L ( d _ { q } [ k ] ) \xi ^ { 2 } } ) ,\tag{12}
$$

where $B _ { 2 }$ denotes the channel bandwidth of LoS. $P ^ { U M }$ denotes the transmit power of the UAV. The data size of the service to be migrated when the UAV hovers to the q-th cluster can be presented as

$$
L _ { q } ^ { m } = \sum _ { y = 1 } ^ { Y } \sum _ { j = 1 } ^ { J } ( y _ { j , q } \times L _ { y } ) , \quad \forall y \in \mathcal { Y } ,\tag{13}
$$

where $L _ { y }$ is the data size of the service $Y _ { j }$ . The transmission time incurred by the UAV when transmitting service program data in the q-th cluster can be denoted by

$$
T _ { q } ^ { U M } = \sum _ { k = 1 } ^ { \eta _ { q } } \frac { L _ { q } ^ { m } } { W _ { q } ^ { U M } [ k ] } .\tag{14}
$$

The state indicator variable of the UAV is

$$
\begin{array} { r } { w _ { k } = \left\{ \begin{array} { l l } { 1 , } & { c _ { t } [ k ] \neq c _ { t } [ k - 1 ] } \\ { 0 , } & { c _ { t } [ k ] = c _ { t } [ k - 1 ] . } \end{array} \right. } \end{array}\tag{15}
$$

The UAV transmission energy consumption is adopted as

$$
E _ { q } ^ { U M } [ k ] = ( \ 1 - w _ { k } ) \times \ P ^ { U M } \times \delta .\tag{16}
$$

The total energy consumption of the UAV during communication in the q-th cluster within time frame t is equal to the summation of the energy consumed in each time slot.

$$
E _ { q } ^ { U M } ( t ) = \sum _ { k = 1 } ^ { K } E _ { q } ^ { U M } [ k ] , \quad \forall t \in T .\tag{17}
$$

The energy model of multi-wing UAV proposed in [38] is employed. The flying power is formulated as

$$
\begin{array} { l } { \displaystyle { f ( v ) = P _ { 0 } \bigg ( 1 + \frac { 3 v ^ { 2 } } { v _ { t i p } ^ { 2 } } \bigg ) } } \\ { \displaystyle { + P _ { 1 } \bigg ( \sqrt { 1 + \frac { v ^ { 4 } } { 4 V ^ { 4 } i n d } } - \frac { v ^ { 2 } } { 2 V _ { i n d } ^ { 2 } } \bigg ) ^ { \frac { 1 } { 2 } } + \frac { 1 } { 2 } d _ { 0 } \rho s A _ { 0 } v ^ { 2 } } , }  \end{array}\tag{18}
$$

where $P _ { 0 }$ and $P _ { 1 }$ are two constants representing the blade profile power and induced power in hovering status, respectively. The flight energy consumption at time slot t generated by the UAV at speed v can be calculated as

$$
E _ { q } ^ { f } \left[ k \right] = w _ { k } \times \delta \times f ( v ) .\tag{19}
$$

The total energy consumption for UAV flight propulsion in the q-th cluster is

$$
E _ { q } ^ { F l y } ( t ) = \sum _ { k = 1 } ^ { K } E _ { q } ^ { f } [ k ] .\tag{20}
$$

The total hovering time in the q-th cluster can be expressed as

$$
\begin{array} { r } { T _ { q } ^ { H o v } ( t ) = \eta _ { q } \times \delta . } \end{array}\tag{21}
$$

Thus, the hovering energy consumption in the q-th cluster can be denoted by

$$
E _ { q } ^ { H o v } ( t ) = P ^ { H o v } \times T _ { q } ^ { H o v } ( t ) ,\tag{22}
$$

where $P ^ { H o v } = P _ { 0 } + P _ { 1 }$ in Equation (18).

The total energy consumption $E _ { q } ^ { U A V }$ of the UAV in the q-th cluster can be obtained, and it consists of three components.

$$
E _ { q } ^ { U A V } = E _ { q } ^ { F l y } ( t ) + E _ { q } ^ { H o v } ( t ) + E _ { q } ^ { U M } ( t ) .\tag{23}
$$

The energy efficiency of the UAV is evaluated by defining the ratio of the bits transmitted in the q-th cluster to the total energy consumption in that cluster.

$$
\overline { { E } } _ { q } = \frac { L _ { q } ^ { m } } { E _ { q } ^ { U A V } } .\tag{24}
$$

## E. Problem Formulation

To sum up, the problem of maximizing energy efficiency of the UAV in the network involves several components. It includes the task offloading of UTs, the service caching of MEC, and the trajectory of the UAV. This can be formulated as

$$
P : m a x \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \overline { { E } } _ { q }\tag{25a}
$$

$$
\operatorname* { l i m } _ { K  \infty } \frac { \operatorname { E } \{ Q _ { j } [ k ] \} } { K } < \varepsilon , \varepsilon \in \mathbb { R } ^ { + } , \forall j \in \mathcal { I } , \quad \forall k \in \mathcal { K } ,\tag{25b}
$$

$$
c _ { 1 } [ 1 ] = c _ { u } ^ { I } , c _ { T } [ K _ { t } ] = c _ { u } ^ { F } , \quad \forall t \in T ,\tag{25c}
$$

$$
\begin{array} { r } { \| c _ { t } [ k + 1 ] - c _ { t } [ k ] \| \le V ^ { \operatorname* { m a x } } \delta , \quad \forall t \in T , \forall k \in { \mathcal K } , } \end{array}\tag{25d}
$$

$$
T _ { q } ^ { U M } \leq T _ { q } ^ { H o v } , \quad \forall q \in \mathcal { Q } ,\tag{25e}
$$

$$
\sum _ { q \in \mathcal { Q } } E _ { q } ^ { U A V } \leq E ^ { T o t a l } ,\tag{25f}
$$

$$
\sum _ { j = 1 } ^ { J } y _ { j , q } \le M _ { q } , \quad \forall j \in \mathcal { I } , \forall q \in \mathcal { Q } ,\tag{25g}
$$

$$
x _ { i , j , q } \leq y _ { j , q } , \quad \forall i \in \mathcal { I } , \forall j \in \mathcal { I } , \forall q \in \mathcal { Q } ,\tag{25h}
$$

$$
x _ { i , j , q } \in \{ 0 , 1 \} , \quad \forall i \in \mathbb { Z } , ~ \forall j \in \mathcal { I } , ~ \forall q \in \mathcal { Q } ,\tag{25i}
$$

$$
y _ { j , q } \in \{ 0 , 1 \} , \quad \forall j \in \mathcal { I } , \forall q \in \mathcal { Q } ,
$$

$$
\rho _ { j , q } \in [ 0 , 1 ] , \quad \forall j \in \mathcal { I } , ~ \forall q \in \mathcal { Q } ,\tag{25j}
$$

(25k)

where (25a) is the objective function, and $\chi \quad =$ $\{ x _ { i , j , q } , y _ { i , j , q } , \rho _ { j , q } , c _ { t } [ k ] , \eta _ { q } \}$ is optimization variables set, including task offload decision, service selection decision, CPU frequency allocation, UAV trajectory, and UAV hovering time. Constraint (25b) means that the system’s queue stability condition, which is met if there exists a positive real number ε that satisfies the inequality. Constraint (25c) and (25d) specify the predefined start and end positions as well as the change in position of the UAV between any two time slots. Constraint (25e) indicates that task of j-type can be offloaded to the MEC for processing only if the associated service program is cached. Constraint (25f) represents the upper limit of energy consumption over the UAV’s operational period. Constraint $( 2 5 \mathrm { g } )$ indicates that due to the limitation of MEC storage, the number of services selected cannot exceed the maximum value that the MEC can host. Constraint (25h) indicates that task of j-type can be offloaded to the MEC for processing only if the associated service program is cached. Constraint (25i) and (25j) respectively indicate that the offloading decision and online service selection variables are binary. Constraint (25k) indicates that the variables assigned to the CPU computation resources on the MEC are continuous.

The joint optimization of online service selection, task offloading, CPU frequency allocation, UAV trajectory and hovering schedule in (25) is equivalent to the traveling salesman problem, which is NP-hard. Thus, problem P is NP-hard.

## IV. PROPOSED SOLUTION APPROACH

The original problem P is formulated as a multi-stage stochastic optimization problem, which is decomposed into two interrelated subproblems for tractability. The first subproblem, $P _ { 1 }$ , focuses on online service selection, task offloading, and CPU frequency allocation, guided by a defined offloading gain function. The second subproblem, $P _ { 2 }$ , involves the joint optimization of UAV trajectory planning and hovering schedule, conditioned on the decisions made in the service selection phase.

The function $G _ { i } ( \rho _ { j } )$ is defined to measure task offloading gain, as well as UT’s QoE. This function represents a weighted sum of the energy consumption and latency reduction when the task is offloaded to the MEC, denoted by

$$
G _ { i } ( \rho _ { j } ) = \alpha \times \frac { E _ { i , j } ^ { L o c a l } - E _ { i , j } ^ { M E C } } { E _ { i , j } ^ { L o c a l } } + ( 1 - \alpha )
$$

$$
\times \frac { T _ { i , j } ^ { L o c a l } - T _ { i , j } ^ { M E C } - T _ { i , j } ^ { T r } } { T _ { i , j } ^ { L o c a l } } ,\tag{26}
$$

where $G _ { i } ( \rho _ { j } )$ denotes the offloading gain of generating a task $A _ { i , j , q }$ after allocating CPU frequency $\rho _ { j }$ , and α is a constant denoting the relative weight between energy savings and latency reduction, where $\alpha \in [ 0 , 1 ]$

In a complex-variable network, randomly arriving tasks also need to be taken into account. Therefore, another gain function $U ( k )$ is defined to represent the network utility of a time slot k, where network utility means the value of gain in a cluster after considering task offloading. The value of the function $U ( k )$ is related to the number of randomly arriving tasks at the MEC and the task offloading decision, where $o _ { j }$ denotes the number of UTs which generate j-type task.

$$
U ( k ) = \sum _ { j = 1 } ^ { \mathcal { I } } \sum _ { i = 1 } ^ { o _ { j } } G _ { i } ( \rho _ { j } ) x _ { i , j } [ k ] .\tag{27}
$$

## A. Alternative Problem Formulation

Regarding the solution to the original objective of maximizing the energy efficiency of the UAV, it is related to the service selection decisions made by the ground MEC. Therefore, we need to determine what kind of service selection decision is optimal. The metric to measure this decision is network utility. Hence, the first sub-problem aims at this objective. Specifically, the corresponding optimization problem $P _ { 1 }$ is formulated as follows:

$$
P I : \underset { \{ x _ { i , j , y _ { i , j , \rho _ { j } } } \} } { m a x } \overline { { U _ { q } } } = \underset { K  + \infty } { \operatorname* { l i m } } \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathrm { E } \{ U ( k ) \}\tag{28a}
$$

$$
\operatorname* { l i m } _ { K  \infty } \frac { \operatorname { E } \{ Q _ { j } [ k ] \} } { K } < \varepsilon , \varepsilon \in \mathbb { R } ^ { + } , \quad \forall j \in \mathcal { I } , \forall k \in \mathcal { K } ,\tag{28b}
$$

$$
\sum _ { j = 1 } ^ { J } y _ { j } \leq M , \quad \forall j \in \mathcal { I } ,\tag{28c}
$$

$$
x _ { i , j } \leq y _ { j } , \quad \forall i \in \mathcal { I } , \forall j \in \mathcal { I } ,\tag{28d}
$$

$$
x _ { i , j } \in \{ 0 , 1 \} , \quad \forall i \in \mathcal { I } , \forall j \in \mathcal { I } ,\tag{28e}
$$

$$
y _ { j } \in \{ 0 , 1 \} , \quad \forall j \in \mathcal { T } ,\tag{28f}
$$

$$
\begin{array} { r } { \rho _ { j } \in [ 0 , 1 ] , \quad \forall j \in \mathcal { I } , } \end{array}\tag{28g}
$$

where P 1 is a joint optimization problem aimed at maximizing network utility in average time slot. It involves jointly optimizing service selection, task offloading, and CPU frequency allocation. Additionally, Lyapunov theory is used to analyze task queues.

## B. Lyapunov Optimization

The algorithm based on Lyapunov optimization theory is online and adaptive.

The Lyapunov optimization function is defined as in [39].

$$
L [ k ] = L ( Q [ k ] ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } Q _ { j } [ k ] ^ { 2 } .\tag{29}
$$

The Lyapunov drift function is denoted by

$$
\Delta ( Q [ k ] ) = \operatorname { E } \{ L [ k + 1 ] - L [ k ] | Q [ k ] \} .\tag{30}
$$

The utility function is added as a penalty function to the above drift function, yielding the drift-plus-penalty function.

$$
\Delta _ { V } L ( Q [ k ] ) { \stackrel { \Delta } { = } } \mathrm { E } \{ L [ k + 1 ] - L [ k ] \big | Q [ k ] \big \} - V \mathrm { E } \{ U [ k ] | ( Q [ k ] ) \} ,\tag{31}
$$

where V is a non-negative constant that controls the proportionality between queue stability and utility function values.

Lemma 1: In [39], any feasible solution to objective Eq. (28) satisfies the following inequality.

$$
\Delta _ { \nu } L ( Q [ k ] ) \leq B + \sum _ { j = 1 } ^ { \mathcal { I } } Q _ { j } [ k ] ( o _ { j } [ k ] - r _ { j } [ k ] ) - V U ( k ) .\tag{32}
$$

where $\begin{array} { r } { B = \frac { 1 } { 2 } \sum _ { i } ^ { \mathcal { I } } ( ( r _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } + ( O _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } ) , } \end{array}$

Proof: Due to $0 ^ { ^ { \circ } } \leq O _ { j } [ k ] \leq O _ { j } ^ { \operatorname* { m a x } } , \ 0 \leq o _ { j } [ k ] \leq O _ { j } [ k ] .$ Eq. (33) denotes by

$$
( \operatorname* { m a x } [ Q _ { j } [ k ] - r _ { j } [ k ] , 0 ] ) ^ { 2 } \leq ( Q _ { j } [ k ] - r _ { j } [ k ] ) ^ { 2 } .\tag{33}
$$

By squaring both sides of Eq. (8), the following inequality is obtained.

$$
\begin{array} { r l } & { Q _ { j } ^ { 2 } [ k + 1 ] } \\ & { \leq ( Q _ { j } [ k ] - r _ { j } [ k ] ) ^ { 2 } + 2 ( Q _ { j } [ k ] - r _ { j } [ k ] ) ) o _ { j } [ k ] + o _ { j } ^ { 2 } [ k ] } \\ & { \leq ( Q _ { j } [ k ] - r _ { j } [ k ] ) ^ { 2 } + 2 Q _ { j } [ k ] o _ { j } [ k ] + o _ { j } ^ { 2 } [ k ] . } \end{array}\tag{34}
$$

The inequality can be derived after performing the algebraic operations

$$
\begin{array} { r l } & { \frac { Q _ { j } ^ { 2 } [ k + 1 ] - Q _ { j } ^ { 2 } [ k ] } { 2 } } \\ & { \qquad \leq \frac { { \left( r _ { j } ^ { \operatorname* { m a x } } \right) } ^ { 2 } + { \left( Q _ { j } ^ { \operatorname* { m a x } } \right) } ^ { 2 } } { 2 } + Q _ { j } [ k ] { \left( o _ { j } [ k ] - r _ { j } [ k ] \right) } , } \end{array}\tag{35}
$$

where $\frac { ( r _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } + ( O _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } } { 2 }$ is a constant. The expectation of $\operatorname { E q }$ (33) is taken, summed over $j \in \mathcal { I } ,$ , and the right-hand side of Eq. (34) is added. The following is obtained.

$$
\Delta _ { \nu } L ( Q [ k ] ) \leq B + \sum _ { j = 1 } ^ { \mathcal { I } } Q _ { j } [ k ] ( o _ { j } [ k ] - r _ { j } [ k ] ) - V U ( k ) .\tag{36}
$$

Based on Lemma 1, an upper bound of the drift-plus-penalty function is derived. To jointly ensure queue stability and utility maximization, the proposed algorithm minimizes this upper bound at each time slot. The queue stability is then formally established through Lyapunov analysis.

Lemma 2: For a solution of Eq. (28), the queue Q[k] is stable.

Proof: In [39], if Eq. (28) has a solution, then for any constant $\zeta ~ > ~ 0 ,$ , a dynamic offloading strategy exists that satisfies the following conditions. For any time slot $k ,$ the following relationships hold.

$$
\mathrm { E } \left\{ U ( k ) \right\} \geq U ^ { o p t } - \zeta ,\tag{37}
$$

$$
\begin{array} { r } { \mathrm { E } \{ o _ { j } [ k ] \} \le \mathrm { E } \{ r _ { j } [ k ] \} + \zeta , } \end{array}\tag{38}
$$

where $U ^ { o p t }$ represents the upper bound on the network benefits across all feasible solutions, derived from the drift-pluspenalty function.

$$
\begin{array} { r l r } {  { \Delta _ { V } L ( Q [ k ] ) } } \\ & { = \operatorname { E } \bigl \{ L [ k + 1 ] - L [ k ] \big | Q [ k ] \bigr \} - V \operatorname { E } \{ U ( k ) | ( Q [ k ] ) \} } \\ & { \leq \frac { \big ( r _ { j } ^ { \operatorname* { m a x } } \big ) ^ { 2 } + ( O _ { j } ^ { \operatorname* { m a x } } \big ) ^ { 2 } } { 2 } + \displaystyle \sum _ { j = 1 } ^ { J } Q _ { j } [ k ] \operatorname { E } \big \{ o _ { j } [ k ] - r _ { j } [ k ] \big \} | Q [ k ] \big \} } \\ & { } & { \quad - V \operatorname { E } \operatorname { E } \{ I / ( k ) | ( O [ k ] ) \} } \end{array}\tag{39}
$$

When ζ = 0, Eq. (40) follows directly from Eqs. (37)-(39)

$$
\Delta _ { \nu } ( Q [ k ] ) \leq B - V U ^ { o p t } ,\tag{40}
$$

where B is a constant and $\begin{array} { r } { B \ge \frac { ( r _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } + ( Q _ { j } ^ { \operatorname* { m a x } } ) ^ { 2 } } { 2 } } \end{array}$

Thus, the proof of queue stability is established.

$$
\operatorname* { l i m } _ { K  + \infty } \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \operatorname { E } \{ U ( k ) \} \leq B - V U ^ { o p t } .\tag{41}
$$

The drift-plus-penalty framework ensures a balance between utility maximization and system stability. Specifically, by minimizing the upper bound of the Lyapunov drift and penalizing lower utility, the proposed method guarantees queue stability while dynamically adapting to stochastic task arrivals. The control parameter V tunes this tradeoff, with larger values prioritizing utility and smaller values enforcing tighter queue control.

## C. Joint Online Service Selection and Task Offloading Optimization

In this subsection, the goal is to maximize the network utility, which is a mixed integer nonlinear optimization problem. The DDPG algorithm, based on Lyapunov Optimization, is adopted to solve service selection, task offloading, and CPU frequency allocation. The DDPG algorithm is used, with state, action, and reward, defined as follows:

1) State: ${ \cal S } = \{ { \cal S } _ { R } , { \cal S } _ { N } , { \cal S } _ { M } \}$ , where $S _ { R }$ denotes number of CPU frequency remaining on the MEC. $S _ { N }$ denotes the number of different arrival task types. $S _ { M }$ denotes the number of services that can be hosted on the MEC.

2) Action: $A = \{ A _ { x } , A _ { y } , A _ { \rho } \}$ denotes the actions, where $A _ { x } \ = \ \{ A _ { x _ { 1 } } , A _ { x _ { 2 } } , \ldots , A _ { x _ { i } } , \ldots , A _ { x _ { I } } \}$ is the task offloading decision of i-th UT. $A _ { y } \ = \ \{ A _ { y _ { 1 } } , A _ { y _ { 2 } } , \dotsc , A _ { y _ { j } } , \dotsc , A _ { y _ { J } } \}$ denotes the service selection decision. $\begin{array} { r l } { A _ { \rho } } & { { } = } \end{array}$ $\{ A _ { \rho _ { 1 } } , A _ { \rho _ { 2 } } , \ldots , A _ { \rho _ { j } } , \ldots , A _ { \rho _ { J } } \}$ represents proportion of CPU frequency allocated to processing the service $Y _ { j }$

<!-- image-->  
Fig. 3. DDPG architecture for UAV-assisted heterogeneous edge networks.

3) Reward: when the action is selected, the reward received is $R \ = \ U ( k )$ , if the action executed does not satisfy the constraints, then $R = - P$ . Where P is a constant.

Fig. 3 shows the DDPG architecture of this paper. In the DDPG algorithm, the actor generates actions based on neural network computations, while the critic evaluates these actions by generating rewards and updating network parameters. The critic maintains a storage set, which stores sample tuples $( S _ { k } , A _ { k } , R _ { k } , S _ { k + 1 } ) \sim \mathcal { D }$ used for training and updating the parameters w. The loss function is defined as

$$
L ( w ) = \operatorname { E } [ R _ { k } + \gamma \operatorname* { m a x } Q _ { w } ( S _ { k + 1 } , A _ { k + 1 } ) - Q _ { w } ( S _ { k } , A _ { k } ) ] .\tag{42}
$$

The neural network is used to calculate the loss in this iteration. To reduce the loss, the parameter w is updated using the gradient descent method.

$$
\Delta w = a _ { c } [ Q _ { k } - Q _ { w } ( S _ { k } , A _ { k } ) \nabla _ { w } Q _ { w } ( S _ { k } , A _ { k } ) ] .\tag{43}
$$

In the actor, the gradient ascent method is used to optimize the policy update, which can be described in

$$
\Psi ( A \mid S ) = { \frac { e ^ { \Theta } \varphi ( S , A ) } { \sum _ { a \in A } e ^ { \Theta } \varphi ( S , A ) } } ,\tag{44}
$$

where Θ is the current network policy parameter, and $\varphi ( S , A )$ denotes the feature vector. The optimization objective function is specified as

$$
J ( \Theta ^ { \Psi } ) { = } \mathrm { E } { \{ } Q ^ { \Psi } ( { \cal S } , { \cal A } ) \} = \sum _ { s \in { \cal S } } d ( { \cal S } ) \sum _ { a \in { \cal A } } \Theta ^ { \Psi } ( { \cal A } \mid { \cal S } ) Q ^ { \Psi } ( { \cal S } , { \cal A } ) ,\tag{45}
$$

where $J ( \Theta ^ { \Psi } )$ denotes the output value of the actor network. The gradient of the strategy yields is

$$
\nabla _ { \Theta } J ( \Theta ^ { \Psi } )
$$

Algorithm 1 Solving the service selection, task offloading,   
and CPU frequency allocation problems in P 1.   
Input: Task arrival rate $\lambda _ { j } ,$ CPU frequency of the MEC fmax, MEC   
Maximum number of hosted services M, number of task types J;   
Output: Task offloading X ∗, service selection Y∗, CPU frequency allocation   
ρ∗, reward based on the offloading gain of UTs.   
1: Initialize the critic network $Q _ { w } ( S , A )$ , actor network $\Psi ( A \mid \mathrm { ~ \bf ~ \mathscr ~ { ~ S ~ } ~ } )$   
experience replay memory D, and initial queue state $Q _ { j } [ k ] ;$   
2: for episode=1,. . . , Episode do   
3: Select the initial state $S _ { k 1 } ;$   
4: for $k = 1 , \dots , K _ { t }$ do   
5: Choose an action by $\Psi ( A ( k ) \mid S ( k ) ) ;$   
6: Calculate the reward using ${ \dot { R _ { k } } } = { \dot { U ( k ) } } ;$   
7: Update the state $S _ { k + 1 } ;$   
8: for $j = 1 , \dots , J$ do   
9: Update queue state $Q _ { j } [ k + 1 ]$ by Eq. (8);   
10: end for   
11: Store the experience $( \mathrm { S } _ { k } , A _ { k } , R _ { k } , S _ { k + 1 } )$ in D;   
12: if The experience replay memory is filled then   
13: Select a Mini-batch of samples from D;   
14: Update the parameters of the critic network by Eq. (42);   
15: Update the parameters of the actor network by Eq. (46);   
16: Update parameters of the actor target network $\theta ^ { \Psi } \overset { \cdot } { \longleftarrow } \tau \overset { \cdot } { \theta ^ { \Psi } } +$   
$( 1 - \tau ) \theta ^ { \Psi ^ { \prime } }$ and the critic target network $\theta ^ { Q } \stackrel {  } {  } \tau \theta ^ { Q } + ( 1 - \tau ) \theta ^ { Q ^ { \prime } }$   
17: end if   
18: end for   
19: end for

$$
\begin{array} { r l } { \displaystyle } & { = \sum _ { s \in S } d ( S ) \sum _ { a \in A } \Theta ^ { \Psi } ( A | S ) \nabla _ { \Theta } \ln \Theta ^ { \Psi } ( A | S ) Q _ { w } ( S , A ) } \\ & { = \operatorname { E } \left\{ \nabla _ { \Theta } \ln \Theta ^ { \Psi } ( A | S ) \ Q _ { w } ( S , A ) \right\} . } \end{array}\tag{46}
$$

Problem P 1 is addressed by Algorithm 1, which utilizes a DDPG-based approach to service selection, optimize task offloading, and CPU frequency allocation.

## D. Joint UAV Trajectory and Hovering Schedule Optimization

For problem P 2, online service selection is obtained by solving problem P 1 using DRL approach based on Lyapunov optimization theory. Given online service selection decision, the energy efficiency of the UAV is maximized by jointly optimizing the UAV trajectory and hovering schedule.

$$
P 2 : \mathop { m a x } _ { \{ c _ { t } [ k ] , \eta _ { q } \} } \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \frac { \left( L _ { q } ^ { m } \right) ^ { * } } { E _ { q } ^ { U A V } }\tag{47a}
$$

$$
s . t . ( 2 5 c ) , ( 2 5 d ) , ( 2 5 e ) , ( 2 5 f ) .\tag{47b}
$$

where $\left( L _ { q } ^ { m } \right) ^ { * }$ represents the data size of the optimal service selected by solving P 1. In this section, the focus is on optimizing the UAV trajectory and hovering strategies. An improved DDPG algorithm based on energy efficiency of the UAV is proposed. The three key components of the DDPG algorithm are outlined as follows:

1) State: $s _ { k } ~ = ~ \{ c _ { q } ^ { M E C } , c _ { t } [ k ] , L _ { q } ^ { m } , E _ { k } \}$ , where $E _ { k }$ is the remaining energy of the UAV.

2) Action: $\begin{array} { r l r } { \alpha _ { k } } & { { } = } & { \left\{ A _ { \eta } , A _ { \theta } , A _ { v } \right\} } \end{array}$ denotes the actions in network, where $A _ { \eta } = \{ A _ { \eta _ { 1 } } , A _ { \eta _ { 2 } } , \dotsc , A _ { \eta _ { k } } , \dotsc , A _ { \eta _ { K } } \}$ denotes the number of hovering time slot. $\begin{array} { r l } { A _ { \theta } } & { { } = } \end{array}$ $\left\{ A _ { \theta _ { 1 } } , A _ { \theta _ { 2 } } , \ldots , A _ { \theta _ { k } } , \ldots , A _ { \theta _ { K } } \right\}$ denotes the heading angle of the UAV during flight, where $\theta ~ \in ~ [ - \pi , \pi ] . ~ A _ { v } ~ =$ $\left\{ A _ { v _ { 1 } } , A _ { v _ { 2 } } , \ldots , A _ { v _ { k } } , \ldots , A _ { v _ { K } } \right\}$ denotes the UAV flight speed.

Algorithm 2 Solving UAV Trajectory and Hovering Schedule   
Problems in P 2   
Input: Service selection ${ \mathcal { V } } ^ { * }$ , MEC coordinate $c _ { q } ^ { M E C }$ , UAV remaining power   
$E _ { k } \bar { ; }$   
Output: The trajectory coordinate of the UAV ct[k]∗, number of time slot   
for UAV hovering $\eta _ { q } ^ { * } .$ , reward based on the energy efficiency of the UAV;   
1: Initialize the actor network $\pi ( s ( k ) \mid \theta ^ { \pi } )$ and the critic network   
$Q ( s ( k ) , a ( k ) \mid \theta ^ { Q } )$ , the actor target network and the critic target network   
$\theta ^ { Q } \longrightarrow \theta ^ { Q ^ { ' } } , \theta ^ { \pi } \longrightarrow \theta ^ { \pi }$ , and experience replay memory ${ \mathcal { F } } ;$   
2: for episode=1,. . . , Episode do   
3: Set up the environment and select the initial state $s _ { k _ { 1 } }$   
4: for $k \bar { = } 1 , \dots , K _ { t }$ do   
5: Choose an action $\alpha _ { k } { = } \{ A _ { \eta } , A _ { \theta } , A _ { v } \}$ by Eq. (51);   
6: Calculate the reward $r ( k )$ using Eq. (50);   
7: Update the state $s _ { k + 1 }$ using Eq. (48)-(49);   
8: Store the experience $\lbrack s ( k ) , \overline { { \alpha } } ( \bar { k ) } , r ( k ) , s ( k + 1 ) ] ;$   
9: end for   
10: if The experience replay memory is filled then   
11: Randomly sample m transitions from experience relay memory   
(s(m), α(m), r(m), s(m + 1));   
12: Compute $y ( m ) = r ( \stackrel { \prime \prime } { m } ) + \gamma Q ^ { ' } ( s ( m + 1 ) , \pi ( s ( m + 1 ) ) \mid \pi ^ { \theta } ) \mid$   
${ \theta } ^ { { Q } ^ { ' } } )$   
13: Update the parameters of the critic network $Q ( s ( k ) , a ( k ) \mid \theta ^ { Q } )$   
by Eq. (53);   
14: Update the parameters of the actor network $\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } ^ { \pi } )$ by $\operatorname { E q . }$   
(46);   
15: Update parameters of the actor target network $\theta ^ { \pi ^ { ' } }$ and the critic   
target network ${ \theta } ^ { ' }$ by Eq. (54);   
16: end if   
17: end for

The UAV’s position update strategy at each time slot is expressed as

$$
\begin{array} { r l } & { x _ { t } ^ { u } [ k + 1 ] = x _ { t } ^ { u } [ k ] + \psi _ { u , k } A _ { \nu _ { k } } \cos ( A _ { \theta _ { k } } ) \delta , } \\ & { y _ { t } ^ { u } [ k + 1 ] = y _ { t } ^ { u } [ k ] + \psi _ { u , k } A _ { \nu _ { k } } \sin ( A _ { \theta _ { k } } ) \delta , } \end{array}\tag{48}
$$

where $\begin{array} { r } { \psi _ { u , k } = \frac { \ln ( 1 + E _ { k } ) } { \ln ( 1 + \varpi ) } } \end{array}$ represents the UAV’s speed attenuation factor at time slot k, and $\varpi$ denotes the UAV’s total battery energy. The UAV’s energy update equation is denoted by

$$
E _ { k + 1 } = E _ { k } - \frac { \delta \| \psi _ { u , k } A _ { \nu _ { k } } \| ^ { 2 } } { 2 } .\tag{49}
$$

3) Reward: Reward function $r ( k )$ regarding the transmitted data volume is defined to prevent the UAV from only flying without transmitting data.

$$
r ( k ) = \frac { \displaystyle \sum _ { \varsigma = 1 } ^ { k } W _ { q } ^ { U M } [ \varsigma ] \delta \left( E _ { q } ^ { U M } + E _ { q } ^ { H o v } \right) } { L _ { q } ^ { m } \times ( E _ { q } ^ { U A V } ) } .\tag{50}
$$

Algorithm 2 describes the joint optimization algorithm based on the DDPG. An experience replay mechanism is designed to maximize training efficiency as

$$
\alpha ( k ) = \pi ( s ( k ) \mid \theta ^ { \pi } ) ,\tag{51}
$$

where $\theta ^ { \pi }$ is the parameter of the actor network. The critic target network calculates the target Q-value as

$$
y ( k ) = r ( k ) + \gamma Q ^ { \prime } ( s ( k + 1 ) , \pi ( s ( k + 1 ) ) \mid \pi ^ { \theta } ) \mid \theta ^ { Q ^ { \prime } } ) ,\tag{52}
$$

where $\theta ^ { Q }$ represents the parameters of the critic network, $\theta ^ { Q ^ { \prime } }$ denotes the parameters of the critic target network, and $\gamma$ is a

Algorithm 3 Solving P Based on LyOSS-PG Algorithm   
Input: Task offloading, service selection, resource allocation decision, UAV   
trajectory, hover scheduling initial state set $X ^ { 0 } , Y ^ { 0 } , \rho ^ { 0 } , c ^ { 0 } , \eta ^ { 0 } ;$   
Output: The final optimized set corresponding to the input variables   
$X ^ { t } , \dot { Y ^ { t } } , \rho ^ { t } , c ^ { t } , \eta ^ { t } ;$   
1: Initialize the location coordinates of UT $c _ { i , j , q } ^ { U T } .$ , local computing frequency   
$f _ { i , j , q }$ and task attributes $\{ C _ { i , j , q } , L _ { i , j , q } \} ;$   
2: Divide UTs into $Q$ clusters by K-means algorithm and determine the   
location of the cluster center MEC ${ c } _ { { q } } ^ { { M } E C } ;$   
3: repeat   
4: for $t = 1 , \dots , T$ do   
5: Solve P 1 based on Algorithm 1 and update $Y ^ { t } , X ^ { t } , \rho ^ { t } ;$   
6: Solve P 2 for given $Y ^ { \tilde { t } }$ by using Algorithm 2 and update $c ^ { t } , \eta ^ { t } ;$   
7: Update queue statue $Q ( t ) \overline { { = \sum _ { i \in . \mathcal { T } } Q _ { j } [ K _ { t } ] } } ;$   
8: Update $t = t + 1 ;$   
9: end for   
10: until $i t e r = i t e r ^ { m a x }$

$$
L o s s ( \theta ^ { Q } ) = \frac { 1 } { m } \sum _ { 1 } ^ { m } ( y ( k ) - Q ( s ( k ) , a ( k ) \mid \theta ^ { Q } ) ) ^ { 2 } ,\tag{53}
$$

where m is the index of transitions in the mini-batch. The parameters of the actor target network and critic target network are updated using the soft update method from

$$
\begin{array} { r } { l c r { \theta ^ { Q } } = \omega \theta ^ { Q } + ( 1 - \omega ) \theta ^ { Q ^ { \prime } } } \\ { { \theta ^ { \pi } } = \omega \theta ^ { \pi } + ( 1 - \omega ) \theta ^ { \pi ^ { \prime } } , } \end{array}\tag{54}
$$

where $\omega$ is the soft update factor. After multiple training iterations, the process concludes when the $\mathrm { U A V } \mathbf { \dot { s } }$ position state, i.e., $\begin{array} { r c l } { c _ { t } [ k ] } & { = } & { c _ { u } ^ { F } } \end{array}$ , reaches the designated endpoint. The final optimized value is obtained through the trained neural network. As shown in Algorithm 3, by integrating task queue analysis with the problem-solving processes of both algorithms, the overall solution is represented by the LyOSS-PG algorithm. Line 1 represents the initialization of parameters, such as UT coordinates, local computational resources, and task attributes. Line 2 denotes clustering the UTs using the K-Means algorithm and determining the center of the cluster as the coordinates of the MEC. Lines 3-10 describe the iterative process. Line 5 involves solving P 1 using Algorithm 1 to obtain all task offloading, service selection, and CPU frequency allocation for the time frame t. Line 6 indicates that, given the solution $Y ^ { t }$ found in P 1, the UAV’s trajectory in the time frame t and hovering schedule is obtained through Algorithm 2. Line 7 updates the task queue for the time frame t. When the maximum number of iterations is reached, the outputs are task offloading, service selection, CPU frequency allocation, UAV trajectory, and hovering time slot. Simultaneously, as the fully connected layers are utilized in DDPG algorithms, the complexity of Algorithm 1 equals $\begin{array} { r } { O ( J \sum _ { l = 1 } ^ { L } k _ { l } \overline { { k _ { l - 1 } } } ) } \end{array}$ and Algorithm 2 equals $\begin{array} { r } { \mathbf { \bar { O } } ( \sum _ { l = 1 } ^ { L } k _ { l } k _ { l - 1 } ) } \end{array}$ ， where L corresponds the number of layers and $k _ { l }$ is the number of neurons in l-th layer. Thus, the complexity of proposed algorithm 3 is denoted by $\begin{array} { r } { O ( J + ( J + 1 ) \sum _ { l = 1 } ^ { L } k _ { l } k _ { l - 1 } ) } \end{array}$

## V. SIMULATION RESULTS

In this section, simulations are conducted to verify the effectiveness and performance of the proposed algorithms. The simulation considers a rectangular area of [1000,1000] square meters. In this space, there are five deployed MEC servers, with UTs randomly distributed and clusters divided using the K-means algorithm. The number of UTs generating j-type tasks $o _ { j }$ follows a Poisson distribution with parameter $\lambda _ { j }$ The UAV flies at a fixed height of 80 meters. The remaining parameters are detailed in Table II.

TABLE II  
SIMULATION PARAMETERS
<table><tr><td rowspan=1 colspan=1>Notation</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Notation</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1> $\overline { { T ^ { 0 } } }$ </td><td rowspan=1 colspan=1>500 s</td><td rowspan=1 colspan=1> $\underline { { L _ { i , j , q } } }$ </td><td rowspan=1 colspan=1>[20,40]Mbit</td></tr><tr><td rowspan=1 colspan=1> $\delta$ </td><td rowspan=1 colspan=1>1 s</td><td rowspan=1 colspan=1> $\frac { f _ { i , j , q } } { f ^ { m a x } }$ </td><td rowspan=1 colspan=1>[0.5,0.7] GHz</td></tr><tr><td rowspan=1 colspan=1>Vmax</td><td rowspan=1 colspan=1>40 m/s</td><td rowspan=1 colspan=1>fmax</td><td rowspan=1 colspan=1>10 GHz</td></tr><tr><td rowspan=1 colspan=1> $y _ { u } ^ { m a x }$ </td><td rowspan=1 colspan=1>1000m</td><td rowspan=1 colspan=1> $\overline { { p _ { i } ^ { t r } } }$ </td><td rowspan=1 colspan=1>[0.1,0.15]W</td></tr><tr><td rowspan=1 colspan=1> $\overline { { x _ { u } ^ { m a x } } }$ </td><td rowspan=1 colspan=1>1000m</td><td rowspan=1 colspan=1> $\overline { { C _ { i , j , q } } }$ </td><td rowspan=1 colspan=1>1000 cycles/bit</td></tr><tr><td rowspan=1 colspan=1> $\kappa$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 2 9 } } }$ </td><td rowspan=1 colspan=1> $L _ { y }$ </td><td rowspan=1 colspan=1>[100,300]Mbit</td></tr><tr><td rowspan=1 colspan=1> $\underline { { \rho _ { 0 } } }$ </td><td rowspan=1 colspan=1>-30 dB</td><td rowspan=1 colspan=1> $\overline { { M } }$ </td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1> $\overline { { B _ { 1 } } }$ </td><td rowspan=1 colspan=1>20 MHz</td><td rowspan=1 colspan=1> $\overrightarrow { P ^ { U M } }$ </td><td rowspan=1 colspan=1>10W</td></tr><tr><td rowspan=1 colspan=1> $\overline { { B _ { 2 } } }$ </td><td rowspan=1 colspan=1>40 MHz</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>250</td></tr><tr><td rowspan=1 colspan=1> $J$ </td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { 2 } } }$ </td><td rowspan=1 colspan=1> $\lambda _ { j }$ </td><td rowspan=1 colspan=1>5</td></tr></table>

TABLE III

DDPG PARAMETERS
<table><tr><td rowspan=1 colspan=1>Notation</td><td rowspan=1 colspan=1>Definition</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Episode</td><td rowspan=1 colspan=1>Numberof iterations</td><td rowspan=1 colspan=1>1000</td></tr><tr><td rowspan=1 colspan=1>memory</td><td rowspan=1 colspan=1>Size of replaymemory</td><td rowspan=1 colspan=1>3200</td></tr><tr><td rowspan=1 colspan=1>batchsize</td><td rowspan=1 colspan=1>Number of samples ina single training</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Numberof stepstoupdate TargetNetparameters</td><td rowspan=1 colspan=1>20</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Rewarddecayrate</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>Ya</td><td rowspan=1 colspan=1>Learning rate of actor network</td><td rowspan=1 colspan=1>0.007</td></tr><tr><td rowspan=1 colspan=1>2B</td><td rowspan=1 colspan=1>Learning rate of critic network</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Softupdate factor of target network</td><td rowspan=1 colspan=1>0.01</td></tr></table>

To compare the effects of different options, five benchmark scenarios are proposed for performance comparison as follows:

1) Offline-Average: The service selection process is conducted in an offline method, meaning that all services are pre-deployed on the MEC servers and cannot be updated. CPU frequency is allocated uniformly across all active services. In contrast, the task offloading decisions, UAV trajectory planning, and hovering time scheduling are jointly optimized using DDPG algorithm.

2) Offline-Gain Based: The same offline service selection strategy is adopted, while CPU frequency is allocated to maximize the offloading gain of UTs. Task offloading, UAV trajectory, and hovering scheduling are optimized using the same components as in the Offline-Average scheme.

3) Online-Average: An online service selection method is used, while the CPU frequency is evenly distributed, and the remaining variables are optimized using DDPG Algorithm.

4) LyOSS-DQN: The LyOSS algorithm is used to solve the first subproblem. The second subproblem, which optimizes the trajectory and hovering schedule of the UAV, is solved using the DQN algorithm.

5) LyOSS-PG-Fixed hovering time: The first subproblem is solved by LyOSS, while the second subproblem is solved using the DDPG algorithm with fixed number of hovering time slots.

The proposed network is trained using PyTorch in the Python 3.9 environment. A prioritized experience replay option is designed to improve training efficiency. The parameters related to network training are shown in Table III. The average network utility $( U _ { a v g } )$ is used as a measure for the P 1 subproblem, where $U _ { a v g }$ equals $\begin{array} { r } { \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \overline { { U } } _ { q } , } \end{array}$ representing the average offloading gain of UTs across the network. Considering the real-time performance of the network, the task arrival of UT is random, so considering the $U _ { a v g }$ including all UT clusters can better reflect the advantages of the proposed algorithm for solving P1.

Fig. 4 shows that all four approaches eventually converge after a certain number of iterations, indicating that each method is capable of making effective task offloading decisions, selecting appropriate services, and allocating CPU frequency for hosted services. Among them, the proposed LyOSS-PG algorithm achieves the best performance, which can be attributed to its dynamic online service selection mechanism driven by real-time task type arrivals, as well as its adaptive CPU frequency allocation strategy. In contrast, the offline service placement approach fails to account for time-varying task arrivals and relies instead on static average CPU frequency allocation. This limitation results in suboptimal energy efficiency performance over time. However, since offline service placement does not involve service selection, the dimensionality of the action space in the reinforcement learning process is reduced. This simplification contributes to a slightly faster convergence rate compared to the more flexible but complex LyOSS-PG algorithm.

Fig. 5 indicates the $U _ { a v g }$ across four scenarios with varying weight values α. It is a parameter in the UT offloading gain function, balancing energy reduction and delay reduction. Adjusting this value influences the reward function in the DRL network, thereby maximizing average network energy efficiency. The results indicate that the optimal weight value for all four schemes lies between 0.4 and 0.5. This value is adopted as the weight for the offloading gain function in subsequent experiments.

Fig. 6 analyzes the impact on $U _ { a v g }$ as the maximum number of services hosted on the MEC server M varies across four options. The results reveal that this parameter significantly influences $U _ { a v g } ,$ as it directly determines the number of services the MEC can host. Naturally, hosting more services allows for a greater variety of tasks to be offloaded, thereby improving network utility. Among the four schemes, the LyOSS-PG algorithm achieves the most substantial increase in $U _ { a v g } ,$ , primarily due to its online service selection approach, which becomes increasingly effective as the value of M rises, thereby maximizing network utility. Compared with offline service selection, LyOSS-PG algorithm is more sensitive to changes in the number of M, and the increase in the number of M means more service selection schemes. Online schemes have better performance in $U _ { a v g }$

Fig. 7 compares the variations in $U _ { \mathrm { a v g } }$ under four different schemes as the number of UTs increases. The number of UTs directly affects both the cluster partitioning process and the volume of computation tasks offloaded to the MEC server. As expected, all four approaches exhibit an upward trend in $U _ { \mathrm { a v g } }$ with the increasing number of UTs, due to the higher task density in the network. Notably, the proposed LyOSS-PG algorithm consistently outperforms the other three schemes. This performance gain can be attributed to two primary factors: the online service selection mechanism dynamically adjusts the service placement strategy in response to variations in task type distributions caused by the increasing UT population. The CPU frequency allocation strategy, guided by the offloading gain of each UT, optimizes resource utilization and enhances average network utility. It is also worth noting that the impact of increasing the number of UTs is relatively less pronounced than the effect of enlarging the MEC’s service capacity $M ,$ since not all additional tasks are offloaded to the MEC due to resource limitations and selective offloading policies.

<!-- image-->  
Fig. 4. The convergence of different algorithms.

<!-- image-->  
Fig. 5. Analysis of different options with different weight value α in $\operatorname { E q . }$ (26).

Fig. 8 compares the different options with respect to variations in task arrival rate λ. In the experiments, different types of task arrivals following a Poisson distribution were considered. The task arrival rate directly impacts the arrival of tasks in the dynamic network. The higher rate indicates an increase in the variety of tasks reaching the MEC servers. This requires the dynamic selection of necessary services based on the incoming task types. However, when the arrival rate is too high, $U _ { a v g }$ becomes constrained, as the MEC’s capacity to host services is limited and cannot accommodate all arriving task types. This limitation can be effectively mitigated by increasing the value of M .

<!-- image-->  
Fig. 6. Analysis of different options with different MEC server hosting services M .

<!-- image-->  
Fig. 7. Analysis of different options with different number of UTs.

Fig. 9 analyzes the evolution of $U _ { \mathrm { a v g } }$ over time for four different approaches. At the initial stage, the offline method exhibits a rapid increase in $U _ { \mathrm { a v g } } ,$ primarily due to the advantage of pre-deployed services on the MEC server that can immediately process a subset of incoming tasks. However, as task arrivals gradually stabilize, the performance of the offline approach reaches a plateau, with minor fluctuations reflecting the stochastic nature of task dynamics. In contrast, the average CPU frequency allocation approach shows minimal variation in $U _ { \mathrm { a v g } } ,$ as it lacks adaptive adjustment mechanisms. The online service selection approach, which requires the UAV to transmit service data to ground nodes, introduces occasional latency, particularly during the initial deployment phase. In our experimental setup, five ground clusters are initialized, corresponding to five distinct inflection points in the curve of the online algorithm. These inflection points occur as the UAV completes service transmission in each cluster sequentially. While the offline strategy achieves faster short-term gains, it fails to adapt to evolving service demand patterns. In the long term, dynamic service placement becomes critical for sustaining network utility. As a result, the online service selection approach ultimately achieves superior performance in terms of energy efficiency and responsiveness.

<!-- image-->  
Fig. 8. Analysis of different options with different arrival rate λ.

<!-- image-->  
Fig. 9. Analysis of different options with slot time.

Fig. 10 (a) and Fig. 10 (b) analyze the average queue length and $U _ { a v g }$ with different control factors V . In Lyapunov optimization, V is used to balance queue stability. In the service placement experiments, a larger number of UTs is considered to better approximate real-world conditions. The average queue length is defined as the number of the task of all types. Variations are compared under three different arrival rates. As the task arrival rate increases, the $U _ { a v g }$ improves, aligning with the previous conclusion in Fig. 8. However, excessively high task arrival rate may result in task queue backlogs. Additionally, the results show that as V increases, the average queue length gradually rises, leading to potential queue backlogs, with minimal changes observed when $V \leq$ 100. Similarly, $U _ { a v g }$ increases and stabilizes as V grows, with insignificant changes when $V ~ \geq ~ 1 0 0$ . Therefore, selecting extreme values of $V$ can be detrimental to the overall network.

<!-- image-->

(a) Average network utility  
<!-- image-->  
(b) Average queue length

Fig. 10. Analyzing the effect of the Lyapunov control parameter V with different options.  
<!-- image-->  
Fig. 11. The trajectory of the UAV in different algorithms.

Fig. 11 illustrates the UAV trajectory under various options. In these options, the LyOSS method is applied to solve the first subproblem, while different DRL algorithms are used for the second subproblem, resulting in distinct UAV trajectory outcomes. The figure displays UTs in different colors to differentiate the clusters they belong to after the clustering process, along with the 5 MECs. The UAV hovers in a specified area, defined as a circular region centered on the

<!-- image-->  
Fig. 12. Analysis of different options with different MEC server hosting services M for energy efficiency of the UAV with UTs = 250.

<!-- image-->  
Fig. 13. Analysis of different options with different number of UTs for energy efficiency of the UAV with M = 8.

MEC coordinates, with a radius of 3 meters at a height of H. It is evident that the LyOSS-PG trajectory outperforms those generated by LyOSS-Double DQN and LyOSS-DQN. This is because PG-based algorithms estimate action values through a critic network, which plays a crucial role in guiding strategy optimization. The critic network offers a more precise estimation of action values, improving the selection of optimal actions and resulting in a more favorable strategy. Additionally, the LyOSS-Double DQN scheme surpasses the remaining two due to the independent design of the target and evaluation networks in the Double DQN algorithm, which prevents overestimation of action values and enables the UAV to execute more advantageous actions. The trajectory of UAV will be affected by the UT mission that arrives in real time, because MEC will respond to the request of UAV service according to the mission request that arrives, so the service selection and task unloading decision under different schemes are one of the factors that cause the difference of UAV trajectory. Lastly, by fixing the UAV’s hovering time in the LyOSS-PG scheme, the trajectory becomes denser in the target area. This trajectory aligns with all of the experimental results mentioned above.

Fig. 12 illustrates the relationship between the number of services that the MEC server can host and the UAV’s energy efficiency under five different schemes. As the number of services M increases, all schemes demonstrate an overall improvement in energy efficiency, primarily due to the increased volume of data offloaded to the UAV. However, when the number of UTs is limited to 150, as compared to 250, the energy efficiency reaches a plateau beyond a certain service capacity (e.g., $M = 8 )$ , indicating that the service demands are sufficiently met even at lower service capacities. The proposed LyOSS-PG algorithm consistently outperforms the baseline methods, especially at higher M values, by jointly optimizing UAV trajectories and minimizing unnecessary energy expenditure. In contrast, schemes employing fixed UAV hovering times exhibit significantly lower energy efficiency due to excessive energy consumption during idle periods. Notably, for $M = 4$ and $M = 6 .$ , the highest energy efficiency is observed in the 150-UT scenario, which can be attributed to modified cluster divisions that result in more compact UAV flight paths.

Fig. 13 shows the change in the average energy efficiency of the UAV for the five schemes when the number of UTs is varied. As the number of UTs increases, the metrics for all five schemes improve, but similar to Fig. 13, the change in energy efficiency of the UAV due to a change in the value of M is smaller. Similarly, when M = 12 is fixed and the number of UTs equals 150 or 200, the average energy efficiency of the UAV under the LyOSS-PG algorithm remains the same. This is because the number of UTs is relatively small, and the types of tasks generated are limited, meaning the service demand is already satisfied when $M = 8$ . However, as the number of UTs increases, the metric value for $M \ = \ 1 2$ becomes greater, especially when there is a storage limitation in the MEC. This also demonstrates that increasing the number of services that can be hosted by the MEC is an effective strategy to accommodate changes in the number of UTs.

## VI. CONCLUSION

This paper investigates the problem of online service selection and task offloading in UAV-assisted heterogeneous edge networks, with the objective of enhancing the Quality of Experience for UTs. A joint optimization framework is formulated to maximize the energy efficiency of the UAV, and an online algorithm, termed LyOSS-PG, is proposed by integrating Lyapunov optimization with a deep reinforcement learning approach. The problem is decomposed into two subproblems. The first subproblem addresses online service selection, task offloading, and CPU frequency allocation, which are jointly solved using a DDPG algorithm guided by Lyapunov-based queue stability analysis. The second subproblem focuses on UAV trajectory planning and hovering schedule optimization, tackled by an enhanced DDPG method. Simulation results validate the effectiveness of LyOSS-PG, showing that it consistently outperforms baseline algorithms in terms of both UAV energy efficiency and UT QoE. Future work will extend the framework to multi-UAV collaborative online service computation, with complex communication scenarios under dynamic UAV altitude variations.

## REFERENCES

[1] L. Liu, X. Yuan, D. Chen, N. Zhang, H. Sun, and A. Taherkordi, “Multi-user dynamic computation offloading and resource allocation in 5G MEC heterogeneous networks with static and dynamic subchannels,” IEEE Trans. Veh. Technol., vol. 72, no. 11, pp. 14924–14938, Nov. 2023.

[2] H. Huang, Z.-Y. Chai, B.-S. Sun, H.-S. Kang, and Y.-J. Zhao, “Multiobjective deep reinforcement learning for computation offloading and trajectory control in UAV-base-station-assisted MEC,” IEEE Internet Things J., vol. 11, no. 19, pp. 31805–31821, Oct. 2024.

[3] H. Kurunathan, H. Huang, K. Li, W. Ni, and E. Hossain, “Machine learning-aided operations and communications of unmanned aerial vehicles: A contemporary survey,” IEEE Commun. Surveys Tuts., vol. 26, no. 1, pp. 496–533, 1st Quart., 2024.

[4] W. Shi et al., “A survey on intelligent solutions for increased video delivery quality in cloud–edge–end networks,” IEEE Commun. Surveys Tuts., vol. 27, no. 2, pp. 1363–1394, Apr. 2025.

[5] Y. He, K. Xiang, X. Cao, and M. Guizani, “Task scheduling and trajectory optimization based on fairness and communication security for multi-UAV-MEC system,” IEEE Internet Things J., vol. 11, no. 19, pp. 30510–30523, Oct. 2024.

[6] Y. Zuo, J. Guo, N. Gao, Y. Zhu, S. Jin, and X. Li, “A survey of blockchain and artificial intelligence for 6G wireless communications,” IEEE Commun. Surveys Tuts., vol. 25, no. 4, pp. 2494–2528, 4th Quart., 2023.

[7] W. Chu, P. Yu, Z. Yu, J. C. S. Lui, and Y. Lin, “Online optimal service selection, resource allocation and task offloading for multi-access edge computing: A utility-based approach,” IEEE Trans. Mobile Comput., vol. 22, no. 7, pp. 4150–4167, Jul. 2023.

[8] G. Zheng, C. Xu, M. Wen, and X. Zhao, “Service caching based aerial cooperative computing and resource allocation in multi-UAV enabled MEC systems,” IEEE Trans. Veh. Technol., vol. 71, no. 10, pp. 10934–10947, Oct. 2022.

[9] Q. Fan, W. Zhang, C. Ling, R. Yadav, D. Wang, and H. He, “Mobilityaware cooperative service caching for mobile augmented reality services in mobile edge computing,” IEEE Trans. Veh. Technol., vol. 73, no. 11, pp. 17543–17557, Nov. 2024.

[10] C. Dong, Y. Tian, Z. Zhou, W. Wen, and X. Chen, “Joint power allocation and task offloading for reliability-aware services in NOMA-enabled MEC,” IEEE Trans. Wireless Commun., vol. 23, no. 7, pp. 7537–7551, Jul. 2024.

[11] W. Chu, X. Jia, Z. Yu, J. C. S. Lui, and Y. Lin, “Joint service caching, resource allocation and task offloading for MEC-based networks: A multi-layer optimization approach,” IEEE Trans. Mobile Comput., vol. 23, no. 4, pp. 2958–2975, Apr. 2024.

[12] C. Xu, J. Guo, Y. Li, H. Zou, W. Jia, and T. Wang, “Dynamic parallel multi-server selection and allocation in collaborative edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 11, pp. 10523–10537, Nov. 2024.

[13] H. Guo, Y. Wang, J. Liu, and C. Liu, “Multi-UAV cooperative task offloading and resource allocation in 5G advanced and beyond,” IEEE Trans. Wireless Commun., vol. 23, no. 1, pp. 347–359, Jan. 2024.

[14] P. Cao et al., “Computational intelligence algorithms for UAV swarm networking and collaboration: A comprehensive survey and future directions,” IEEE Commun. Surveys Tuts., vol. 26, no. 4, pp. 2684–2728, 4th Quart., 2024.

[15] J. Li, W. Liang, W. Xu, Z. Xu, Y. Li, and X. Jia, “Service home identification of multiple-source IoT applications in edge computing,” IEEE Trans. Services Comput., vol. 16, no. 2, pp. 1417–1430, Mar. 2023.

[16] Y. Bai, D. Wang, G. Huang, and B. Song, “A deep-reinforcementlearning-based social-aware cooperative caching scheme in D2D communication networks,” IEEE Internet Things J., vol. 10, no. 11, pp. 9634–9645, Jun. 2023.

[17] Y. Zhang, X. Hou, H. Du, L. Zhang, J. Du, and W. Men, “Joint trajectory and resource optimization for UAV and D2D-enabled heterogeneous edge computing networks,” IEEE Trans. Veh. Technol., vol. 73, no. 9, pp. 13816–13827, Sep. 2024.

[18] Y. Zeng, S. Chen, J. Li, Y. Cui, and J. Du, “Online optimization in UAVenabled MEC system: Minimizing long-term energy consumption under adapting to heterogeneous demands,” IEEE Internet Things J., vol. 11, no. 19, pp. 32143–32159, Oct. 2024.

[19] B. Xu, Z. Kuang, J. Gao, L. Zhao, and C. Wu, “Joint offloading decision and trajectory design for UAV-enabled edge computing with task dependency,” IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5043–5055, Aug. 2023.

[20] Y. Gao, X. Yuan, D. Yang, Y. Hu, Y. Cao, and A. Schmeink, “UAVassisted MEC system with mobile ground terminals: DRL-based joint terminal scheduling and UAV 3D trajectory design,” IEEE Trans. Veh. Technol., vol. 73, no. 7, pp. 10164–10180, Jul. 2024.

[21] H. Cui, N. Zhang, and P. Liu, “Trajectory optimization for 6G-UAV based on deep reinforcement learning,” IEEE Trans. Veh. Technol., vol. 73, no. 11, pp. 17935–17939, Nov. 2024.

[22] Z. Kuang, H. Wang, J. Li, and F. Hou, “Utility-aware UAV deployment and task offloading in multi-UAV edge computing networks,” IEEE Internet Things J., vol. 11, no. 8, pp. 14755–14770, Apr. 2024.

[23] T. Du, X. Gui, X. Teng, K. Zhang, and D. Ren, “Dynamic trajectory design and bandwidth adjustment for energy-efficient UAVassisted relaying with deep reinforcement learning in MEC IoT system,” IEEE Internet Things J., vol. 11, no. 23, pp. 37463–37479, Dec. 2024.

[24] Z. Liu, J. Qi, Y. Shen, K. Ma, and X. Guan, “Maximizing energy efficiency in UAV-assisted NOMA–MEC networks,” IEEE Internet Things J., vol. 10, no. 24, pp. 22208–22222, Dec. 2023.

[25] P. Wang et al., “Decentralized navigation with heterogeneous federated reinforcement learning for UAV-enabled mobile edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 13621–13638, Dec. 2024.

[26] J. Li, C. Yi, J. Chen, K. Zhu, and J. Cai, “Joint trajectory planning, application placement, and energy renewal for UAV-assisted MEC: A triple-learner-based approach,” IEEE Internet Things J., vol. 10, no. 15, pp. 13622–13636, Aug. 2023.

[27] Z. Wei et al., “UAV-assisted data collection for Internet of Things: A survey,” IEEE Internet Things J., vol. 9, no. 17, pp. 15460–15483, Sep. 2022.

[28] B. Zeng, C. Zhan, C. Xu, and J. Liao, “Caching and 3D deployment strategy for scalable videos in cache-enabled multi-UAV networks,” IEEE Trans. Veh. Technol., vol. 72, no. 11, pp. 14875–14888, Nov. 2023.

[29] Y. Zhao, A. Xiao, S. Wu, C. Jiang, L. Kuang, and Y. Shi, “Adaptive partitioning and placement for two-layer collaborative caching in mobile edge computing networks,” IEEE Trans. Wireless Commun., vol. 23, no. 8, pp. 8215–8231, Aug. 2024.

[30] J. Chen, H. Xing, X. Lin, A. Nallanathan, and S. Bi, “Joint resource allocation and cache placement for location-aware multi-user mobile-edge computing,” IEEE Internet Things J., vol. 9, no. 24, pp. 25698–25714, Dec. 2022.

[31] Z. Xu et al., “Near-optimal and collaborative service caching in mobile edge clouds,” IEEE Trans. Mobile Comput., vol. 22, no. 7, pp. 4070–4085, Jul. 2023.

[32] D. Ren, X. Gui, and K. Zhang, “Adaptive request scheduling and service caching for MEC-assisted IoT networks: An online learning approach,” IEEE Internet Things J., vol. 9, no. 18, pp. 17372–17386, Sep. 2022.

[33] T. Liu, S. Ni, X. Li, Y. Zhu, L. Kong, and Y. Yang, “Deep reinforcement learning based approach for online service placement and computation resource allocation in edge computing,” IEEE Trans. Mobile Comput., vol. 22, no. 7, pp. 3870–3881, Jul. 2023.

[34] C. Feng, Q. Yang, T. Q. S. Quek, W. Wu, and K. Guo, “Spatiallytemporally collaborative service placement and task scheduling in MEC networks,” IEEE Trans. Veh. Technol., vol. 72, no. 12, pp. 16650–16666, Dec. 2023.

[35] X. Gao and L. Zhai, “Service experience oriented cooperative computing in cache-enabled UAVs assisted MEC networks,” IEEE Trans. Mobile Comput., vol. 23, no. 10, pp. 9721–9736, Oct. 2024.

[36] S. Bi, L. Huang, H. Wang, and Y.-J.-A. Zhang, “Lyapunov-guided deep reinforcement learning for stable online computation offloading in mobile-edge computing networks,” IEEE Trans. Wireless Commun., vol. 20, no. 11, pp. 7519–7537, Nov. 2021.

[37] Y. Yuan, L. Lei, T. X. Vu, S. Chatzinotas, S. Sun, and B. Ottersten, “Energy minimization in UAV-aided networks: Actor-critic learning for constrained scheduling optimization,” IEEE Trans. Veh. Technol., vol. 70, no. 5, pp. 5028–5042, May 2021.

[38] Y. Zeng, J. Xu, and R. Zhang, “Energy minimization for wireless communication with rotary-wing UAV,” IEEE Trans. Wireless Commun., vol. 18, no. 4, pp. 2329–2345, Apr. 2019.

[39] M. Neely, Stochastic Network Optimization With Application to Communication and Queueing Systems. San Rafael, CA, USA: Morgan & Claypool, 2010.

<!-- image-->  
Youwei Tao received the B.Eng. degree in software engineering from Chongqing Three Gorges University, Chongqing, China, in 2022. He is currently pursuing the M.Sc. degree in computer technology with the Central South University of Forestry and Technology. His current research interests include mobile edge computing, optimization algorithms, and its application.

<!-- image-->

Zhufang Kuang (Member, IEEE) received the M.Sc. degree in computer science from the National University of Defense Technology in 2006 and the Ph.D. degree in computer science from Central South University, Changsha, China, in 2012. He was a Post-Doctoral Researcher with the School of Software, Central South University. From 2015 to 2016, he was a Visiting Scholar/Professor with the University of Victoria, Victoria, BC, Canada. He is currently a Full Professor with the Department of Computer Science and Technology, Central South

University of Forestry and Technology. His current research interests include wireless communications and networking, the Internet of Things (IoT), mobile edge computing, and artificial intelligence. He is a Distinguished Member of CCF and a member of the CCF Internet of Things Council, the CCF Network and Data Communications Council, and ACM. He was the Chair of CCF YOCSEF CHANGSHA from 2022 to 2023.

<!-- image-->

Fen Hou (Member, IEEE) received the Ph.D. degree in electrical and computer engineering from the University of Waterloo, Waterloo, Canada, in 2008. She is currently an Associate Professor with the State Key Laboratory of IoT for Smart City, the Department of Electrical and Computer Engineering, and Guangdong–Hong Kong–Macau Joint Laboratory for Smart Cities, University of Macau. Her research interests include resource allocation intelligent computing networks, mechanism design, and optimal user behavior in crowd sensing networks.

She was a co-recipient of the IEEE Globecom Best Paper Award in 2010, the Distinguished Service Award in the IEEE MMTC in 2011, and the IEEE VTC-Fall Best Student Paper Award in 2021. She served as the TPC Chair and the Co-Chair for several IEEE conferences, such as ICCS 2014, INFOCOM 2014, ICCC 2015, ICC 2016, and ICCC 2021. She also serves as an Associate Editor for IEEE COMMUNICATIONS SURVEYS AND TUTORIALS.

<!-- image-->

Anfeng Liu received the M.Sc. and Ph.D. degrees in computer science from Central South University, China, in 2002 and 2005, respectively. He was a Post-Doctoral Researcher with the School of Electronic Science and Technology, National University of Defense Technology, Changsha, China. From 2009 to 2012. He was a Visiting Scholar/Professor with the Department of Electrical and Computer Engineering (ELCE), University of Waterloo, Canada. He is currently a Full Professor with the School of Electronic Information, Central South University, China. His current research interests include wireless sensor networks and mobile edge computing. He is also a Member (E200012141M) of China Computer Federation (CCF).