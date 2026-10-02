# Deep Reinforcement Learning-Based Online Resource Management for UAV-Assisted Edge Computing With Dual Connectivity

Linh T. Hoang, Graduate Student Member, IEEE, Chuyen T. Nguyen , and Anh T. Pham , Senior Member, IEEE

Abstract— Mobile Edge Computing (MEC) is a key technology towards delay-sensitive and computation-intensive applications in future cellular networks. In this paper, we consider a multi-user, multi-server system where the cellular base station is assisted by a UAV, both of which provide additional MEC services to the terrestrial users. Via dual connectivity (DC), each user can simultaneously offload tasks to the macro base station and the UAV-mounted MEC server for parallel computing, while also processing some tasks locally. We aim to propose an online resource management framework that minimizes the average power consumption of the whole system, considering long-term constraints on queue stability and computational delay of the queueing system. Due to the coexistence of two servers, the problem is highly complex and formulated as a multi-stage mixed integer non-linear programming (MINLP) problem. To solve the MINLP with reduced computational complexity, we first adopt Lyapunov optimization to transform the original multi-stage problem into deterministic problems that are manageable in each time slot. Afterward, the transformed problem is solved using an integrated learning-optimization approach, where model-free Deep Reinforcement Learning (DRL) is combined with modelbased optimization. Via extensive simulation and theoretical analyses, we show that the proposed framework is guaranteed to converge and can produce nearly the same performance as the optimal solution obtained via an exhaustive search.

Index Terms— Lyapunov optimization, mobile edge computing, deep reinforcement learning (DRL), queueing networks.

## I. INTRODUCTION

MOBILE Edge Computing (MEC) refers to an emerg-ing distributed computing paradigm that brings cloud (i.e., to the network’s edges) [1], [2], [3]. The technology has been widely recognized as a promising solution to solve the challenges aligned with the rapid growth of mobile applications and the Internet of Things (IoT). By offloading part of computational tasks to the MEC server, the endusers, especially ones that are battery-based and with limited hardware capabilities, can experience a better quality of service (QoS) in computation-intensive and latency-critical applications [4], [5], [6], [7].

In support of the dynamic and rapid deployment of MEC networks, mounting the MEC server on Unmanned Aerial Vehicles (UAVs) has recently attracted attention in both industry and academia [2], [3], [8], [9], [10]. In such an approach, the UAV can act as a flying base station that can effectively complement existing cellular networks by providing additional computational services to the ground user. Due to the inherent mobility and flexibility of UAVs, this approach is an on-demand solution well-suited in hotspot areas or rural areas where network capacity is insufficient [3], [8], [9], [11]. In addition, the Dual Connectivity (DC) technology [12] can also be integrated into the network to further enhance the computational efficiency of the edge device. Using DC, edge devices are enabled to communicate simultaneously with several eNodeBs, which might significantly improve the network’s throughput and mobility support [12], [13], [14], [15], [16]. Indeed, the concept of DC was introduced in the Third-Generation Partnership Project (3GPP) Release 12 and has recently been widely recognized as a promising approach for the deployment of ultra-dense 5G heterogeneous networks [12], [13], [14], [15], [16]. Associating users requesting computationally intensive services to a single MEC server might result in an overload and possible service denials of the server to other users. DC allows the users to offload tasks to two servers simultaneously for parallel computing, thus being a promising solution to balance the workload and avoid the service denial issue. Following these trends, MEC networks can be configured where the mBS and the UAV act as the Master eNodeB (MeNB) and the Second eNodeB (SeNB), respectively, both equipped with a MEC server. The edge user can then offload tasks to both the MEC servers simultaneously for parallel computing with abundant computational resources.

Besides mentioned advantages, integrating UAVs and DC into the MEC networks poses various challenges; one is the time complexity in resource management of the system. In general, the optimization in a multi-user, multi-server MEC network involves solving a mixed integer non-linear programming (MINLP) problem that jointly determines the channel assignment (i.e., the user association) for MEC servers and resource management for communication (e.g., bandwidth allocation) and computation (e.g., CPU frequency selection for local and remote computing) at the server and the user. Solving such a problem is computationally expensive, especially given the existence of a large number of users. Various solutions have been proposed to tackle the issue, such as metaheuristic methods [4], [17], [18], convex relaxation of binary variables [19], local search-based approaches [20], and decomposition-based methods [21]. Still, these conventional methods share a common of requiring a large number of iterations to bring out a good performance and are thus not suitable for real-time control of dynamic MEC systems.

To tackle the issue of time complexity, data-driven solutions such as deep reinforcement learning (DRL) are promising candidates that can perform very well while satisfying real-time control requirements [22], [23], [24], [25], [26], [27]. The DRL framework harnesses deep neural networks (DNNs) to learn the optimal policy that directly maps the system state (e.g., the channel condition and the number of backlog tasks) to a proper action (i.e., resource management decision) in each time slot. Training is performed via continuous interaction between the optimization solver and the environment (i.e., the MEC network) to maximize the reward following the decision in each step (e.g., the system’s energy efficiency and throughput). Indeed, using DNNs in optimization is a model-free approach in which the solver learns from experience (i.e., driven by the training data) to construct an optimal mapping policy, rather than relying on complex mathematical models that might not always be accurate and readily available. However, purely relying on the model-free solution has been reported to lead to unstable performance and suffer from slow convergence or even divergence [23], [25], [26]. A proper approach could be letting the DNN take part of the optimization (e.g., for optimizing binary variables) while still using conventional model-based methods for the rest. Indeed, the integration of data-driven and conventional model-based methods has improved the robustness and convergence of the DRL framework via online training [23], [24], [25], [26].

The other challenge in optimizing MEC networks is the preference for long-term key performance indicators (KPIs) of dynamic queueing systems, which refer to time-evolving queueing networks where decisions made in a time slot affect the optimization in subsequent slots [26], [28]. A typical example could be minimizing the long-term average power consumption, subject to queue stability constraints and given the randomness of the environment (e.g., channel gains and task arrival). Despite its importance, most existing DRL-based solutions [23], [24], [29], [30], [31] do not focus on the long-term performance when solving resource management problems in MEC networks. A well-known approach to cope with the long-term KPIs of a dynamic system is Lyapunov optimization [28]. The framework can be used to transform a multi-state problem into deterministic per-time slot subproblems while providing a theoretical guarantee to long-term system stability. The combination of the two robust tools, DRL and Lyapunov framework, thus is a promising approach to solving the MINLP problem of network resource management while monitoring the long-term KPIs, especially in large-scale multi-user, multi-server MEC systems [26].

In this paper, we consider a UAV-assisted MEC network with DC, where the UAV and the mBS act as the MeNB and the SeNB, respectively, to provide edge computing services to a set of ground users. Via DC, one mobile user can simultaneously obtain communication resources from both MEC servers in support of parallel computing. Under randomness of channel condition and task arrival, an MINLP is formulated to minimize the average power consumption of the whole system (including the user and the UAV, which are all battery-operated), given constraints on long-term queue stability and average task execution delay threshold. We aim to develop an online resource management algorithm with reduced computational complexity that produces system-wide energy efficiency while satisfying all QoS requirements for the user. We jointly optimize various system variables to achieve the goal, including channel assignment, local and remote computational resource scheduling and bandwidth allocation in each time slot. The Lyapunov framework is adopted to transform the original multi-stage problem into deterministic per-time slot problems. A hybrid scheme of combining the model-free DRL and model-based optimization is then proposed to solve resource management optimization in each time slot. To the authors’ best knowledge, this is the first work considering a dynamic multi-user, multi-server MEC system with assistance from UAVs via DC and developing a DRL framework for resource management in such a system. The main contributions can be summarized as follows:

1) Power Minimization for a multi-user, multi-server MEC System with dual connectivity: In support of parallel computing, we propose to utilize a UAV to assist edge computing in a cellular network via DC. The problem of resource management is formulated as a multi-stage MINLP to minimize the long-term average of the weighted-sum power consumption, constrained on long-term queue stability and task execution delay.

2) Lyapunov-guided DRL Approach: we develop a Lyapunov-guided DRL framework that can efficiently produce sub-optimal solutions. DRL is to cope with the complexity of the problem with the coexistence of two servers. Meanwhile, Lyapunov optimization is to deal with the long-term constraints on queue stability and average task execution delay of the queueing system.

3) Hybrid Approach for Actor-Critic Structure: The proposed framework integrates conventional model-based optimization and a data-driven approach via model-free DRL. The actor module utilizes a DNN and an efficient action quantizer to balance exploration and exploitation in producing channel assignment decisions. To accurately evaluate decisions made by the actor, the critic module utilizes model-based optimization rather than using another DNN conventionally.

4) We provide theoretical analyses and numerical results via extensive simulation to demonstrate the efficiency of the proposed method.

The remainder is organized as follows. Related works are provided in Section II. Sections III and IV detail the system model and problem formulation, respectively. Sections V and VI provide the description and theoretical analyses of the proposed framework. The numerical results are provided in Section VII. Finally, Section VIII concludes the paper.

TABLE I  
A COMPARISON OF OUR WORK WITH EXISTING DRL-BASED RESOURCE MANAGEMENT SCHEMES
<table><tr><td></td><td rowspan=1 colspan=2>Our work</td><td rowspan=1 colspan=1>[25]</td><td rowspan=1 colspan=1>[23]</td><td rowspan=1 colspan=1>[24]</td><td rowspan=1 colspan=1>[29]</td><td rowspan=1 colspan=1>[30]</td><td rowspan=1 colspan=1>[31]</td></tr><tr><td rowspan=1 colspan=1>Multipleusers</td><td rowspan=1 colspan=2>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>·</td><td rowspan=1 colspan=1>·</td><td rowspan=1 colspan=1>·</td></tr><tr><td rowspan=1 colspan=1>Multiple servers</td><td rowspan=1 colspan=2>√</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>：</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Optimize network resourcesallocation(channel bandwidth or transmission time)</td><td rowspan=1 colspan=2>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Optimize task offloading of the user</td><td rowspan=1 colspan=2>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Optimize computation on the server side</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Towards long-term KPIs of queueing systems</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>Dual connectivity and parallel computation</td><td rowspan=1 colspan=2>√</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr></table>

## II. RELATED WORKS

User Association. In recent years, several works have studied the user association problem (i.e., channel assignment or server selection) in resource management for multi-user, multiserver MEC networks [5], [20], [21], [30], [31]. Dai et al. [5] formulated a problem of joint computation offloading and user association to minimize overall power consumption of a MEC system where each user has multiple mutually dependent tasks. Tran and Pompili [20] studied the problem of joint sub-channel assignment and resource allocation of a multi-cell network to maximize a weighted sum of reduction in task completion time and energy consumption. Liu and Cao [30] considered the switching cost when a mobile user migrates its service from one server to another and modeled the problem of continuous server selection as a Markov Decision Process (MDP). Guo et al. [31] proposed an online learning-based MEC server selection mechanism under incomplete network information to minimize the time average task execution delay. Hu et al. [21] proposed a submodular optimization-based server selection method for mobile users to optimize the long-term energy-delay tradeoff. However, the works mentioned above have yet to investigate the user association problem in a parallel-computing scenario with dual connectivity. Instead, users have been grouped into separate clusters and allowed to connect to only one server at a time. More investigation is thus needed to fill the gap.

DRL for Resource Management. The utilization of DRL-based methods in optimization for MEC networks has recently attracted lots of attention from research community [23], [24], [25], [29], [30], [31]. Min et al. [29] proposed a reinforcement learning-based offloading scheme for IoT devices with energy harvesting (EH) to select the MEC server according to the current battery level and the predicted amount of the harvested energy. Huang et al. [23] proposed a DRL-based offloading framework that utilizes a deep neural network to produce potential binary offloading solutions, while a model-based optimization module is responsible for evaluating candidate decisions and labeling training data samples. Wu et al. [24] developed a hybrid framework that combines a deep Q network and convex optimization for determining offloading strategies at the user side and allocating resources at the computational access point. However, the works [23], [24], [29], [30], [31] only considered quasi-static scenarios and failed to adequately address long-term performance requirements (such as queue stability and average energy consumption) of a time-evolving queueing system. Following the direction toward long-term KPIs, Bi et al. [25] recently proposed a hybrid optimization-learning framework called LyDROO. The framework combines Lyapunov optimization and DRL to optimize task offloading, where the user either computes tasks locally or offloads the tasks (i.e., binary offloading).

In this paper, we also integrate Lyapunov optimization and DRL into a resource management framework to cope with long-term KPIs of a queueing network. Compared to [25], our innovations are three-fold. First, we propose a new actor module that utilizes a DNN to optimize parallel computation between users and servers. The method in [25] allows the user to either process local computation or offload tasks to an edge serve; thus it dose not apply to the parallel paradigm. Second, our proposed framework jointly optimizes not only the user side but also the server side. Since the problem involves many interacting network entities, we propose a new model-based critic module, which is entirely different from [25]. Third, [25] considered a single powerful MEC server with no limit on the number of users served at a time. Thus, their algorithms cannot be directly applied in our study, where we consider a multi-server system with dual connectivity and specify constraints on the server’s serving capability. A comprehensive comparison between our proposed scheme and other existing DRL-based methods is summarized in Table I. It is noteworthy that the critic module’s algorithm in this paper is adopted in part from our previous work [32] to optimize local computation on the user side.

## III. SYSTEM MODEL

As illustrated in Fig. 1, we consider a multi-server MEC system with Dual Connectivity (DC). There are two MEC servers, one located at a macro base station (i.e., the Master eNB, MeNB) and the other mounted on a UAV (i.e., the Secondary eNB, SeNB), provides additional edge computing services to a set of ground users. Each mobile user can simultaneously connect to the two MEC servers simultaneously. From now on, the UAV-mounted MEC sever and the SeNB, similarly to the macro base station and the MeNB, will be used interchangeably.

For convenience, we denote the index sets of the mobile devices, the MEC servers, and the time slots respectively as ${ \mathcal { N } } \triangleq \{ 1 , 2 , \ldots N \} , S \triangleq \{ { \mathrm { U A V } } , { \mathrm { m B S } } \}$ , and ${ \mathcal { T } } \triangleq \{ 1 , 2 , \ldots \}$ It is noted that in the following, we index each user by the letter i and the MEC server by letter j, i.e., $i \in \mathcal { N } , j \in \mathcal { S }$

The modeling of task computation, queueing system, task offloading, and power consumption are presented below. For ease of reference, key notations used in the article are summarized in Table II.

## A. Task Queuing Model

We assume that the mobile devices are processing independent and fine-grained tasks [1]. Each task is represented by a volume of bits, which can be decomposed into several packets transmitted to nearby MEC servers and processed in parallel. At the beginning of each slot, a volume of $A _ { i } ^ { t }$ bits arrives at the user i and can be processed starting from the next slot. Without loss of generality, we assume that $A _ { i } ^ { t }$ is independent and identically distributed (i.i.d) over time slots with Poisson distribution and an average rate $\mathbb { E } [ A _ { i } ^ { t } ] = \lambda _ { i }$ (bits), $i \in \mathcal N .$

<!-- image-->  
Fig. 1. Dual-Connectivity-supported UAV-assisted MEC system model.

TABLE II  
SUMMARY OF KEY NOTATIONS
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td>(t）</td><td>Local queue length of the user i at the beginning of time slot t</td></tr><tr><td>()</td><td>Queue length of the UAV dedicated for the useri at time slot t</td></tr><tr><td>A</td><td>Taskarrival ofuseriintimeslott</td></tr><tr><td> $f _ { l , i } ^ { t }$ </td><td>Local computation frequency of useri in time slot t</td></tr><tr><td> $f _ { c , i } ^ { \prime }$ </td><td>The UAV&#x27;s CPU frequency dedicated for processing user i&#x27;s tasks in time slot t</td></tr><tr><td> $\overline { { p _ { l , i } ^ { t } } }$ </td><td>Power consumption for local task execution of user i in time slot t</td></tr><tr><td> $\overline { { p _ { \mathrm { T x } , i } ( t ) } }$ </td><td>Transmit power for task offloading of useriin time slot t</td></tr><tr><td> $\underline { { p _ { c } ( t ) } }$ </td><td>Power consumption for task execution of the UAV in time slot t</td></tr><tr><td> $\alpha _ { i , j } ^ { t }$ </td><td>Theratio of bandwidth allocated to useri on the server j&#x27;schannel in time slot t</td></tr><tr><td> $\overline { { r _ { i , j } ^ { t } } }$ </td><td>Offloading volume of useri on server j&#x27;s link in time slot t</td></tr><tr><td> $\overline { { l _ { i } ^ { t } } }$ </td><td>The amount of tasks processed locally by user iin time slot t</td></tr><tr><td>G</td><td>The amount of useri&#x27;s tasks processed by the UAV in time slot t</td></tr><tr><td> $\overline { { W _ { j } } }$ </td><td>The total bandwidth of server j</td></tr><tr><td> $\overline { { h _ { i , j } ^ { t } } }$ </td><td>Channel gain of useri for the link to server j in time slot t</td></tr><tr><td> $\overline { { x _ { i , j } ^ { t } } }$ </td><td>Link association for useri on the server j’s channel in time slot t</td></tr></table>

In the tth time slot, the user processes $l _ { i } ^ { t }$ bits locally and on opportunity can upload $r _ { i , \mathrm { U A V } } ^ { t }$ and $r _ { i , \mathrm { m B S } } ^ { t }$ bits to the UAV-mounted MEC and the macro base station, respectively. The newly arrived task volumes at the beginning of one slot will be buffered in the user queue before they can be processed in subsequent slots. Let $Q _ { i } ^ { l } ( t )$ denote the local queue length of user i at the beginning of time slot t; the queue update process can be expressed as

$$
Q _ { i } ^ { l } ( t + 1 ) = \operatorname* { m a x } \left\{ Q _ { i } ^ { l } ( t ) - D _ { i } ^ { t } , 0 \right\} + A _ { i } ^ { t } , t \in \mathcal { T } ,\tag{1}
$$

where $D _ { i } ^ { t } \triangleq l _ { i } ^ { t } + { r } _ { i , \mathrm { U A V } } ^ { t } + { r } _ { i , \mathrm { m B S } } ^ { t }$ denotes the amount of tasks departing from the user i’s local queue in time slot t.

At the UAV side (i.e., the SeNB), we assume that the UAV maintains N dedicated queues, one for each user, to buffer tasks offloaded by users. Let $c _ { i } ^ { t }$ denote the amount of tasks from user i executed by the UAV in time slot $t ;$ the UAV’s task queue dedicated for user $i ,$ denoted by $Q _ { i } ^ { s } ( t )$ , can be derived similarly as

$$
Q _ { i } ^ { s } ( t + 1 ) = \operatorname* { m a x } \left\{ Q _ { i } ^ { s } ( t ) - c _ { i } ^ { t } , 0 \right\} + r _ { i , \mathrm { U A V } } ^ { t } , t \in \mathcal { T } .\tag{2}
$$

In this paper, all user and UAV task queues are assumed with sufficiently large capacity. In addition, without loss of generalization, all task queues are empty initially, i.e., $Q _ { i } ^ { l } ( 0 ) =$ $Q _ { i } ^ { s } ( 0 ) = 0 , i \in \mathcal { N }$

Regarding the macro BS (i.e., the MeNB), we assume that the server has redundant computational resources and is powered by an electrical grid; therefore, we do not consider the macro BS’s queues and power consumption in optimization. In other words, tasks offloaded to the macro BS will not be buffered in queues but executed promptly, and the energy consumed is less important than other network entities.

According to Little’s Law [33], the average delay experienced by one user is proportional to the long-term average number of tasks awaiting in the system. Thus, we exploit the average queue length at the user and the UAV, denoted by $\overline { { Q } } _ { i } ^ { l }$ and $\breve { Q } _ { i } ^ { s }$ , as a measure of the task completion delay for local and remote task processing. Furthermore, the two thresholds $Q _ { l , i } ^ { \mathrm { t h } }$ and $Q _ { s , i } ^ { \mathrm { t h } }$ are defined as a Quality of Service (QoS) constraint for the ith user as

$$
\overline { { Q } } _ { i } ^ { l } = \operatorname* { l i m } _ { T  \infty } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ Q _ { i } ^ { l } ( t ) ] \leq Q _ { l , i } ^ { \mathrm { t h } }\tag{3}
$$

$$
\overline { { Q } } _ { i } ^ { s } = \operatorname* { l i m } _ { T  \infty } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ Q _ { i } ^ { s } ( t ) ] \leq Q _ { s , i } ^ { \mathrm { t h } }\tag{4}
$$

where the expected values of the queue length (i.e., E -Qli(t) and E $[ Q _ { i } ^ { s } ( t ) { \bar { ] } } )$ are taken over the randomness of the channel gain and task arrival in a time slot. It is worth noting that from (1) we have $Q _ { i } ^ { l } ( t + 1 ) \geq A _ { i } ^ { t }$ , thus $\overline { { Q } } _ { i } ^ { l } \geq \mathbb { E } [ A _ { i } ^ { t } ]$ for all users. Therefore, $Q _ { l , i } ^ { \mathrm { t h } }$ should be selected such that $Q _ { l , i } ^ { \mathrm { t h } } \geq \lambda _ { i } , i \in \mathcal { N }$

## B. Task Execution Model

To process tasks locally, the mobile user needs to assign a specific number of CPU frequencies for each task. Let $f _ { l , i } ^ { t }$ denote the local CPU frequency of user i in time slot $t ;$ the amount of locally-computed tasks in time slot t can then be expressed as

$$
l _ { i } ^ { t } = \tau f _ { l , i } ^ { t } / L _ { i } .\tag{5}
$$

Here, τ denotes the time slot length and $L _ { i }$ denotes the processing density, defined as the number of CPU cycles required for user i to process one bit. According to circuit theory, the power consumption for local execution at the ith user is given by [34] and [35]

$$
p _ { l , i } ( t ) = \kappa _ { i } \left( f _ { l , i } ^ { t } \right) ^ { 3 } ,\tag{6}
$$

where the parameter $\kappa _ { i }$ is the effective switched capacitance of the CPU at the ith device, and dependent on the hardware architecture.

Similarly, at the UAV, the amount of tasks computed by the MEC server for the ith user in time slot t can be expressed by

$$
c _ { i } ^ { t } = \tau f _ { c , i } ^ { t } / L _ { s } ,\tag{7}
$$

where $f _ { c , i } ^ { t }$ denotes the CPU frequency resources that the UAV allocates for computing the ith user’s tasks; $L _ { s }$ denotes the processing density for the UAV’s CPU to process one bit. The power consumption of the UAV for computation can be defined similarly as

$$
p _ { c } ( t ) = \sum _ { i \in \mathcal { N } } \kappa _ { s } \left( f _ { c , i } ^ { t } \right) ^ { 3 } ,\tag{8}
$$

where $\kappa _ { s }$ denotes effective switched capacitance of the UAV’s CPU. We assume that the computational capability of the UAV is stronger than that of the mobile user, but limited by the maximum CPU frequency, i.e., $f _ { c , i } ^ { t } \le f _ { c } ^ { \operatorname* { m a x } } , i \in \mathcal { N }$

## C. Task Offloading Model

Channel power gain from user i to the MEC server $j \colon$

$$
h _ { i , j } ^ { t } = \frac { \tilde { h } _ { i , j } ^ { t } g _ { j } } { ( d _ { i , j } ) ^ { \gamma _ { j } } } ,\tag{9}
$$

where $d _ { i , j }$ denotes the distance from user i to the MEC server $j , \gamma _ { j }$ denotes the path loss exponent $( \gamma _ { j } \ge 2 ) , g _ { j }$ denotes the reference channel gain, and $\tilde { h } _ { i , j } ^ { t }$ denotes the small-scale fading channel power gain, which is assumed to have a finite mean value, E $\prime [ \tilde { h } _ { i , j } ^ { t } ] < \infty \ [ 3 4 ]$ , for the server $j ^ { \flat } \boldsymbol { \mathrm { s } }$ link.

To allocate radio resources to the mobile device, each MEC server will first select a subset of users (assuming that one server cannot serve all the users at the same time), then allocate each user in that subset an appropriate bandwidth for communication offloading. Let $\boldsymbol { x } _ { i , j } ^ { t }$ denotes the link association for the user i on the server $\bar { j } ^ { \bullet }$ communication channel in time slot t: $x _ { i , j } ^ { t } = 1$ indicates that the user i could utilize bandwidth allocated on the server $j ^ { \circ } \mathbf { s }$ channel; otherwise, no bandwidth is allocated and offloading is prohibited. $\mathcal { N } _ { j } ^ { t } \triangleq \big \{ i \in \mathcal { N } \big | x _ { i , j } ^ { t } = 1 \big \} \subset \mathcal { N }$ then can be defined as the set of mobile devices associated with the MEC server $j$ in time slot t. Similarly, the set of the MEC servers that associate with the ith user in time slot t can be defined as $\begin{array} { r } { S _ { i } ^ { t } \triangleq \left\{ j \in \mathcal { S } \left| x _ { i . j } ^ { t } = 1 \right. \right\} \subset \mathcal { S } } \end{array}$

Regarding the communication energy, according to the Shannon-Hartley formula, the transmit power for user i to offload $r _ { i } ^ { t }$ bits can be obtained as [36]

$$
p _ { \mathrm { T x } , i } ( t ) = \sum _ { j \in S _ { i } ^ { t } } \left( 2 ^ { \frac { r _ { i , j } ^ { t } } { W _ { j } \alpha _ { i , j } ^ { t } \tau } } - 1 \right) \frac { N _ { 0 } W _ { j } } { h _ { i , j } ^ { t } } ,\tag{10}
$$

where $W _ { j }$ denotes the total bandwidth of the server j, $\alpha _ { i , j } ^ { t }$ denotes the bandwidth ratio allocated to user i on the server $j ^ { \prime } { \bf s }$ channel, and $N _ { 0 }$ denotes the background noise density.

In (10), $r _ { i , j } ^ { t }$ denotes the offloading volume of the user i on the server $\mathbf { \bar { \rho } } _ { j ^ { \prime } \mathbf { s } }$ communication channel in time slot t, thus $\begin{array} { r } { r _ { i } ^ { t } = \sum _ { j \in S _ { i } ^ { t } } r _ { i , j } ^ { t } . } \end{array}$

It is worth noting that since the mobile user is supported by dual connectivity, one user can connect to both the two MEC servers at the same time, i.e., $x _ { i , \mathrm { U A V } } ^ { t }$ and $x _ { i , \mathrm { m B S } } ^ { t }$ can be both equal to one in a time slot, thus

$$
\left| S _ { i } ^ { t } \right| = \sum _ { j \in \cal S } x _ { i , j } ^ { t } \le 2 , \quad i \in \mathcal { N } , \ t \in \mathcal { T } ,\tag{11}
$$

where $| { \cal A } |$ denotes the number of elements in set A. Due to signaling overhead for resource management, we assume that server j is able to serve at most $\chi _ { j } ^ { \operatorname* { m a x } }$ users in a time slot, i.e.,

$$
\left| \mathcal { N } _ { j } ^ { t } \right| = \sum _ { i \in \mathcal { N } } x _ { i , j } ^ { t } \leq \chi _ { j } ^ { \operatorname* { m a x } } , \quad j \in \mathcal { S } , \ t \in \mathcal { T } .\tag{12}
$$

## IV. THE MULTI-STAGE MINLP PROBLEM OF POWERCONSUMPTION MINIMIZATION

## A. Problem Formulation

We focus on the weighted-sum system power consumption, which consists of power consumed for task execution at the user device and the UAV-mounted MEC server, as well as the user’s transmit power for task offloading. Energy consumed for other purposes, such as for maintaining the basic operations of the MEC system and for propulsion of the UAV, are omitted for simplicity. Accordingly, the system’s power consumption at time slot t, denoted by $P _ { \mathrm { s y s } } ( t )$ , can be calculated as a weighted sum as

$$
P _ { \mathrm { s y s } } ( t ) = \psi _ { c } p _ { c } ( t ) + \sum _ { i \in \mathcal { N } } \psi _ { i } \left( p _ { l , i } ( t ) + p _ { \mathrm { T x } , i } ( t ) \right) ,\tag{13}
$$

where $\psi _ { i }$ and $\psi _ { c }$ are positive numbers denoting the weight factors for the power consumption of user i and the UAV, respectively. $\psi _ { i }$ and $\psi _ { c }$ can be adjusted to reflect the system’s preference in optimizing the power consumption of different nodes, as well as to balance the impact of the UAV’s and the mobile device’s energy [34].

The ultimate goal of the optimization is to minimize the long-term average of the system power consumption, given constraints on the stability of task queues and the limit on the radio and computational resources. The optimization variables include the user’s local computation, the volume of offloaded tasks, the UAV’s remote processing scheduling, and the radio resource allocation.

Let $\begin{array} { r c l } { \mathbf { X } } & { = } & { \{ \mathbf { X } ^ { t } \} _ { t \in \mathcal { T } } } \end{array}$ denote the combination of all optimization variables over time. Additionally, let $\begin{array} { r l } { \mathbf { X } ^ { t } } & { { } = } \end{array}$ $\left\{ \mathbf { \bar { x } } _ { j } ^ { t } , \alpha _ { j } ^ { t } , \mathbf { r } _ { j } ^ { t } , \mathbf { f } _ { l } ^ { t } , \mathbf { f } _ { c } ^ { t } \right\} _ { j \in \mathcal { S } }$ denote the combined vector of optimization variables at time slot t for all server $j$ in ${ \mathcal { S } } \colon { \bf x } _ { j } ^ { t } =$ $\{ \boldsymbol { x } _ { i , j } ^ { t } \} _ { i \in \mathcal { N } } , \boldsymbol { \alpha } _ { j } ^ { t } = \{ \boldsymbol { \alpha } _ { i , j } ^ { t } \} _ { i \in \mathcal { N } } , \mathbf { r } _ { j } ^ { t } = \left\{ \boldsymbol { r } _ { i , j } ^ { t } \right\} _ { i \in \mathcal { N } } , \mathbf { f } _ { l } ^ { t } = \{ f _ { l , i } ^ { t } \} _ { i \in \mathcal { N } } ,$ $\mathbf { f } _ { c } ^ { t } = \{ f _ { c , i } ^ { t } \} _ { i \in \mathcal { N } }$ . Then, the problem can be formulated as a multi-stage MINLP problem as

$$
\begin{array} { r l r } { \mathbf { P 1 } : } & { \displaystyle \operatorname* { m i n } _ { \mathbf { X } } \displaystyle \operatorname* { l i m } _ { t  \infty } \frac { 1 } { T } \sum _ { t = 0 } \mathbb { E } [ P _ { \mathrm { s y s } } ( t ) ] } & { \mathrm { ( 1 4 a ) } } \\ & { \mathrm { s . t . } } & { x _ { i , j } ^ { t } \in \{ 0 , 1 \} , \displaystyle | \mathcal { N } _ { j } ^ { t } | \leq \chi _ { j } ^ { \operatorname* { m a x } } , \quad i \in \mathcal { N } , \ j \in \mathcal { S } , \in \mathcal { T } , } \end{array}\tag{14b}
$$

$$
0 \leq \alpha _ { i , j } ( t ) , \sum _ { i \in \mathcal { N } _ { j } } \alpha _ { i , j } ^ { t } \leq 1 , \quad i \in \mathcal { N } , \ j \in \mathcal { S } , \ t \in \mathcal { T } ,\tag{14c}
$$

$$
0 \leq f _ { l , i } ^ { t } \leq f _ { i } ^ { \operatorname* { m a x } } , \quad i \in \mathcal { N } , \ t \in \mathcal { T } ,\tag{14d}
$$

$$
0 \leq f _ { c , i } ^ { t } \leq f _ { c } ^ { \operatorname* { m a x } } , \quad i \in \mathcal { N } , \ t \in \mathcal { T } ,\tag{14e}
$$

$$
0 \leq p _ { \mathrm { T x } , i } ( t ) \leq p _ { \mathrm { T x } , i } ^ { \operatorname* { m a x } } , \quad i \in \mathcal { N } , \ t \in \mathcal { T } ,\tag{14f}
$$

$$
l _ { i } ^ { t } + r _ { i , \mathrm { U A V } } ^ { t } + r _ { i , \mathrm { m B S } } ^ { t } \leq Q _ { i } ^ { l } ( t ) , \quad i \in \mathcal { N } , t \in \mathcal { T } ,\tag{14g}
$$

$$
c _ { i } ^ { t } \leq Q _ { i } ^ { s } ( t ) , \quad i \in \mathcal { N } , \ t \in \mathcal { T }\tag{14h}
$$

$$
\operatorname* { l i m } _ { t \to \infty } \frac { \mathbb { E } \left[ Q _ { i } ^ { l } ( t ) \right] } { t } = 0 , \quad i \in \mathcal { N }\tag{14i}
$$

$$
\operatorname* { l i m } _ { t  \infty } \frac { \mathbb { E } [ Q _ { i } ^ { s } ( t ) ] } { t } = 0 , \quad i \in \mathcal { N }\tag{14j}
$$

In P1, (14b) denotes that the server $j$ can server at most $\chi _ { j } ^ { \mathrm { m a x } }$ users at a time. (14c) ensures that the total bandwidth used for the server $j ^ { \prime } \mathbf { s } ^ { \prime }$ uplink communication is bounded by Wj. (14d) and (14e) indicate the maximum CPU frequency of the user and the UAV, denoted by $f _ { i } ^ { \mathrm { m a x } }$ and $f _ { c } ^ { \mathrm { m a x } }$ , respectively. (14f) denotes the maximum transmit power of the mobile device on each communication link. (14g) and (14h) guarantee that the amount of tasks processed in a time slot (i.e., tasks offloaded and computed by the user and tasks computed by the UAV) does not surpass the backlog of task queues in each time slot. Finally, (14i) and (14j) indicate the mean rate stability [28] for the local and remote queues. Note that (14i) and (14j) do not provide any guarantee of the time-average expected backlogs in queues (and thus the average computation delay). The QoS constraints (3) and (4), which is a stronger form of stability [28], are thus useful.

We observe that P1 is a stochastic optimization problem of a time-evolving system. Indeed, radio and computation resource management decisions need to be made in each time slot under the randomness of the arrival task and the fading channel. Furthermore, optimal decisions are temporally correlated and should be adaptive to the time-varying system states, such as the current queue size at the mobile device and the UAV. Solving P1 is challenging also because the optimization evolves a large number of interdependent optimization variables. Specifically, the radio resource management variables among different end users $( \mathrm { i } . \mathrm { e } . , x _ { i , j } ^ { t }$ and $\alpha _ { i , j } ^ { t } , i \in \mathcal { N } )$ are coupled and interdependent with the computational resource scheduling at both sides $( \mathrm { i . e . , ~ } f _ { l , i } ^ { t }$ and $f _ { c , i } ^ { t } , i \in \mathcal { N } )$ , which suggests that a joint optimization approach is indeed needed. Later in the numerical results, we show that the over-aggressive approach (e.g., a greedy policy based on channel gain or queue length) could not solve the formulated problem effectively.

In the following, instead of solving P1 directly, we consider its modified version, denoted by P2, to obtain an efficient asymptotically optimal online solution as follows. First, we can rewrite (14c) as $\begin{array} { r l r l } { \pmb { \alpha } _ { j } ^ { t } } & { { } \in } & { { \mathcal { A } } } & { { } \triangleq } \end{array}$ $\begin{array} { r } { \left\{ \alpha _ { j } ^ { t } \in \mathbb { R } _ { N } ^ { + } \left| \sum _ { i \in \mathcal { N } _ { j } } \alpha _ { i , j } ^ { t } \le 1 \right. \right\} , j \in \mathcal { S } , t \in \mathcal { T } } \end{array}$ . To obtain P2, we replace (14c) by the following constraint,

$$
\alpha _ { j } ^ { t } \in \tilde { \mathcal { A } } \triangleq \left\{ \alpha _ { j } ^ { t } \in \mathbb { R } _ { + } ^ { N } \bigg | \sum _ { i \in \mathcal { N } _ { j } } { \alpha _ { i , j } ^ { t } } \leq 1 \mathrm { ~ a n d ~ } \alpha _ { i , j } ^ { t } \geq \epsilon _ { A } \mathrm { , ~ } i \in \mathcal { N } _ { j } ^ { t } \right\}\tag{14k}
$$

where $\epsilon _ { A } \in ( 0 , 1 / N )$ is a constant. Constraint (14k) causes the transmit power function in (10) continuous and differentiable with respect to $\alpha _ { i , j } ^ { t } , j \in S _ { i } ^ { t }$ . It is worth noting that although the optimal solution to P2 is only a approximation of the optimal solution to P1, we can make them arbitrarily close by setting ϵA to be sufficiently small.

## B. Lyapunov-Guided Problem Transformation

We adopt Lyapunov optimization framework [28] to decouple the multi-stage problem P2 into deterministic problems that can be solved in each time slot.

First, to cope with the QoS constraints (3) and (4) on the long-term average of the queue length, we introduce two virtual queues for each mobile,

$$
Z _ { i } ^ { l } ( t + 1 ) = \operatorname* { m a x } \left. Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { l } ( t + 1 ) - Q _ { l , i } ^ { \mathrm { t h } } , 0 \right. ,\tag{15}
$$

$$
Z _ { i } ^ { s } ( t + 1 ) = \operatorname* { m a x } \left. \vphantom { Z _ { i } ^ { s } } Z _ { i } ^ { s } ( t ) + Q _ { i } ^ { s } ( t + 1 ) - Q _ { s , i } ^ { \mathrm { t h } } , 0 \right. ,\tag{16}
$$

for $i \in \mathcal { N } , t \in \mathcal { T }$ , where $Z _ { i } ^ { l } ( 0 ) ~ = ~ Z _ { i } ^ { s } ( 0 ) ~ = ~ 0 .$ . By the definition of the two virtual queues, it is proved in [28] that constraints (3) and (4) are satisfied if the two virtual queues are mean-rate stable, i.e., lim $_ { T  \infty } \mathbb { E } [ Z _ { i } ^ { l } ( t ) ] / T = \bar { \mathrm { ~ 0 ~ } }$ and limT→∞ E $\left[ Z _ { i } ^ { s } ( t ) \right] / T = 0$

In support of the problem transformation, we define the system state at time slot t as $\Theta ( t ) \triangleq \{ Q _ { i } ^ { l } ( t ) , Q _ { i } ^ { s } ( t )$ $Z _ { i } ^ { l } ( t ) , Z _ { i } ^ { s } ( t ) \bigr \} _ { i \in \mathcal { N } } .$ . The Lyapunov function is then defined as a measure of the total queue backlog at time slot t as

$$
\mathcal { L } \left( \boldsymbol { \Theta } ( t ) \right) \triangleq \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } \left( Q _ { i } ^ { l } ( t ) ^ { 2 } + Q _ { i } ^ { s } ( t ) ^ { 2 } + Z _ { i } ^ { l } ( t ) ^ { 2 } + Z _ { i } ^ { s } ( t ) ^ { 2 } \right) ,\tag{17}
$$

To keep all the queues stable, the Lyapunov drift function is introduced as

$$
\Delta \left( \Theta ( t ) \right) \triangleq \mathbb { E } \left[ \mathcal { L } \left( \Theta ( t + 1 ) \right) - \mathcal { L } \left( \Theta ( t ) \right) \vert \Theta ( t ) \right]\tag{18}
$$

To minimize the long-term average power consumption while ensuring the queue stability constraint, we define the Lyapunov-drift-plus-penalty as

$$
\Delta _ { V } \left( \Theta ( t ) \right) \triangleq \Delta \left( \Theta ( t ) \right) + V \mathbb { E } \left[ P _ { \mathrm { s y s } } ( t ) \vert \Theta ( t ) \right] ,\tag{19}
$$

where V is a positive number denoting a control parameter for the trade-off between the system’s power consumption and the average queueing delay. The following theorem provides an upper bound of $\Delta _ { V } \left( \Theta ( t ) \right)$ ), which is crucial to the transformation of the multi-stage problem P2 into per-time slot deterministic problems.

Theorem 1: The drift-plus-penalty $\Delta _ { V } \left( \Theta ( t ) \right)$ is bounded as

$$
\Delta _ { V } \left( \Theta ( t ) \right) \leq \hat { B } - \sum _ { i \in \mathcal { N } } \mathbb { E } \left[ \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) \right) \left( D _ { i } ^ { t } - A _ { i } ^ { t } \right) \right|\tag{Θ ( t ) }
$$

$$
- \sum _ { i \in \mathcal { N } } \mathbb { E } \left[ \left( Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) \right) \left( c _ { i } ^ { t } - r _ { i , U A V } ^ { t } \right) \right.\tag{Θ ( t ) }
$$

$$
+ \ V \mathbb { E } \left[ P _ { s y s } ( t ) | \Theta ( t ) \right]\tag{20}
$$

where $\begin{array} { r } { D _ { i } ^ { t } = l _ { i } ^ { t } + r _ { i , U A V } ^ { t } + r _ { i , m B S } ^ { t } ; ~ \hat { B } } \end{array}$ consists of constant terms from the observation at the beginning of time slot t, thus can be put aside from the optimization of the target variables Xt. Proof: Please refer to Appendix A. □

<!-- image-->  
Fig. 2. Schematic of the proposed DRLRM framework.

With support from Theorem 1, we can transform the multi-stage problem P2 into a deterministic problem that can be solved in each time slot using the opportunistic expectation minimization technique [28]. Specifically, the long-term goal of P2 can be archived by minimizing the upper bound of $\Delta _ { V } ( \Theta ( t ) )$ in each time slot, taking the system state $\Theta ( t )$ observed at the beginning of a time slot as the input. By removing constant terms in (20), the deterministic per-time slot problem is formulated as

$$
\begin{array} { r l } & { \mathbf { P 3 } : \displaystyle \operatorname* { m i n } _ { \mathbf { X } ^ { t } } G \left( \mathbf { X } ^ { t } \right) } \\ & { \qquad \mathrm { s . t . } ( 1 4 \mathbf { b } ) , ( 1 4 \mathbf { k } ) , ( 1 4 \mathbf { d } ) - ( 1 4 \mathbf { h } ) } \end{array}\tag{21a}
$$

with the objective function as follows

$$
\begin{array} { r l } { G ( \mathbf { X } ^ { t } ) = } & { - \displaystyle \sum _ { i \in \mathcal { N } } \left[ \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) \right) \left( l _ { i } ^ { t } + r _ { i , \mathrm { U A V } } ^ { t } + r _ { i , \mathrm { m B S } } ^ { t } \right) \right] } \\ & { - \displaystyle \sum _ { i \in \mathcal { N } } \left[ \left( Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) \right) \left( c _ { i } ^ { t } - r _ { i , j } ^ { t } \right) \right] } \\ & { + \ V \left( \psi _ { c } p _ { c } ( t ) + \displaystyle \sum _ { i \in \mathcal { N } } \psi _ { i } \left( p _ { l , i } ( t ) + p _ { \mathrm { T x } , i } ( t ) \right) \right) } \end{array}\tag{22}
$$

It is worth mentioning that solving P3 does not require any future information about incoming tasks and wireless channel state other than the current state of the task queues, making the approach an online optimization design. In the next section, we will introduce a DRL-based online optimization algorithm to solve P3 efficiently.

## V. DEEP REINFORCEMENT LEARNING-BASED ONLINE RESOURCE MANAGEMENT

We propose a DRL-based resource management (DRLRM) scheme for solving the per-time slot problem P3. The proposed framework, as depicted in Fig.2, consists of three main modules, which are (1) the actor module, (2) the critic module, and (3) the policy update module. The actor module obtains necessary information for optimization from the system observer and adopts a DNN to output several potential decisions for channel assignment, $\tilde { \mathbf { x } } ^ { t } = \mathbf { \bar { \Psi } } \{ x _ { i , j } ^ { t } \} _ { i \in \mathcal { N } , j \in \mathcal { S } } .$ The critic module evaluates each decision made by the actor module by solving remaining variables $\mathbf { y } ^ { t } = \left\{ \alpha _ { i , j } ^ { t } , r _ { i , j } ^ { t } , f _ { l , i } ^ { t } , f _ { c , i } ^ { t } \right\} _ { i \in \mathcal { N } , j \in \mathcal { S } }$ using model-based optimization. The policy update module logs a history of the system state-optimal decision pairs on the fly and re-trains the actor module’s DNN in a periodic manner so that the mapping policy will be updated and adaptive to the time-varying channel condition. In the following, we describe in detail the learning-based framework depicted in Fig. 2, followed by the model-based optimization of the critic module.

For ease of convenience, the input and output of the actor module are denoted as follows. The input is denoted by $\Xi ^ { t } = \mathbf { \Lambda }$ $\left\{ h _ { i , j } ^ { t } , Q _ { i } ^ { l } ( t ) , Q _ { i } ^ { s } ( t ) , Z _ { i } ^ { l } ( t ) , Z _ { i } ^ { s } ( t ) \right\} _ { i \in \mathcal { N } , j \in \mathcal { S } } ,$ which includes the channel gain and the queue length of all physical and virtual queues for each user. The module’s output is presented by $\{ \tilde { \mathbf { x } } _ { m } ^ { t } \} _ { m = 1 } ^ { k }$ , where k denotes the number of potential channel assignment decisions made.

Remark 1: The above approach is backed by the Tammer decomposition technique [37], where combinatorial variables in $\tilde { \mathbf { x } } ^ { t }$ are decoupled from continuous variables in $\mathbf { y } ^ { t }$ . Indeed, by temporarily fixing $\tilde { \mathbf { x } } ^ { t }$ , we can further decompose P3 into several sub-problems with separate objectives and constraints.

Remark 2: To obtain the optimal decision for $\tilde { \mathbf { x } } ^ { t } ,$ an exhaustive search requires evaluating

$$
k _ { \mathrm { m a x } } = \binom { N } { \chi _ { U A V } ^ { \mathrm { m a x } } } \binom { N } { \chi _ { m B } ^ { \mathrm { m a x } } }\tag{23}
$$

possible channel assignment decisions, where $\begin{array} { r } { \binom { n } { k } = \frac { n ! } { k ! ( n - k ) ! } } \end{array}$ indicates the binomial coefficient of selecting an (unordered) subset of k elements from a fixed set of n elements. Since $\binom { n } { k }$ is in $O ( n ^ { k } )$ , an exhaustive search over all possible channel assignments is with complexity $\mathcal { O } \left( N ^ { \chi _ { U A V } ^ { \operatorname* { m a x } } + \chi _ { m B S } ^ { \operatorname* { m a x } } } \right)$ In the following sub-section, we propose to use a DNN to find $( \tilde { \mathbf { x } } ^ { t } ) ^ { \star }$ with much less computational complexity. The DNN will be trained periodically to dynamically adapt to the time-varying channel condition and approximate an optimal policy $\Pi _ { t } ^ { \star }$ that maps the current system state to an optimal channel assignment decision in time slot t, $\Pi _ { t } ^ { \star } : \Xi ^ { t } \mapsto ( \tilde { \mathbf { x } } ^ { t } ) ^ { \star }$

## A. Outline of the Proposed DRLRM Framework

1) Model-Free Actor Module: The actor module consists of a DNN and an action quantizer. The DNN obtains $\Xi ^ { t }$ and processes the forward propagation to output a relaxed channel assignment $\hat { \mathbf { x } } ^ { t } = \left\{ \bar { { x } } _ { i , j } ^ { t } \in \lbrack 0 , 1 ] \right\} _ { i \in \mathcal { N } , j \in \mathcal { S } } ,$ which will be later quantized into a number of potential channel assignment decisions. We adopt a Deep Neural Network (DNN) to represent the channel assignment policy (an approximation to the optimal policy $\Pi _ { t } ^ { \star } )$ as $\Pi _ { \Phi _ { t } } : \Xi ^ { t } \mapsto { \hat { \mathbf { x } } } ^ { t }$ , where $\Phi _ { t }$ denotes the DNN’s parameters at time slot t. To ensure that $\hat { x } _ { i , j } ^ { t } \in [ 0 ; 1 ]$ for all $i \in \mathcal { N } , j \in \mathcal { S }$ , we use the sigmoid activation function at the output layer of the DNN.

The action quantizer of the actor module will then obtain the relaxed channel assignment $\hat { \mathbf { x } } ^ { t }$ to generate a batch of k potential decisions. Let Γ denote the quantizer policy, we have $\bar { \Gamma } : \hat { \mathbf { x } } ^ { t } \mapsto \tilde { \mathbf { x } } ^ { t }$ , where $\tilde { \mathbf { x } } ^ { t } = \{ \tilde { x } _ { i , j } ^ { t } \in \{ 0 , \overset { \cdot } { 1 } \} \} _ { i \in \mathcal { N } , j \in \mathcal { S } }$ denotes the output channel assignment decision. Let $\varsigma _ { j } ^ { t }$ denote the $\chi _ { j } ^ { \mathrm { { m a x } . } }$ th largest element of $\hat { \mathbf { x } } _ { j } ^ { t } \triangleq \left\{ \hat { x } _ { i , j } ^ { t } \in \{ 0 , \overset { \cdot } { 1 } \} \right\} _ { i \in \mathcal { N } }$ (the set of relaxed channel assignments corresponding to the server $j )$ Then, for each $\hat { x } _ { i , j } ^ { t }$ in $\hat { \mathbf { x } } ^ { t } .$ , the quantization policy Γ generates a corresponding value of $\tilde { x } _ { i , j } ^ { t }$ in $\tilde { \mathbf { x } } ^ { t }$ as

$$
\tilde { x } _ { i , j } ^ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } \hat { x } _ { i , j } ^ { t } \geq \varsigma _ { j } ^ { t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{24}
$$

To generate the first decision $( \tilde { \mathbf { x } } _ { 1 } ^ { t } )$ , we apply the policy Γ directly to the output of the DNN, i.e., $\hat { \mathbf { x } } ^ { t }$ . The remaining $( k - 1 )$ decisions are generated by applying the policy Γ to noise-added versions of $\hat { \mathbf { x } } ^ { t }$ , denoted as Sigmoid $( \hat { \mathbf { x } } ^ { t } + \mathbf { n } )$ where Sigmoid(·) denotes the element-wise sigmoid function to ensure each element of the output vector falls within the range (0, 1). Here, $\mathbf { n } ~ \sim ~ \mathcal { N } ( \mathbf { 0 } , \sigma _ { n } ^ { 2 } \mathbf { I } )$ is a 2N-dimensional zero-mean random vector following the normal (Gaussian) distribution with a diagonal covariance matrix $\sigma _ { n } ^ { 2 } \mathbf { I } ;$ I denotes the identity matrix. The variable $\sigma _ { n }$ is a hyper-parameter responsible for the balance between exploration and exploitation of the action quantizer. Too relaxed values of $\sigma _ { n }$ might not take advantage of the DNN output to predict the optimal channel assignment. In contrast, too strict values of $\sigma _ { n }$ might hamper the critic module from extracting good approximations of the per-time slot problem’s global minimum in each time slot. In the long term, this phenomenon causes the DNN to experience difficulties in learning the optimal channel assignment policy due to the noisy labels extracted by the critic module.

2) Model-Based Critic Module: The model-based critic module obtains the set of potential channel assignment decisions from the actor module to select the best decision among them and solve the optimization problem for the remaining variables. Unlike the conventional approach that adopts another DNN for the critic module, our approach leverages the model information on the user-server communication and power consumption to evaluate each channel assignment decision analytically. Indeed, by fixing the setting for $\mathbf { x } _ { j } ^ { t } , j \in S$ it is feasible to find optimal settings for the remaining variables in $\mathbf { y } ^ { t }$ , which are all continuous. Specifically, let $( \mathbf { y } ^ { t } ) ^ { \star }$ denote the optimal decision for $\mathbf { y } ^ { t }$ and $J ^ { \star } \left( \tilde { \mathbf { x } } ^ { t } , \Xi ^ { t } \right)$ denote the optimal value of the objective function (21a) given $\tilde { \mathbf { x } } ^ { t }$ and $\Xi ^ { t }$ , P3 is equivalent to the problem, denoted as P4, of finding

$$
( \tilde { \mathbf { x } } ^ { t } ) ^ { \star } \triangleq \arg \operatorname* { m i n } J ^ { \star } \left( \tilde { \mathbf { x } } ^ { t } , \Xi ^ { t } \right) ,\tag{25}
$$

where $\tilde { \mathbf { x } } ^ { t }$ is one among k channel assignment decisions given by the actor module. We will introduce in detail the algorithm to obtain $J ^ { \star } \left( \tilde { \mathbf { x } } ^ { t } , \Xi ^ { t } \right)$ in Section V-B.

Using a model-based critic module brings the advantage of having an accurate evaluation for each decision on the channel assignment, thus improving the convergence of training. Besides, it is worth noting that to obtain $( \tilde { \mathbf { x } } ^ { t } ) ^ { \star }$ and $( \mathbf { y } ^ { t } ) ^ { \star }$ ， we need to evaluate k times the function $J ^ { \star } \left( \tilde { \mathbf { x } } ^ { t } , \Xi ^ { t } \right)$ . Thus, k is another hyper-parameter of the system that will affect the trade-off between performance and computational complexity. In general, larger values of k result in a better performance in terms of convergence time but require more computational resources.

3) Policy Update Module: The policy update module exploits training samples labeled by the critic module $( \mathrm { i . e . , }$ the pair $\{ \Xi ^ { t } , ( \bar { \tilde { \mathbf { x } } } ^ { t } ) ^ { \star } \} )$ to update the parameters of the actor module’s DNN. A replay memory of size $q$ is adopted to record the training samples. Only the most recent data samples are kept, i.e., new data will continuously replace the old ones to avoid memory bloat. Beginning with an empty memory, we start training the DNN only when at least $q / 2$ data samples are available. Afterward, the DNN is trained periodically once every $\delta _ { T }$ time slots. Such a training scheme helps prevent the DNN from overfitting with noise in the input and enables the neural network to adapt dynamically to the time-varying channel condition.

Specifically, a batch of training samples is randomly selected from the replay memory when mod $( t , \delta _ { T } ) = 0$ (mod indicates the modulo operation). We then use these samples to train the DNN by using the Adam algorithm [38] to minimize the cross-entropy cost function $L ( \Phi _ { t } )$ , given as

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \Phi _ { t } ) = \frac { - 1 } { | { \cal S } ^ { t } | } \sum _ { \tau \in { \cal S } ^ { t } } \left[ ( \tilde { \mathbf { x } } ^ { \tau } ) ^ { \mathsf { T } } \log \left( \Pi _ { \Phi ^ { t } } \left( \Xi ^ { \tau } \right) \right) \right. } } \\ { { \displaystyle \qquad + \left. ( 1 - \tilde { \mathbf { x } } ^ { \tau } ) ^ { \mathsf { T } } \log \left( 1 - \Pi _ { \Phi ^ { t } } \left( \Xi ^ { \tau } \right) \right) \right] } . } \end{array}\tag{26}
$$

In (26), $S ^ { t }$ denotes the set of time indices of data samples selected for training at time $t ; | S ^ { t } |$ denotes the size of ${ \mathcal { S } } ^ { t } ;$ $( \cdot ) ^ { \mathsf { T } }$ denotes the transpose operator; and log(·) denotes the element-wise logarithm operation of a vector. The cost function in (26) measures the goodness of the relaxed channel assignment (i.e., the output of the DNN) compared to the best decision selected by the critic module. Lower values of $L ( \Phi _ { t } )$ indicate good performance and that $\Phi _ { t }$ is appropriate in generalizing the mapping rule for channel assignment.

## B. Model-Based Optimization of the Critic Module

In this section, we present in detail the optimization algorithm used by the critic module to obtain $\bar { \boldsymbol { J } } ^ { \star } \left( \tilde { \mathbf { x } } ^ { t } , \Xi ^ { t } \right)$ in (25). For a given the channel assignment decision $\tilde { \mathbf { x } } ^ { t } =$ $\left\{ x _ { i , j } ^ { t } \right\} _ { i \in N , j \in S } ,$ we can obtain the optimal solution for $\mathbf { y } ^ { t } = \mathbf { \alpha }$ $\left\{ \alpha _ { i , j } ^ { t } , r _ { i , j } ^ { t } , f _ { l , i } ^ { t } , f _ { c , i } ^ { t } \right\} _ { i \in \mathcal { N } , j \in \mathcal { S } }$ by decomposing the per-time slot problem P3 into four sub-problems, including optimization for offloading volume and bandwidth allocation for the UAV and the mBS links, optimization for local computation, and optimization for UAV’s computational resource scheduling.

1) Optimization on Offloading Volume and Bandwidth Allocation for the UAV Link: Given a feasible channel assignment decision, the optimization variables related to the UAV includes the bandwidth allocation, $\begin{array} { r l } { \alpha _ { \mathrm { U A V } } ^ { t } } & { { } \triangleq } \end{array}$ $\{ \alpha _ { i , j } ^ { t } | i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } , j = \mathrm { U A V } \}$ , and the offloading volume on the

UAV link, $\mathbf { r } _ { \mathrm { U A V } } ^ { t } \triangleq \{ r _ { i , j } ^ { t } | i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } , j = \mathrm { U A V } \}$ . Let $\mathbf { x } _ { \mathrm { U A V } } ^ { t } \triangleq$ $\{ \alpha _ { \mathrm { U A V } } ^ { t } , \mathbf { r } _ { \mathrm { U A V } } ^ { t } \}$ denote the combination of these variables. The optimal decision for $\mathbf { x } _ { \mathrm { U A V } } ^ { t }$ can be obtained by solving the following problem, hereinafter referred to as P3.1:

$$
\begin{array} { r l } & { \displaystyle \underset { \mathbf { x } _ { \mathrm { U A V } } ^ { t } } { \mathrm { m i n } } - \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \left[ \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) - Q _ { i } ^ { s } ( t ) - Z _ { i } ^ { s } ( t ) \right) r _ { i , \mathrm { U A V } } ^ { t } \right] } \\ & { ~ + V \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \psi _ { i } \left[ \left( 2 ^ { \frac { r _ { i , \mathrm { U A V } } ^ { t } } { w _ { \mathrm { U A V } i , \mathrm { U A V } } ^ { t } T } } - 1 \right) \frac { N _ { 0 } W _ { \mathrm { U A V } } } { h _ { i , \mathrm { U A V } } ^ { t } } \right] } \end{array}\tag{27a}
$$

$$
\mathrm { s . t . } \quad \epsilon _ { A } \leq \alpha _ { i , \mathrm { U A V } } ^ { t } , \quad \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \alpha _ { i , \mathrm { U A V } } ^ { t } \leq 1 , \quad i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t }\tag{27b}
$$

$$
0 \leq r _ { i , \mathrm { U A V } } ^ { t } \leq \operatorname* { m i n } \left\{ Q _ { i } ^ { l } ( t ) , r _ { i , \mathrm { U A V } } ^ { t , \operatorname* { m a x } } \right\} , \quad i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } ,\tag{27c}
$$

where $\begin{array} { r } { r _ { i , \mathrm { U A V } } ^ { t , \mathrm { m a x } } = W _ { \mathrm { U A V } } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau \log _ { 2 } \left( 1 + \frac { p _ { \mathrm { T x } , i } ^ { \mathrm { m a x } } h _ { i , \mathrm { U A V } } ^ { t } } { N _ { 0 } W _ { \mathrm { U A V } } } \right) } \end{array}$ denotes the upper bound of $r _ { i , \mathrm { U A V } } ^ { t }$ using the maximum transmit power. Note that for all users that are not associated with the UAV in time slot $t ,$ the bandwidth and offloading volume assigned to them equal zero, i.e., $r _ { i , \mathrm { U A V } } ^ { t , \mathrm { m a x } } = 0 , \alpha _ { i , \mathrm { U A V } } ^ { t } = 0 , \forall i \notin \mathcal { N } _ { \mathrm { U A V } } ^ { t }$

To solve P3.1, we adopt the Gauss-Seidel approach [39] to optimize the offloading volume and the bandwidth allocation in an alternating manner. Specifically, in each iteration, the offloading volume decision is obtained in closed forms, and the bandwidth allocation is determined by the Lagrangian method. The alternating approach is guaranteed to converge to the optimal solution since P3.1 is convex, and the feasible region is a Cartesian product [39] of $\mathbf { r } _ { \mathrm { U A V } } ^ { t }$ and $\alpha _ { \mathrm { U A V } } ^ { t }$

a) Optimal offloading volume: For a feasible bandwidth allocation $\begin{array} { r } { \alpha _ { \mathrm { U A V } } ^ { t } , } \end{array}$ the optimal offloading volume for mobile devices in $\bar { \mathcal { N } } _ { \mathrm { U A V } } ^ { t }$ can be obtained by solving

$$
\begin{array} { c } { { { \bf { P 3 . 1 . 1 : ~ m i n i m i z e } } } } \\ { { { \bf { r } } _ { \mathrm { { U A V } } } ^ { t } } } \\ { { { \bf { s u b j e c t ~ t o } } } } \end{array}\tag{27a}
$$

(27c)

The optimal solution to the above problem is either the stationary point of (27a) or one of the boundary points. Specifically, for $\begin{array} { r l r l r l r } { i } & { { } \in } & { \mathcal { N } _ { \mathrm { U A V } } ^ { t } , } & { ( r _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } } & { { } = } & { \mathrm { ~ \widehat { ~ \theta ~ } ~ } } & { } \end{array}$ if $Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) \leq Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t )$ , otherwise, $( r _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } = \operatorname* { m a x } ^ { } .$  min $\{ \hat { r } _ { i , \mathrm { U A V } } ^ { t } , \dot { r } _ { i , \mathrm { U A V } } ^ { t , \operatorname* { m a x } } \} , 0 \}$ , where $\hat { r } _ { i , \mathrm { U A V } } ^ { t } =$ $\begin{array} { r } { W _ { \mathrm { U A V } } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau \times \log _ { 2 } \left( \frac { \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) - Q _ { i } ^ { s } ( t ) - Z _ { i } ^ { s } ( t ) \right) \alpha _ { i , \mathrm { U A V } } ^ { t } \tau h _ { i , \mathrm { U A V } } ^ { t } } { V \psi _ { i } \ln ( 2 ) N _ { 0 } } \right) } \end{array}$

b) Optimal bandwidth allocation: For a feasible offloading volume decision $\mathbf { r } _ { i } ^ { t }$ , the optimal bandwidth allocation can be obtained by solving

$$
\mathbf { P 3 . 1 . 2 : } \ \operatorname* { m i n } _ { \alpha _ { \mathrm { U A V } } ^ { t } } \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \psi _ { i } p _ { \mathrm { T x } , i } ^ { \mathrm { U A V } } ( t )\tag{28a}
$$

$$
\mathrm { s . t . } \epsilon _ { A } \leq \alpha _ { i , \mathrm { U A V } } ^ { t } , \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \alpha _ { i , \mathrm { U A V } } ^ { t } \leq 1 ,\tag{28b}
$$

$$
p _ { \mathrm { T x } , i } ^ { \mathrm { U A V } } ( t ) \leq p _ { \mathrm { T x } , i } ^ { \mathrm { m a x } } , i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } , P\tag{28c}
$$

where $\begin{array} { r l r } { p _ { \mathrm { T x } , i } ^ { \mathrm { U A V } } ( t ) } & { { } = } & { \left( 2 ^ { \frac { r _ { i , \mathrm { U A V } } ^ { t } } { W _ { \mathrm { U A V } } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau } } - 1 \right) \frac { N _ { 0 } W _ { \mathrm { U A V } } } { h _ { i , \mathrm { U A V } } ^ { t } } } \end{array}$ denotes the transmit power of user i on the UAV link. First, we observe that (28c) is equivalent to $\alpha _ { i , \mathrm { U A V } } ^ { t } \geq \alpha _ { i , \mathrm { U A V } } ^ { t , \mathrm { m i n } }$ , where

$$
\alpha _ { i , \mathrm { U A V } } ^ { t , \mathrm { m i n } } \triangleq \frac { r _ { i , \mathrm { U A V } } ^ { t } } { W _ { \mathrm { U A V } } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau \log _ { 2 } \left( 1 + \frac { p _ { \mathrm { T x } , i } ^ { \operatorname* { m a x } } h _ { i , \mathrm { U A V } } ^ { t } } { N _ { 0 } W _ { \mathrm { U A V } } } \right) }\tag{29}
$$

Then, the partial Lagrangian function associated with the above problem can be written as

$$
\mathcal { L } ( \alpha _ { i , \mathrm { U A V } } ^ { t } , \lambda _ { \mathrm { U A V } } ^ { t } ) \triangleq \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \psi _ { i } \left[ \left( 2 ^ { \frac { r _ { i , \mathrm { U A V } } ^ { t } } { W _ { \mathrm { U A V } } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau } } - 1 \right) \frac { N _ { 0 } W _ { \mathrm { U A V } } } { h _ { i , \mathrm { U A V } } ^ { t } } \right]
$$

$$
+ \lambda _ { \mathrm { U A V } } ^ { t } \left( \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \alpha _ { i , \mathrm { U A V } } ^ { t } - 1 \right) ,\tag{30}
$$

where $\lambda _ { \mathrm { U A V } } ^ { t } \geq 0$ is the Lagrangian multiplier associated with the constraint $\begin{array} { r } { \sum _ { i \in \mathcal { N } _ { \mathrm { H a V } } ^ { t } } \alpha _ { i , \mathrm { U A V } } ^ { t } \ \le \ 1 } \end{array}$ . Based on the Karush-Kuhn-Tucker (KKT) condition, the optimal bandwidth allocation $( \alpha _ { \mathrm { U A V } } ^ { t } ) ^ { \star }$ and the optimal Lagrangian multiplier $( \lambda _ { \mathrm { U A V } } ^ { t } ) ^ { \star }$ should satisfy the following equation set

$$
\left\{ \begin{array} { l l } { ( \alpha _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } = \operatorname* { m a x } \left\{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } ( ( { \lambda } _ { \mathrm { U A V } } ^ { t } ) ^ { \star } ) \right\} , i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } , } \\ { \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } ( \alpha _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } = 1 , } \end{array} \right.\tag{31}
$$

where $\hat { \epsilon } _ { A } = \operatorname* { m a x } \left\{ \epsilon _ { A } , \alpha _ { i , \mathrm { U A V } } ^ { t , \operatorname* { m i n } } \right\}$ and $\mathcal { R } _ { i , \mathrm { U A V } } ( \lambda _ { \mathrm { U A V } } ^ { t } )$ denotes the root of $\begin{array} { r } { \frac { \delta } { \delta \alpha _ { i , \mathrm { U A V } } ^ { t } } \mathcal { L } ( \alpha _ { i , \mathrm { U A V } } ^ { \hat { t } } , \lambda _ { \mathrm { U A V } } ^ { t } ) = 0 } \end{array}$ . The following proposition provides a close-form expression for $\mathcal { R } _ { i , \mathrm { U A V } } ( \lambda _ { \mathrm { U A V } } ^ { t } )$

Proposition 1: Given $\lambda _ { U A V } ^ { t } \quad \mathrm { ~ > ~ } \quad \mathrm { ~ 0 , ~ }$ the root of $\begin{array} { r } { \frac { \delta } { \delta \alpha _ { i , U A V } ^ { t } } \mathcal { L } ( \alpha _ { i , U A V } ^ { t } , \lambda _ { U A V } ^ { t } ) = \mathrm { ~ 0 ~ } } \end{array}$ is positive and unique as

$$
\mathcal { R } _ { i , U A V } ( \lambda _ { U A V } ^ { t } ) = \frac { r _ { i , U A V } ^ { t } \ln ( 2 ) } { 2 W _ { U A V T } \mathcal { W } \left( \frac { r _ { i , U A V } ^ { t } \ln ( 2 ) } { 2 W _ { U A V T } } \sqrt { \frac { \lambda _ { U A V } ^ { t } h _ { i , U A V } ^ { t } W _ { U A V T } } { \psi _ { i } \sigma _ { U A V } ^ { 2 } r _ { i , U A V } ^ { t } \ln ( 2 ) } } \right) } ,\tag{32}
$$

where W(·) denotes the Lambert-W function.

Proof: Please refer to Appendix B

It is observed that $\begin{array} { r } { \frac { \delta } { \delta \alpha _ { i , \mathrm { I I A V } } ^ { t } } \mathcal { L } ( \alpha _ { i , \mathrm { U A V } } ^ { t } , \lambda _ { \mathrm { U A V } } ^ { t } ) } \end{array}$ is a monotonic function with response to $\alpha _ { i , \mathrm { U A V } } ^ { t }$ . Thus, the optimal Lagrangian multiplier $\big ( \lambda _ { \mathrm { U A V } } ^ { t } \big ) ^ { \star }$ can be found using a bisection search over $[ \lambda _ { \mathrm { U A V } } ^ { L } ( t ) ; \lambda _ { \mathrm { U A V } } ^ { U } ( t ) ]$ in which $\lambda _ { \mathrm { U A V } } ^ { L } ( t )$ and $\lambda _ { \mathrm { U A V } } ^ { U } ( t )$ are selected so that $\textstyle \sum _ { i \in { \mathcal { N } } _ { \operatorname { U A V } } ^ { t } }$ max $\big \{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } \big ( \lambda _ { \mathrm { U A V } } ^ { L } ( t ) \big ) \big \} > 1$ and $\sum { _ { i \in \mathcal { N } _ { \mathrm { I 7 A V } } ^ { t } } }$ max $\{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } ^ { \mathrm { ~ w ~ } } ( \lambda _ { \mathrm { U A V } } ^ { U } ( t ) ) \} \ < \ 1$ . Specifically, the two boundaries can be selected as follows

$$
\left\{ \begin{array} { l } { \displaystyle \lambda _ { \mathrm { U A V } } ^ { L } ( t ) = \left. \operatorname* { m a x } _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \frac { - \delta p _ { \mathrm { T x } , i } ^ { \mathrm { U A V } } ( t ) } { \delta \alpha _ { i , \mathrm { U A V } } ^ { t } } \right. _ { \alpha _ { i , \mathrm { U A V } } ^ { t } = 1 } , } \\ { \displaystyle \lambda _ { \mathrm { U A V } } ^ { U } ( t ) = \left. \operatorname* { m i n } _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \frac { - \delta p _ { \mathrm { T x } , i } ^ { \mathrm { U A V } } ( t ) } { \delta \alpha _ { i , \mathrm { U A V } } ^ { t } } \right. _ { \alpha _ { i , \mathrm { U A V } } ^ { t } = \hat { \epsilon } _ { A } } , } \end{array} \right.\tag{33}
$$

The searching process for $\big ( \lambda _ { \mathrm { U A V } } ^ { t } \big ) ^ { \star }$ can terminate when $| \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } }$ max $\{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } ( \lambda _ { \mathrm { U A V } } ^ { U } ( t ) ) \} - 1 | \le \xi$ , where ξ is the accuracy of the algorithm. Details of the Lagrangian method for solving P3.1.2 are summarized in Algorithm 1.

```latex
Algorithm 1 Lagrangian Method for P3.1.2
1: Initialization: $\xi = 1 0 ^ { - 7 } , \tilde { \lambda } _ { L } = \lambda _ { \mathrm { U A V } } ^ { L } ( t ) , \tilde { \lambda } _ { U } = \lambda _ { \mathrm { U A V } } ^ { U } ( t )$
$l = 0 , I _ { \mathrm { m a x } } = 2 0 0 , \epsilon _ { A } = 1 0 ^ { - 4 } , \alpha _ { i , \mathrm { U A V } } ^ { t } = \hat { \epsilon } _ { A } , i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t }$
2: while $\begin{array} { r } { \Big | \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \operatorname* { m a x } \left\{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } ( \lambda _ { \mathrm { U A V } } ^ { U } ( t ) ) \right\} - 1 \Big | > \xi } \end{array}$
and $i \le ^ { \prime } I _ { \mathrm { m a x } }$ do
3: $\begin{array} { r } { \tilde { \lambda } = \frac { 1 } { 2 } \left( \tilde { \lambda } _ { L } + \tilde { \lambda } _ { R } \right) } \end{array}$ and $l = l + 1$
4: Set $\alpha _ { i , \mathrm { U A V } } ^ { t } = \operatorname* { m a x } \left\{ \hat { \epsilon } _ { A } , \mathcal { R } _ { i , \mathrm { U A V } } ( \tilde { \lambda } ) \right\} , i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t }$
5: if $\begin{array} { r } { \sum _ { i \in \mathcal { N } _ { \mathrm { U A V } } ^ { t } } \alpha _ { i , \mathrm { U A V } } ^ { t } \le 1 } \end{array}$ then
6: $\tilde { \lambda } _ { L } = \tilde { \lambda }$
7: else
8: $\tilde { \lambda } _ { U } = \tilde { \lambda }$
9: end if
10: end while
```

2) Optimization on Offloading Volume and Bandwidth Allocation for the mBS Link: Similar to the UAV link, given a feasible channel assignment decision, the optimization variables related to the mBS include the bandwidth allocation, $\alpha _ { \mathrm { m B S } } ^ { t } \triangleq$ $\left\{ \alpha _ { i , j } ^ { t } \middle | i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } , j = \mathrm { m B S } \right\}$ , and the offloading volume on the mBS link, $\mathbf { r } _ { \mathrm { m B S } } ^ { t } \triangleq \{ r _ { i , j } ^ { t } \} i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } , j = \mathrm { m B S } \}$ . Denoted by $\mathbf { x } _ { \mathrm { m B S } } ^ { t } \triangleq \{ \alpha _ { \mathrm { m B S } } ^ { t } , \mathbf { r } _ { \mathrm { m B S } } ^ { t } \}$ the combination of these variables, the optimal decisions for the mBS link can be obtained by solving

P3.2 :

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { \mathrm {  ~ \Lambda ~ } } - \sum _ { i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } } [ ( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) - ( r _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } ) r _ { i , \mathrm { m B S } } ^ { t } ] } } \\ & { } & { + V \sum _ { i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } } \psi _ { i } [ ( 2 ^ { \frac { r _ { i , \mathrm { m B S } } ^ { t } } { W _ { \mathrm { m B S } } ^ { t } \alpha _ { i , \mathrm { m B S } } ^ { t } } } - 1 ) \frac { N _ { 0 } W _ { \mathrm { m B S } } } { h _ { i , \mathrm { m B S } } ^ { t } } ] } \end{array}\tag{34a}
$$

$$
\mathrm { s . t . } ~ \epsilon _ { A } \leq \alpha _ { i , \mathrm { m B S } } ^ { t } , ~ \sum ~ \alpha _ { i , \mathrm { m B S } } ^ { t } \leq 1 ,\tag{34b}
$$

$$
\begin{array} { r l r } { \ } & { { } } & { \mathrm { ~ \ } ^ { \imath \in \mathcal { N } _ { \mathrm { m B S } } } } \\ { \ } & { { } } & { 0 \leq { r } _ { i , \mathrm { m B S } } ^ { t } \leq \operatorname* { m i n } \left\{ Q _ { i } ^ { l } ( t ) - \left( { r } _ { i , \mathrm { U A V } } ^ { t } \right) ^ { \star } , { r } _ { i , \mathrm { m B S } } ^ { t , \mathrm { m a x } } \right\} , } \\ { \ } & { { } } & { i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } \qquad ( \mathrm { \Sigma } ) } \end{array}\tag{4c}
$$

where $\begin{array} { r } { r _ { i , \mathrm { m B S } } ^ { t , \mathrm { m a x } } = W _ { \mathrm { m B S } } \alpha _ { i , \mathrm { m B S } } ^ { t } \tau \log _ { 2 } \left( 1 + \frac { p _ { \mathrm { T x } , i } ^ { \mathrm { m a x } } h _ { i , \mathrm { m B S } } ^ { t } } { N _ { 0 } W _ { \mathrm { m B S } } } \right) } \end{array}$ and the right-hand side (RHS) of (34c) denotes the upper bound of $r _ { i , \mathrm { m B S } } ^ { t }$ at time slot t. Similar to P3.1, we adopt the Gauss-Seidel method to solve $\alpha _ { \mathrm { m B S } } ^ { t }$ and $\mathbf { r } _ { \mathrm { m B S } } ^ { t }$ of P3.2 in an alternating manner, in which the optimal offloading volume is given in a close-form expression and the bandwidth allocation is determined by the Lagrangian method.

a) Optimal offloading volume: For a feasible bandwidth allocation $\alpha _ { \mathrm { m B S } } ^ { t }$ , the optimal offloading volume for mobile devices in $\mathcal { N } _ { \mathrm { m B S } } ^ { t }$ can be obtained by solving

$$
\begin{array} { c } { { { \bf P 3 . 2 . 1 : \ m i n i m i z e } \quad ( 3 4 \mathrm { a } ) } } \\ { { { \bf r } _ { \mathrm { m B S } } ^ { t } } } \\ { { \mathrm { s u b j e c t \ t o \ ( 3 4 c ) } } } \end{array}
$$

The optimal solution to the above problem is either the stationary point of (34a) or one of the boundary points. Specifically, for $\begin{array} { r l r l r l } { i } & { { } \in } & { \mathcal { N } _ { \mathrm { m B S } } ^ { t } , } & { ( r _ { i , \mathrm { m B S } } ^ { t } ) ^ { \star } } & { { } = } & { } \end{array}$ max  min $\{ \hat { r } _ { i , \mathrm { m B S } } ^ { t }$ , RHS of (34c) 	, 0	, where $\begin{array} { r l } { \hat { r } _ { i , \mathrm { m B S } } ^ { t } } & { { } = } \end{array}$ $\begin{array} { r } { W _ { \mathrm { m B S } } \alpha _ { i , \mathrm { m B S } } ^ { t } \tau \times \log _ { 2 } \left( \frac { \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) - \left( r _ { i , \mathrm { m B S } } ^ { t } \right) ^ { \star } \right) \alpha _ { i , \mathrm { m B S } } ^ { t } \tau h _ { i , \mathrm { m B S } } ^ { t } } { V \psi _ { i } \ln ( 2 ) N _ { 0 } } \right) } \end{array}$

b) Optimal bandwidth allocation: For a feasible offloading volume decision, the optimal bandwidth allocation on the mBS link can be obtained by solving the problem

$$
\mathbf { P 3 . 2 . 2 : } \operatorname* { m i n } _ { \alpha _ { \mathrm { m B S } } ^ { t } } \sum _ { i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } } \psi _ { i } p _ { \mathrm { T x } , i } ^ { \mathrm { m B S } } ( t )\tag{35a}
$$

$$
\mathrm { s . t . } \epsilon _ { A } \leq \alpha _ { i , \mathrm { m B S } } ^ { t } , \sum _ { i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t } } \alpha _ { i , \mathrm { m B S } } ^ { t } \leq 1 ,\tag{35b}
$$

$$
p _ { \mathrm { T x } , i } ^ { \mathrm { m B S } } ( t ) \leq p _ { \mathrm { T x } , i } ^ { \mathrm { m a x } } , i \in \mathcal { N } _ { \mathrm { m B S } } ^ { t }\tag{35c}
$$

We observe that P3.2.2 is similar to P3.1.2, in which one is for the SeNB, the other is for the MeNB. Thus, we can adopt Algorithm 1 to solve P3.2.2; the procedure is exactly the same by replacing j = UAV with $j = \mathrm { m B S }$

3) Optimization on User’s Local Computation: Given the optimal decision on offloading volume of each user, the problem of resource allocation for local computation can be decomposed for each individual $f _ { l , i } ^ { t }$

$$
\mathbf { P 3 . 3 : } \operatorname* { m i n } _ { \mathbf { f } _ { l } ^ { t } } \sum _ { i \in \mathcal { N } } \left( - Q _ { i } ^ { l } ( t ) \tau f _ { l , i } ^ { t } L _ { i } ^ { - 1 } + V \psi _ { i } \kappa _ { i } \left( f _ { l , i } ^ { t } \right) ^ { 3 } \right)\tag{36a}
$$

$$
\mathrm { s . t . ~ } 0 \leq { f } _ { l , i } ^ { t } \leq { f } _ { i } ^ { \operatorname* { m a x } } , \quad i \in \mathcal { N } ,\tag{36b}
$$

$$
l _ { i } ^ { t } \le Q _ { i } ^ { l } ( t ) - \left( r _ { i , \mathrm { U A V } } ^ { t } \right) ^ { \star } - \left( r _ { i , \mathrm { m B S } } ^ { t } \right) ^ { \star } , \quad i \in \mathcal { N }\tag{36c}
$$

in which constraint (36c) is added to ensure (14g) of the original problem P1. First, we observe that P3.3 is a convex problem since its objective function (36a) is convex and all constraints are linear. Furthermore, since both the objective function and constraints of P3.3 can be decomposed for each $f _ { l , i } ^ { t } ,$ the optimization for $\mathbf { f } _ { l } ^ { t }$ can be done by solving each $f _ { l , i } ^ { t }$ separately. Specifically, the optimal solution to P3.3 is either at the stationary point of the objective function (36a) or one of the boundary points as $\begin{array} { r l r } { ( f _ { l , i } ^ { t } ) ^ { \star } } & { = } & { \operatorname* { m i n } \left\{ F _ { i } , \sqrt { \frac { Q _ { i } ^ { l } ( t ) \tau } { 3 \kappa _ { i } V \psi _ { i } L _ { i } } } \right\} , i \mathrm { ~ \in ~ \cal { N } } . } \end{array}$ with $\begin{array} { r l } { F _ { i } } & { { } = } \end{array}$ min $\left\{ f _ { i } ^ { \operatorname* { m a x } } , \left( Q _ { i } ^ { l } ( t ) - ( r _ { i , \mathrm { U A V } } ^ { t } ) ^ { \star } - \left( \overline { { r } } _ { i , \mathrm { m B S } } ^ { t } \right) ^ { \star } \right) L _ { i } / \tau \right\}$

4) Optimization on UAV’s Computational Resource Scheduling: The optimal $\mathbf { f _ { c } } ^ { \star } ( t )$ can be obtained by solving

$$
\mathbf { P 3 . 4 } : \operatorname* { m i n } _ { \mathbf { f _ { c } ( \mathit { t } ) } } - \sum _ { i \in \mathcal { N } } Q _ { i } ^ { s } ( t ) \tau f _ { c , i } ^ { t } / L _ { s } + V \psi _ { c } \sum _ { i \in \mathcal { N } } \kappa _ { s } \left( f _ { c , i } ^ { t } \right) ^ { 3 }\tag{37a}
$$

$$
\mathrm { s . t . ~ 0 } \leq { f _ { c , i } ^ { t } } \leq { f _ { c } ^ { \operatorname* { m a x } } } , i \in \mathcal { N } ,
$$

$$
c _ { i } ^ { t } = \tau f _ { c , i } ^ { t } / L _ { s } \leq Q _ { i } ^ { s } ( t ) , i \in \mathcal { N }\tag{37b}
$$

(37c)

We observe that similar to P3.3, P3.4 is convex and the optimization can be decomposed into sub-problems that involve $f _ { c , i } ^ { t }$ separately. Specifically, the optimal solution for each $f _ { c , i } ^ { t }$ is either at the stationary point of the objective function or one of the boundary points as $\begin{array} { r } { ( f _ { c , i } ^ { t } ) ^ { \star } = \operatorname* { m i n } \left\{ f _ { c , i } ^ { \operatorname* { m a x } } , \sqrt { \frac { Q _ { i } ^ { s } ( t ) \tau } { 3 \kappa _ { s } V \psi _ { c } L _ { s } } } \right\} , i \in \mathcal { N } } \end{array}$ , where $f _ { c , i } ^ { \operatorname* { m a x } } =$ min $\{ f _ { c } ^ { \operatorname* { m a x } } , Q _ { i } ^ { s } ( t ) L _ { s } / \tau \}$

## VI. ALGORITHM ANALYSIS

In this section, we provide the computational complexity of the proposed scheme in solving the per-time slot problem, followed by an analysis of the scheme’s optimality.

## A. Computational Complexity

The calculation mainly comes from the model-based critic module because, in each time slot, this module must examine k potential channel assignment decisions made by the actor module to select the best one. Importantly, examining one decision involves solving the optimization of computation and communication of all users and servers. The time complexity in each problem is as follows.

1) Complexity of Joint Optimization on Offloading Volume and Bandwidth Allocation: . Given feasible offloading volumes, Algorithm 1 requires $\mathcal { O } \left( \log _ { 2 } \left( N / \xi \right) \right)$ iterations via a bi-section search to find the optimal bandwidth allocation, where N denotes the number of users and ξ specifies the algorithm accuracy $( { \bf e . g . } , \xi = 1 0 ^ { - 4 } )$ . Given a feasible bandwidth allocation, the optimization on the offloading volume for each user can be obtained in closed forms; thus, the complexity is in $\mathcal { O } ( N )$ considering N users. Consequently, the Gauss-Seidel joint optimization with $I _ { \mathrm { m a x } }$ iterations maximum is with complexity $\mathcal { O } \left( I _ { \mathrm { m a x } } \left( N + \log _ { 2 } \left( N / \xi \right) \right) \right)$ ).

Complexity of optimization on the computation of the user and the UAV. Since solutions to P3.3 and P3.4 can both be obtained in closed forms, the optimization is with complexity O(1) for each user.

In summary, the computational complexity for evaluating one channel assignment decision is $\dot { \mathcal { O } } ( I _ { \mathrm { m a x } } ( N \ +$ $\bar { \log _ { 2 } { ( N / \xi ) } } ) \big )$ . Since the critic module examines k potential decisions in each time slot, the overall computational complexity of the proposed scheme is in $\mathcal { O } \big ( k I _ { \operatorname* { m a x } } \big ( N + \log _ { 2 } \big ( N / \xi \big ) \big ) \big )$ with fast execution.1

## B. Optimality Analysis

We assume that the traffic arrival and channel gain fluctuation for each user is an independent and identically distributed (i.i.d) random process, denoted as $\omega ( t ) = \left\{ h _ { i , j } ^ { t } , A _ { i } ^ { t } \right\} _ { i \in \mathcal { N } , j \in \mathcal { S } } .$ A policy that observes ω(t) in each time slot and makes control decisions independent of the queue backlog is referred to as an ω-only policy.

To ensure the strong stability of queues, we assume that the following assumption holds, which is the Slater condition for Lyapunov optimization [28]. The asymptotic optimality of the proposed DRLRM scheme is then provided in Theorem 2.

Assumption 1: (Slater Condition) There are values $\epsilon > 0$ and $\Phi ( \epsilon )$ (where $0 \leq \Phi ( \epsilon ) \leq P _ { s y s } ^ { \operatorname* { m a x } } )$ and an ω-only policy Π making control decision $\alpha ^ { \Pi , t }$ in time slot t that satisfies

$$
\mathbb { E } \left[ P _ { s y s } ( t ) | \alpha ^ { \Pi , t } \right] = \Phi ( \epsilon ) ,
$$

$$
\mathbb { E } \left[ A _ { i } ^ { t } \right] \leq \mathbb { E } \left[ D _ { i } ^ { t } \big | \alpha ^ { \mathrm { I I } , t } \right] - \epsilon , \quad \forall i \in \mathcal { N } ,
$$

$$
\begin{array} { r } { \mathbb { E } \left[ r _ { i , U A V } ^ { t } \middle | \alpha ^ { \mathrm { I I } , t } \right] \leq \mathbb { E } \left[ c _ { i } ^ { t } \middle | \alpha ^ { \mathrm { I I } , t } \right] - \epsilon , \quad \forall i \in \mathcal { N } . } \end{array}\tag{38}
$$

Theorem 2: Suppose that the $\omega ( t )$ process is i.i.d over time slots, P1 is feasible, and the Slater condition holds for some ϵ and $\Phi ( \epsilon )$ . Suppose that the proposed DRLRM algorithm produces a C-additive approximation $( C \ge 0 )$ of the minimum of (22) every time slot, then the following statements hold.

(a) The average system power consumption satisfies

$$
\operatorname* { l i m } _ { t  \infty } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ P _ { s y s } ( t ) ] \leq P _ { s y s } ^ { * } + \frac { \hat { B } + C } { V } ,\tag{39}
$$

where $P _ { s y s } ^ { * }$ is the minimum average power cost achievable by any policy that meets the required constraints.

(b) All queues $Q _ { i } ^ { l } ( t ) , ~ Q _ { i } ^ { s } ( t ) , ~ Z _ { i } ^ { l } ( t ) , ~ Z _ { i } ^ { s } ( t )$ are mean rate stable and QoE constraints (3) and (4) are satisfied.

Proof: Please refer to Appendix C

## VII. NUMERICAL RESULTS AND DISCUSSION

In this section, simulation results are provided to evaluate the proposed scheme’s performance. In simulation, we consider $N = 8$ mobile devices placed randomly around a hot spot within a radius of 100 meters. The MeNB (i.e., the mBS) is 500 meters away from the users while the SeNB (i.e., the UAV) flies above them at a fixed altitude of 50 meters. Each MEC server serves at most two users at a time. The small-scale fading channel power gains in dB, $\tilde { h } _ { i , j } ^ { t }$ [dB], are normally distributed with mean $\mathbb { E } [ \tilde { h } _ { i , j } ^ { t } [ \mathrm { d B } ] ] = 0$ and standard variance $\sigma _ { \tilde { h } _ { i . i } ^ { t } [ \bf { d B } ] } \ = \ 4$ . The task arrival in each time slot follows the Poisson distribution with the same mean for all users $\mathbb { E } [ A _ { i } ^ { t } ] = \lambda = 1 5$ kbits, unless otherwise stated. Other parameters include $\kappa = 1 0 ^ { - 2 8 } ~ \mathrm { { W s ^ { 3 } / c y c l e ^ { 3 } , } ~ f _ { c } ^ { m a x } = 1 ~ G H z , }$ ψc = 0, 1, $N _ { 0 } ~ = ~ - 1 7 4$ dBm/Hz, g0 = −50 dB, γUAV = $\gamma _ { \mathrm { m B S } } ~ = ~ 2 . 7 6 0 1 . ~ L _ { i } ~ = ~ L _ { s } ~ = ~ 7 3 7 . 5 ~ \mathrm { c y c l e s / b i t } , ~ \psi _ { i } ~ = ~ 1$ $p _ { \mathrm { T x } , i } ^ { \mathrm { m a x } } = 2 0$ dBm, $f _ { i } ^ { \mathrm { m a x } } = 0 . 5$ GHz, $i \in \mathcal N$ . Each simulation is conducted over a time duration of $T = 1 0 0 0 0$ with slot length τ = 10 milliseconds.

For the proposed method, we set $\sigma _ { n } = 0 . 2 5$ and $k = 1 6$ to balance the exploration and exploitation for the actor module. The DNN consists of three 1-D convolutional layers, followed by a Flatten layer and three Dense layers; all are implemented using Tensorflow. The ReLU (rectified linear unit) activation is used for all layers except the last one with the sigmoid function.

Benchmarking schemes: To evaluate the performance, we use the following four benchmark schemes:

– Exhaustive Channel Assignment and Joint Resource Allocation (Exhaustive): The network investigate all possible decisions to select the best channel assignment.

– Queue Length-based Greedy Channel Assignment and Joint Resource Allocation (QL-JRA): The network prioritizes users having longest virtual queues $Z _ { i } ^ { l } ( t )$ to assign communication channels.

– Channel Gain-based Greedy Channel Assignment and Joint Resource Allocation (CG-JRA): Users with highest channel gains are prioritized to obtain communications resources.

– Random Channel Assignment Policy and Joint Resource Allocation $( R A \cdot J R A ) ;$ The network randomly assigns users to the mBS and the UAV.

TABLE III  
DETAILS OF TWO SCENARIOS FOR PERFORMANCE EVALUATION
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ScenarioI(EnergyEfficiency Comparison)</td><td rowspan=1 colspan=1>Scenario II (Stress Test for Queue Stability)</td></tr><tr><td rowspan=1 colspan=1>Objective</td><td rowspan=1 colspan=1>Investigate energy efficiency,given a strict constraint on thresholdsatwhich the average queue length must be less than or equal to.</td><td rowspan=1 colspan=1>Relaxconstraints on the queue length threshold,investigate the abilityto stabilize queues under very high computational traffic.</td></tr><tr><td rowspan=1 colspan=1>Arrival rate</td><td rowspan=1 colspan=1>Reasonable</td><td rowspan=1 colspan=1>Very high</td></tr><tr><td rowspan=1 colspan=1>Focused KPI</td><td rowspan=1 colspan=1>Power consumptionHow much energy is consumed to satisfy the predefined queue lengththreshold constraint?</td><td rowspan=1 colspan=1>Queue stabilityGiven that the arrival rate approximates_the maximum processingcapability,at which level are queues stable?</td></tr></table>

<!-- image-->  
(a) User's queue length

<!-- image-->  
(b) UAV's queue length

<!-- image-->  
(c) Weighted-sum system power  
Fig. 3. Performance in Scenario I (Energy efficiency comparison), λ = 10 kb, $Q _ { l } ^ { \mathrm { t h } } = 1 5$ kb, $Q _ { s } ^ { \mathrm { t h } } = 3$ kb, k = 16, $V = 1 0 ^ { 1 0 }$

These benchmark schemes differentiate in the approach to solving $\mathbf { x } ^ { t }$ for channel assignment. However, all apply the critic module’s optimization procedure specified in section V-B to optimize the remaining variables in $\mathbf { y } ^ { t }$

It is noteworthy that the exhaustive method is considered the optimal solution for benchmarking. However, finding the optimal solution via exhaustive search is not feasible if we consider a system with many users.

Performance evaluation scenarios: To evaluate the degree of suboptimality and convergence, we compare the proposed DRLRM method with the benchmark schemes in the two scenarios described in Table III.

– Scenario I: Power consumption is considered the main KPI for performance evaluation since all methods can satisfy the queue stability constraints given a reasonable computational load. The arrival rate is set at a reasonable level, λ = 10 kb. Accordingly, the queue length thresholds are $Q _ { l } ^ { \mathrm { t h } } = 1 5$ kb and $Q _ { s } ^ { \mathrm { t h } } = 3$ kb. For the proposed DRLRM method, the number of generated actions is set at k = 16 (i.e., 2% of the search space).

– Scenario II: A very high computational load with $\lambda =$ 30 kb is set to demonstrate high-traffic periods. All methods are expected to run at the highest energy level virtually all the time to stabilize queues. Thus, power consumption is not our focus. The queue length threshold is relaxed (i.e., $Q _ { l } ^ { \mathrm { t h } } ~ = ~ Q _ { s } ^ { \mathrm { t h } } ~ = ~ \infty )$ to facilitate the optimization. For the proposed method, $k = 8 0$ is set (approximately 10% of the search space).

Performance in Scenario I: In Fig. 3, we compare all methods in Scenario I, where we focus on the power consumption given predefined queue threshold constraints. Each point in the figure is a moving average of 1500 time slots. From Fig. 3a and Fig. 3b, we observe that thanks to the critic module’s optimization, all methods can keep the user’s and the UAV’s queues stable at levels lower than or equal to the predefined thresholds (i.e., queue threshold constraints are satisfied). From Fig. 3c, we observe a power consumption reduction for the proposed scheme over time. In the early stage, the proposed method consumes much more power than the optimal channel assignment (as much as the QL-JRA method). In the later phase, the scheme’s power consumption gently decreases over time and eventually converges to the same level as if the optimal decision was selected. This result proves the effectiveness of the training when the DNN gradually learns from experience to mimic the optimal policy. With the considered settings, our proposed method provides a reduction of 13.4%, 26.6%, and 30.5% compared to the QL-JRA, CH-JRA, and RA-JRA schemes, respectively.

Performance in Scenario II: Figure 4 illustrates the performance of all schemes in Scenario II, where we focus on the queue stability KPI under very high traffic. Each point in the figure is a moving average of 1500 time slots. We observe that while the UAV’s queue is kept stable by all methods, only the optimal channel assignment and the proposed scheme can cause local queues of users to be stable. Specifically, the user’s backlog queue of the QL-JRA, CH-JRA, and RA-JRA schemes increases almost linearly over time, indicating that the queue is not stable. This is because, without a proper policy for channel assignment, the achievable task processing capability is very limited and cannot afford the given task arrival rate. In contrast, the proposed method’s user queue length also increases rapidly in the early stage but decreases gradually in the later phase. In the end, the scheme’s user queue length is almost the same as that of the optimal channel assignment. This result once again demonstrates the effectiveness and convergence performance of the proposed DRLRM framework, even in unfavorable circumstances with a very high computational workload.

Effect of the parameter k: In Fig. 5, we investigate the convergence behavior of the proposed DRLRM method under different settings of the hyper-parameter k in Scenario II. The moving average rolling over 2000 time slots of the user queue length is plotted. The number of potential actions of the actor module (k) is set at 40, 80, and 120 for evaluation. The figure clearly shows that the hyperparameter plays a critical role in the convergence speed of the proposed method. Higher values of k help the DNN learn the optimal channel assignment policy faster. They speed up the convergence at the cost of higher computational resources required for the critic module to investigate the generated potential decisions. Specifically, the time duration until convergence of the proposed method (within a 1% gap compared to the optimal decision) for k = 120 and $k = 8 0$ are 6000 and 8000 time slots, respectively. This is because, with more decisions generated, the critic module has more chance to extract good approximations of the global minimum of the per-time slot problem. In other words, by investigating more potential decisions, the critic module helps improve the quality of the training data. In the long term, this advantage speeds up the learning process.

<!-- image-->  
(a) User's queue length

<!-- image-->  
(b) UAV's queue length

<!-- image-->  
(c） Weighted-sum system power

Fig. 4. Performance in Scenario II (Stress test for queue stability), λ = 30 kb, $Q _ { l } ^ { \mathrm { t h } } = Q _ { s } ^ { \mathrm { t h } } = \infty .$ , k = 80, $V = 1 0 ^ { 1 0 }$  
<!-- image-->  
Fig. 5. Convergence behavior of the proposed scheme in Scenario II, λ = 30 kb, $Q _ { l } ^ { \mathrm { t h } } = Q _ { s } ^ { \mathrm { t h } } = \infty .$ $V = 1 0 ^ { 1 0 }$

<!-- image-->

Effect of the queue length threshold: Fig. 6a illustrates the impact of the queue threshold on the average system queue length when changing V . It is noteworthy that the queue length threshold can be referred to as a means of controlling the service delay since the average task execution delay is proportional to the average queue length. We observe that if the queue stability level is not constrained, the average queue length increases almost linearly with the Lyapunov parameter. In addition, a higher task arrival rate leads to an increase in the average queue length. The interesting point is when we add additional constraints (3) and (4) to enforce the stability level for the queue length. We notice that with small values of V , the queue length threshold constraint does not affect the optimization. When V increases, the average queue length is effectively controlled so that the constraint is satisfied for all considered task arrival rates. Note that in case λ = 10 kb, the queue stability level with and without constraints are almost the same since the setting of V is not large enough to make the average queue length surpass the threshold.

(a) Queue length  
<!-- image-->  
(b) System power  
Fig. 6. Effect of the queue length threshold constraints.

Fig. 6b investigates the impact of the queue length threshold on the system power consumption when changing V . We observe that the power consumption of all settings decreases, corresponding to the increase of the parameter V , as expected. This is because increasing V puts more emphasis on the system power consumption than stabilizing the queue length in the per-time slot problem. In addition, we notice that as the arrival rate increases, the power consumption increases accordingly. The queue threshold constraint also significantly impacts power consumption. The gap between the two scenarios (with and without the constraint) tends to enlarge with increasing the arrival rate and control parameter V . This is because the network has no choice but to consume more energy to maintain queues stable at a satisfactory level (as depicted in Fig. 6a), especially in cases with high arrival rates.

Power-delay trade-off: In Fig. 7, we investigate the trade-off between the average weighted sum system power consumption and the average task computation delay by varying the Lyapunov control parameter V with unconstrained queue thresholds. As can be observed, the average task computation delay increases as the power consumption decreases, indicating that a proper setting of V is critical to balance the two objectives. Besides, we observe that given a specific execution delay level, the average weighted sum power consumption increases with the task arrival rate. The result is logical since more power consumption is required to keep the queue stable when the workload grows.

<!-- image-->  
Fig. 7. Power consumption vs. task completion delay, $Q _ { l } ^ { \mathrm { t h } } = Q _ { s } ^ { \mathrm { t h } } = \infty$ $V ^ { ' } = 1 0 ^ { 8 } , 1 0 ^ { 9 } , \{ 2 . 5 , 5 , ^ { ' } 7 . 5 \} \times 1 0 ^ { 9 } , 1 0 ^ { 1 0 }$

## VIII. CONCLUSION

This paper proposes a hybrid method that combines conventional model-based optimization and model-free DRL to minimize the power consumption of a multi-user, multiserver MEC network. Via DC, one user can offload tasks to the macro base station and the UAV-mounted MEC server simultaneously. The power minimization is formulated as a multi-stage MINLP problem with long-term constraints of queue stability and average task computation delay. Lyapunov optimization is exploited to transform the original multi-stage problem into a per-time slot problem, which is then solved using a DRL framework. Theoretical analyses are provided to demonstrate the proposed method’s optimality and computational complexity. Extensive simulations show that the proposed framework can produce nearly the same performance as the optimal solution obtained via an exhaustive search. In future research, it would be interesting to investigate the impact on system performance of the UAV-ground base station collaboration and the adaptive deployment of multiple UAVs. Other research directions, such as multi-layer edge computing, vertical networks of edge servers, and quality of experienceaware deployment, should also be investigated.

## APPENDIX A

PROOF OF THEOREM 1

To begin with, let $\begin{array} { r c l } { { { \bf Q } ^ { l } } ( t ) } & { \triangleq } & { \{ Q _ { i } ^ { l } ( t ) \} _ { i \in \mathcal { N } } , { \bf Q } ^ { s } ( t ) } \end{array}$ ≜ $\begin{array} { r l r } { \{ Q _ { i } ^ { s } ( t ) \} _ { i \in \mathcal { N } } , \mathbf { Z } ^ { l } ( t ) } & { { } \stackrel { \triangle } { = } } & { \big \{ Z _ { i } ^ { l } ( t ) \big \} _ { i \in \mathcal { N } } } \end{array}$ , and ${ \bf Z } ^ { s } ( t )$ ≜ $\{ Z _ { i } ^ { s } ( t ) \} _ { i \in \mathcal { N } }$ denote the system state variables. We then define the Lyapunov function $\mathcal { L } ( \cdot )$ and the drift function $\Delta ( \cdot )$ for ${ \bf Q } ^ { l } ( t ) , \bar { { \bf Q } } ^ { s } ( t ) , { \bf Z } ^ { l } ( t )$ , and ${ \bf Z } ^ { s } ( t )$ in a similar way as for $\Theta ( t )$ in (17) and (18), respectively, where the conditional expectation is taken given $\Theta ( t )$ . The following lemmas give the upper bound of the drift function for each of the above system state. Note that in what follows, $[ x ] ^ { + }$ denotes the function max {x, 0}. Additionally, since the communication and computation resources at the user and the UAV are limited, we denote the upper bound of $D _ { i } ^ { t } , \ r _ { i } ^ { t }$ and $c _ { i } ^ { t }$ as $D _ { i , \mathrm { m a x } } , \ r _ { i , \mathrm { m a x } }$ and $c _ { i , \mathrm { m a x } } ,$ respectively. Similarly, Ai,max denotes the user $i \gamma _ { \mathrm { s } }$ maximal task arrival in a time slot.

Lemma 1: The drift function for $\mathbf { Q } ^ { l } ( t )$ is bounded as

$$
\Delta ( \mathbf { Q } ^ { l } ( t ) ) \leq B _ { 1 } - \sum _ { i \in \mathcal { N } } \mathbb { E } [ Q _ { i } ^ { l } ( t ) ( l _ { i } ^ { t } + r _ { i } ^ { t } - A _ { i } ^ { t } ) | \Theta ( t ) ] ,\tag{40}
$$

where $\begin{array} { r } { B _ { 1 } = \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } \left( D _ { i , \operatorname* { m a x } } ^ { 2 } + A _ { i , \operatorname* { m a x } } ^ { 2 } \right) , r _ { i } ^ { t } = r _ { i , U A V } ^ { t } + r _ { i , m B S } ^ { t } . } \end{array}$

Proof: We have $Q _ { i } ^ { l } ( t { + } 1 ) ^ { 2 } = \left( \left[ Q _ { i } ^ { l } ( t ) - D _ { i } ^ { t } \right] ^ { + } + A _ { i } ^ { t } \right) ^ { 2 } \underline { { \stackrel { ( \dag ) } { = } } }$ $( Q _ { i } ^ { l } ( t ) - D _ { i } ^ { t } + A _ { i } ^ { t } ) ^ { 2 } = Q _ { i } ^ { l } ( t ) ^ { 2 } - 2 \ Q _ { i } ^ { l } ( t ) ( D _ { i } ^ { t } - A _ { i } ^ { t } ) + ( D _ { i } ^ { t } -$ ${ \dot { A } } _ { i } ^ { t } ) ^ { 2 }$ , where (†) is obtained since $D _ { i } ^ { t } = l _ { i } ^ { t } + r _ { i } ^ { t } \le Q _ { i } ^ { l } ( t )$ as in (14g). By some algebraic transformations, we have $\begin{array} { r } { \frac { 1 } { 2 } \left( Q _ { i } ^ { l } ( t + 1 ) ^ { 2 } - Q _ { i } ^ { l } ( t ) ^ { 2 } \right) \leq - Q _ { i } ^ { l } ( t ) ( D _ { i } ^ { t } - A _ { i } ^ { t } ) + \frac { 1 } { 2 } ( D _ { i } ^ { t } - A _ { i } ^ { t } ) ^ { 2 } } \end{array}$ $\begin{array} { r } { \bar { \leq } - Q _ { i } ^ { l } ( t ) ( D _ { i } ^ { t } - A _ { i } ^ { t } ) + \frac { 1 } { 2 } \left( D _ { i , \operatorname* { m a x } } ^ { 2 } + A _ { i , \operatorname* { m a x } } ^ { 2 } \right) } \end{array}$ By taking the conditional expectation on both sides of the inequation and summing up over all users $i \in \mathcal N$ , we obtain (40). □

Lemma 2: The drift function $\Delta \left( \mathbf { Z } ^ { l } ( t ) \right)$ is upper bounded by

$$
B _ { 2 } - \sum _ { i \in { \cal N } } \mathbb { E } \left[ Z _ { i } ^ { l } ( t ) \left( l _ { i } ^ { t } + r _ { i } ^ { t } - Q _ { i } ^ { l } ( t ) - A _ { i } ^ { t } + Q _ { l , i } ^ { t h } \right) \big | \Theta ( t ) \right] ,\tag{41}
$$

where $\begin{array} { r } { B _ { 2 } = \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } \left[ D _ { i , \operatorname* { m a x } } ^ { 2 } + Q _ { i } ^ { l } ( t ) ^ { 2 } + ( A _ { i } ^ { t } ) ^ { 2 } + ( Q _ { l , i } ^ { t h } ) ^ { 2 } + \right. } \end{array}$ $D _ { i , \operatorname* { m a x } } Q _ { l , i } ^ { t h } + \mathbf { \bar { { Q } } } _ { i } ^ { l } ( t ) \mathbf { \bar { { A } } } _ { i } ^ { t } ]$ and $r _ { i } ^ { t } = r _ { i , U A V } ^ { t } + r _ { i , m B S } ^ { t } .$

Proof: From (15), we have

$$
\begin{array} { r l r } { Z _ { i } ^ { l } ( t + 1 ) ^ { 2 } \stackrel { \mathrm { ( ) } } { \leq } } & { \left( Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { l } ( t + 1 ) - Q _ { l , i } ^ { \mathrm { t h } } \right) ^ { 2 } } & \\ { \stackrel { \mathrm { ( ) } } { = } } & { \left( Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { l } ( t ) - D _ { i } ^ { t } + A _ { i } ^ { t } - Q _ { l , i } ^ { \mathrm { t h } } \right) ^ { 2 } } & \\ & { = Z _ { i } ^ { l } ( t ) ^ { 2 } - 2 \ Z _ { i } ^ { l } ( t ) \left( D _ { i } ^ { t } - Q _ { i } ^ { l } ( t ) - A _ { i } ^ { t } + Q _ { l , i } ^ { \mathrm { t h } } \right) } & \\ & { \quad + \ \left( D _ { i } ^ { t } - Q _ { i } ^ { l } ( t ) - A _ { i } ^ { t } + Q _ { l , i } ^ { \mathrm { t h } } \right) ^ { 2 } } & { ( 4 \varDelta _ { i } ^ { \prime } ) } \end{array}\tag{2}
$$

where (†) is due to the fact that (max $\left\{ a - b , 0 \right\} ) ^ { 2 } \leq \left( a - b \right) ^ { 2 } ;$ and (‡) is with condition (14g) of P1 that $D _ { i } ^ { l } ~ = ~ l _ { i } ^ { t } +$ $\begin{array} { r l r } { r _ { i } ^ { t } } & { { } \le } & { Q _ { i } ^ { l } ( t ) } \end{array}$ . Thus, we have $\begin{array} { r l } { \frac { 1 } { 2 } \left( Z _ { i } ^ { l } ( t + 1 ) ^ { 2 } - Z _ { i } ^ { l } ( t ) ^ { 2 } \right) } & { { } \overset { ( \dagger ) } { \leq } } \end{array}$ $\begin{array} { r } { - Z _ { i } ^ { l } ( t ) \left( D _ { i } ^ { t } - Q _ { i } ^ { l } ( t ) - A _ { i } ^ { t } + Q _ { l , i } ^ { \mathrm { t h } } \right) \ + \ \frac { 1 } { 2 } \Big [ D _ { i , \mathrm { m a x } } ^ { 2 } \ + \ Q _ { i } ^ { l } ( t ) ^ { 2 } \ + \ } \end{array}$ $( A _ { i } ^ { t } ) ^ { 2 } + ( Q _ { l , i } ^ { \mathrm { t h } } ) ^ { 2 } + D _ { i , \operatorname* { m a x } } Q _ { l , i } ^ { \mathrm { t h } } + Q _ { i } ^ { l } ( t ) A _ { i } ^ { t } |$ where (†) is obtained since $( a - b ) ^ { 2 } \leq a ^ { 2 } + b ^ { 2 }$ for $a b \geq 0$ and $0 \leq D _ { i } ^ { t } \leq D _ { i , \operatorname* { m a x } } .$ By replacing $D _ { i } ^ { t }$ by ${ l _ { i } ^ { t } + r _ { i } ^ { t } }$ , taking the conditional expectation on both sides of the inequation and summing up over all users $i \in \mathcal { N } .$ , we obtain (41). □

Lemma 3: The drift function for $\mathbf { Q } ^ { s } ( t )$ is bounded as

$$
\Delta \left( \mathbf { Q } ^ { s } ( t ) \right) \leq B _ { 3 } - \sum _ { i \in \mathcal { N } } \mathbb { E } \left[ Q _ { i } ^ { s } ( t ) \left( c _ { i } ^ { t } - r _ { i , U A V } ^ { t } \right) \Big | \Theta ( t ) \right] ,\tag{43}
$$

where $\begin{array} { r } { B _ { 3 } = \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } \big ( c _ { i , \operatorname* { m a x } } ^ { 2 } + r _ { i , \operatorname* { m a x } } ^ { 2 } \big ) . } \end{array}$

Proof: The proof is similar to that of Lemma 1.

Lemma 4: The drift function $\Delta \left( \mathbf { Z } ^ { s } ( t ) \right)$ is upper bounded $b y$

$$
B _ { 4 } - \sum _ { i \in \mathcal { N } } \mathbb { E } \left[ Z _ { i } ^ { s } ( t ) \left( c _ { i } ^ { t } - Q _ { i } ^ { s } ( t ) - r _ { i , U A V } ^ { t } + Q _ { s , i } ^ { t h } \right) \middle | \Theta ( t ) \right] ,\tag{44}
$$

where $\begin{array} { r } { B _ { 4 } = \frac { 1 } { 2 } \sum _ { i \in \mathcal { N } } \left[ c _ { i , \operatorname* { m a x } } ^ { 2 } + Q _ { i } ^ { s } ( t ) ^ { 2 } + r _ { i , \operatorname* { m a x } } ^ { 2 } + ( Q _ { s , i } ^ { t h } ) ^ { 2 } + \right. } \end{array}$ $c _ { i , \operatorname* { m a x } } Q _ { s , i } ^ { t h } + Q _ { i } ^ { s } ( t ) r _ { i , \operatorname* { m a x } } ] ,$

Proof: The proof is similar to that of Lemma 2.

It is straightforward that $\begin{array} { r c l } { \Delta \left( \Theta ( t ) \right) } & { = } & { \Delta \left( \mathbf { Q } ^ { l } ( t ) \right) } \end{array}$ + $\Delta \left( \mathbf { Q } ^ { s } ( t ) \right) + \Delta \left( \mathbf { Z } ^ { l } ( t ) \right) + \Delta \left( \mathbf { Z } ^ { s } ( t ) \right)$ . Thus, by summing up the left hand sides of (40), (41), (43), (44) we obtain the upper bound of the drift-plus-penalty as in (20), where $\begin{array} { r } { \hat { B } = B _ { 1 } + B _ { 2 } + B _ { 3 } + B _ { 4 } + \sum _ { i \in \mathcal { N } } \left[ Z _ { i } ^ { l } ( t ) \left( Q _ { i } ^ { l } ( t ) - Q _ { l , i } ^ { \mathrm { t h } } \right) + \right. } \end{array}$ $Z _ { i } ^ { s } ( t ) \left( Q _ { i } ^ { s } ( t ) - Q _ { s , i } ^ { \operatorname { t h } } \right) \bigg ]$ . We observe that $\hat { B }$ consists of constant terms from the observation at the beginning of time slot t, thus can be put aside from the optimization of the target control variables $\bar { \mathbf { X } } ^ { t }$

## APPENDIX B PROOF OF PROPOSITION 1

Let $\mathcal { L } ^ { \prime }$ denote the derivative of $\mathcal { L } ( \alpha _ { i , \mathrm { U A V } } ^ { t } , \lambda _ { \mathrm { U A V } } ^ { t } )$ with response to $\alpha _ { i , \mathrm { U A V } } ^ { t }$ . we have $\begin{array} { r } { \mathcal { L } ^ { \prime } = \frac { \delta } { \delta \alpha _ { i , \mathrm { I A V } } ^ { t } } \mathcal { L } ( \alpha _ { i , \mathrm { U A V } } ^ { t } , \lambda _ { \mathrm { U A V } } ^ { t } ) = } \end{array}$ $\begin{array} { r l r } { - \frac { \psi _ { i } \sigma _ { j } ^ { 2 } r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } { h _ { i , j } ^ { t } W _ { j } \tau ( \alpha _ { i , \mathrm { U A V } } ^ { t } ) ^ { 2 } } \exp \left( \frac { r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } { W _ { j } \alpha _ { i , \mathrm { U A V } } ^ { t } \tau } \right) } & { { } + } & { \lambda _ { \mathrm { U A V } } ^ { t } } \end{array}$ The first term of ${ \mathcal { L } } ^ { \prime } ,$ denoted as $\ddot { F } ( \alpha _ { i , \mathrm { U A V } } ^ { t } )$ , is an increasing function with response $\begin{array} { r l } { \mathrm { t o } } & { { } \quad \alpha _ { i , \mathrm { U A V } } ^ { t } . } \end{array}$ Furthermore, we have $\begin{array} { r l r } { \operatorname* { l i m } _ { \alpha _ { i , \mathrm { U A V } } ^ { t } \to 0 ^ { + } } F ( \alpha _ { i , \mathrm { U A V } } ^ { t } ) } & { { } \quad = } & { \quad - \infty } \end{array}$ and $\begin{array} { r } { \operatorname* { l i m } _ { \alpha _ { i , \mathrm { U A V } } ^ { t } \to + \infty } F ( \alpha _ { i , \mathrm { U A V } } ^ { t } ) = 0 } \end{array}$ . Thus, given $\lambda _ { \mathrm { U A V } } ^ { t } > 0$ , the root of equation $\begin{array} { r } { \mathcal { L } ^ { \prime } = 0 } \end{array}$ is positive and unique. Solving $\mathcal { L } ^ { \prime } = 0$ ， we have $\begin{array} { r l r } { \frac { 1 } { \left( \alpha _ { i , \mathrm { U A V } } ^ { t } \right) ^ { 2 } } \exp \left( \frac { r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } { W _ { \mathrm { U A V } } \tau } \frac { 1 } { \alpha _ { i , \mathrm { U A V } } ^ { t } } \right) } & { = } & { \frac { \lambda _ { \mathrm { U A V } } ^ { t } \tilde { h } _ { i , j } ^ { t } W _ { \mathrm { U A V } } \tau } { \psi _ { i } \sigma _ { j } ^ { 2 } r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } } \end{array}$ We recognize that the equation has the form $x ^ { 2 } \exp ( a x ) = c ,$ where $\begin{array} { r l r } { \bar { { { x } } } } & { = } & { \frac { 1 } { \alpha _ { i , \mathrm { U A V } } ^ { t } } , \mathrm { ~ } a \mathrm { ~ \ } = \frac { r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } { W _ { j } \tau } , \mathrm { ~ } c \mathrm { ~ \ } = \frac { \lambda _ { \mathrm { U A V } } ^ { t - } \hat { h } _ { i , j } ^ { t } \hat { W } _ { j } \tau } { \psi _ { i } \sigma _ { j } ^ { 2 } r _ { i , \mathrm { U A V } } ^ { t } \ln ( 2 ) } . } \end{array}$ Solving the equation using the Lambert-W function yields $\begin{array} { r } { \alpha _ { i , \mathrm { U A V } } ^ { t } = \frac { a } { 2 \mathcal { W } \left( \frac { a \sqrt { c } } { 2 } \right) } } \end{array}$ . Substituting a and $c ,$ we obtain (32).

## APPENDIX C PROOF OF THEOREM 2

To begin with, we introduce the following lemma for feasibility of problem P1.

Lemma 5: Suppose that $\omega ( t )$ is stationary. If P1 is feasible, then for any $\delta > 0$ , there is an ω-only policy Π that satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ P _ { s y s } ( t ) \middle | \alpha ^ { \Pi , t } \right] \leq P _ { s y s } ^ { * } + \delta } \\ & { \qquad \mathbb { E } \left[ A _ { i } ^ { t } \right] \leq \mathbb { E } \left[ { D } _ { i } ^ { t } \middle | \alpha ^ { \Pi , t } \right] + \delta , \forall i \in \mathcal { N } } \\ & { \mathbb { E } \left[ r _ { i , U A V } ^ { t } \middle | \alpha ^ { \Pi , t } \right] \leq \mathbb { E } \left[ { c } _ { i } ^ { t } \middle | \alpha ^ { \Pi , t } \right] + \delta , \forall i \in \mathcal { N } } \end{array}\tag{45}
$$

Proof: Please see Theorem 4.5 of [28] for detailed proof. □

Proof of statement (a). Considering an ω-only policy Π with a corresponding value $\delta > 0 _ { : }$ , applying Lemma 5 into the right-hand side (RHS) of (20) yields

$$
\begin{array} { r l r } {  { \Delta ( \Theta ( t ) ) + V \mathbb { E } [ P _ { \mathrm { s y s } } ( t ) ] } } \\ & { \stackrel { \mathrm { ( ) } } { \leq } \hat { B } + C - \sum _ { i \in N } \mathbb { E } [ ( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) ) ( D _ { i } ^ { t } - A _ { i } ^ { t } ) ] } \\ & { } & { - \sum _ { i \in N } \mathbb { E } [ ( Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) ) ( c _ { i } ^ { t } - r _ { i , \mathrm { U A V } } ^ { t } ) ] + V \mathbb { E } [ P _ { \mathrm { s y s } } ( t ) ] } \\ & { \stackrel { \mathrm { ( ) } } { \leq } \hat { B } + C + \delta \sum _ { i \in N } ( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) ) } \\ & { } & { + V ( P _ { \mathrm { s y s } } ^ { * } + \delta ) , \qquad \quad ( 4 6 } \end{array}
$$

where (†) is because the ω-only policy Π is independent of the queue backlog Θ(t) and $\omega ( t )$ is i.i.d over time; (‡) is obtained by plugging in (45). By letting $\delta  0$ , we obtain $\Delta \left( \Theta ( t ) \right) + V \mathbb { E } \left[ P _ { \mathrm { s y s } } ( t ) \right] \leq \hat { B } + C + V P _ { \mathrm { s v s } } ^ { * }$ . By summing up both sides from $t = 0 ~ { \mathrm { t o } } ~ T - 1$ , taking iterated expectation and telescoping sums, then dividing both sides by $T V$ , we obtain $\begin{array} { r } { \frac { 1 } { T V } \mathbb { E } \left[ \bar { \mathcal { L } } ( \bar { \Theta } ( T ) ) \right] - \frac { 1 } { T V } \mathbb { E } \left[ \mathcal { L } ( \bar { \Theta ( 0 ) } ) \right] + \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ P _ { \mathrm { s y s } } ( t ) \right] \leq } \end{array}$ $( \hat { B } + C ) / V + P _ { \mathrm { s y s } } ^ { * }$ . Taking the limit on both sides of the inequation as $T  \infty$ , we obtain (39). This concludes the proof of statement (a).

Proof of statement (b). We consider an ω-only policy Π that satisfies the Slater condition in Assumption 1. Plugging (38) into (†) of (46), we obtain $\begin{array} { r l } { \Delta \left( \Theta ( \bar { t } ) \right) + V \mathbb { E } \left[ \bar { P _ { \mathrm { s y s } } } ( \bar { t } ) \right] } & { \leq } \end{array}$ $\begin{array} { r } { \hat { B } + C - \epsilon \sum _ { i \in \mathcal { N } } \big ( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) \big ) + V \bar { \Phi } ( \epsilon ) } \end{array}$ By taking integrated expectations, summing the telescoping series, and rearranging terms, we obtain $\textstyle { \frac { 1 } { T } } \sum _ { t = 0 } ^ { T - 1 }$ $\begin{array} { r l r } { \sum _ { i = 1 } ^ { N } \left( Q _ { i } ^ { l } ( t ) + Z _ { i } ^ { l } ( t ) + Q _ { i } ^ { s } ( t ) + Z _ { i } ^ { s } ( t ) \right) } & { { } \le } & { \frac { \hat { B } + \hat { C } } { \epsilon } - \frac { \hat { V } } { \epsilon } } \end{array}$ $\begin{array} { r } { ( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ P _ { \mathrm { s y s } } ( t ) ] - \frac { } { } \Phi ( \epsilon ) ) + \frac { \mathbb { E } [ \mathcal { L } ( \Phi ( 0 ) ] } { \epsilon T } } \end{array}$ . By taking the limit of both sides of the inequation as $T ~  ~ \infty$ and considering the fact $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ P _ { \mathrm { { s y s } } } ( t ) \right] \ \leq \ P _ { \mathrm { { s y s } } } ^ { * } } \end{array}$ , we obtain $\begin{array} { r l r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { i = 1 } ^ { N } \big ( \dot { Q _ { i } ^ { l } } ( { t } ) + \dot { Z _ { i } ^ { l } } ( { t } ) + \dot { Q _ { i } ^ { s } } ( t ) + \dot { Z _ { i } ^ { s } } ( t ) \big ) } & { \leq } & { \frac { 1 } { \epsilon } } \end{array}$ $\Bigl ( \hat { B } + C - V ( P _ { \mathrm { s y s } } ^ { * } - \Phi ( \epsilon ) ) \Bigr ) < \infty$ . The inequation indicates that all queues $Q _ { i } ^ { l } ( t ) , Z _ { i } ^ { l } ( \dot { t } ) , Q _ { i } ^ { s } ( t ) , Z _ { i } ^ { s } ( t )$ are strongly stable, which also implies mean rate stability (Theorem 2.8 of [28]). In addition, since the two virtual queues $Z _ { i } ^ { l } ( t ) , ~ Z _ { i } ^ { s } ( t )$ are mean-rate stable, the QoE constraints (3) and (4) are satisfied. This conludes the proof of statement (b).

## REFERENCES

[1] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, “A survey on mobile edge computing: The communication perspective,” IEEE Commun. Surveys Tuts., vol. 19, no. 4, pp. 2322–2358, 4th Quart., 2017.

[2] Q.-V. Pham et al., “A survey of multi-access edge computing in 5G and beyond: Fundamentals, technology integration, and state-of-the-art,” IEEE Access, vol. 8, pp. 116974–117017, 2020.

[3] Q.-V. Pham et al., “Aerial computing: A new computing paradigm, applications, and challenges,” IEEE Internet Things J., vol. 9, no. 11, pp. 8339–8363, 2022.

[4] F. Guo, H. Zhang, H. Ji, X. Li, and V. C. M. Leung, “An efficient computation offloading management scheme in the densely deployed small cell networks with mobile edge computing,” IEEE/ACM Trans. Netw., vol. 26, no. 6, pp. 2651–2664, Dec. 2018.

[5] Y. Dai, D. Xu, S. Maharjan, and Y. Zhang, “Joint computation offloading and user association in multi-task mobile edge computing,” IEEE Trans. Veh. Technol., vol. 67, no. 12, pp. 12313–12325, Dec. 2018.

[6] S. Josilo and G. Dan, “Computation offloading scheduling for periodic tasks in mobile edge computing,” IEEE/ACM Trans. Netw., vol. 28, no. 2, pp. 667–680, Apr. 2020.

[7] H. Tout, A. Mourad, N. Kara, and C. Talhi, “Multi-persona mobility: Joint cost-effective and resource-aware mobile-edge computation offloading,” IEEE/ACM Trans. Netw., vol. 29, no. 3, pp. 1408–1421, Jun. 2021.

[8] M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A tutorial on UAVs for wireless networks: Applications, challenges, and open problems,” IEEE Commun. Surveys Tuts., vol. 21, no. 3, pp. 2334–2360, 3rd Quart., 2019.

[9] B. Li, Z. Fei, and Y. Zhang, “UAV communications for 5G and beyond: Recent advances and future trends,” IEEE Internet Things J., vol. 6, no. 2, pp. 2241–2263, Apr. 2019.

[10] X. Pang, M. Sheng, N. Zhao, J. Tang, D. Niyato, and K.-K. Wong, “When UAV meets IRS: Expanding air-ground networks via passive reflection,” IEEE Wireless Commun., vol. 28, no. 5, pp. 164–170, Oct. 2021.

[11] W. Xu et al., “Throughput maximization of UAV networks,” IEEE/ACM Trans. Netw., vol. 30, no. 2, pp. 881–895, Apr. 2022.

[12] M. Agiwal, H. Kwon, S. Park, and H. Jin, “A survey on 4G-5G dual connectivity: Road to 5G implementation,” IEEE Access, vol. 9, pp. 16193–16210, 2021.

[13] Y. Li, Y. Wu, M. Dai, B. Lin, W. Jia, and X. Shen, “Hybrid NOMA-FDMA assisted dual computation offloading: A latency minimization approach,” IEEE Trans. Netw. Sci. Eng., vol. 9, no. 5, pp. 3345–3360, Sep. 2022.

[14] C. Li, H. Wang, and R. Song, “Intelligent offloading for NOMA-assisted MEC via dual connectivity,” IEEE Internet Things J., vol. 8, no. 4, pp. 2802–2813, Feb. 2020.

[15] H. Guo, J. Liu, J. Zhang, W. Sun, and N. Kato, “Mobile-edge computation offloading for ultradense IoT networks,” IEEE Internet Things J., vol. 5, no. 6, pp. 4977–4988, Dec. 2018.

[16] H. Guo, J. Liu, and J. Zhang, “Computation offloading for multi-access mobile edge computing in ultra-dense networks,” IEEE Commun. Mag., vol. 56, no. 8, pp. 14–19, Aug. 2018.

[17] Q.-V. Pham, S. Mirjalili, N. Kumar, M. Alazab, and W.-J. Hwang, “Whale optimization algorithm with applications to resource allocation in wireless networks,” IEEE Trans. Veh. Technol., vol. 69, no. 4, pp. 4285–4297, Apr. 2020.

[18] H.-G.-T. Pham, Q.-V. Pham, A. T. Pham, and C. T. Nguyen, “Joint task offloading and resource management in NOMA-based MEC systems: A swarm intelligence approach,” IEEE Access, vol. 8, pp. 190463–190474, 2020.

[19] T. Q. Dinh, J. Tang, Q. D. La, and T. Q. S. Quek, “Offloading in mobile edge computing: Task allocation and computational frequency scaling,” IEEE Trans. Commun., vol. 65, no. 8, pp. 3571–3584, Aug. 2017.

[20] T. X. Tran and D. Pompili, “Joint task offloading and resource allocation for multi-server mobile-edge computing networks,” IEEE Trans. Veh. Technol., vol. 68, no. 1, pp. 856–868, Jan. 2019.

[21] H. Hu, W. Song, Q. Wang, R. Q. Hu, and H. Zhu, “Energy efficiency and delay tradeoff in an MEC-enabled mobile IoT network,” IEEE Internet Things J., vol. 9, no. 17, pp. 15942–15956, Sep. 2022.

[22] P. Wei et al., “Reinforcement learning-empowered mobile edge computing for 6G edge intelligence,” IEEE Access, vol. 10, pp. 65156–65192, 2022.

[23] L. Huang, S. Bi, and Y.-J. A. Zhang, “Deep reinforcement learning for online computation offloading in wireless powered mobile-edge computing networks,” IEEE Trans. Mobile Comput., vol. 19, no. 11, pp. 2581–2593, 2020.

[24] Y.-C. Wu, T. Q. Dinh, Y. Fu, C. Lin, and T. Q. Quek, “A hybrid DQN and optimization approach for strategy and resource allocation in MEC networks,” IEEE Trans. Wireless Commun., vol. 20, no. 7, pp. 4282–4295, Jul. 2021.

[25] S. Bi, L. Huang, H. Wang, and Y.-J.-A. Zhang, “Lyapunov-guided deep reinforcement learning for stable online computation offloading in mobile-edge computing networks,” IEEE Trans. Wireless Commun., vol. 20, no. 11, pp. 7519–7537, Nov. 2021.

[26] X. Li, L. Huang, H. Wang, S. Bi, and Y.-J.-A. Zhang, “An integrated optimization-learning framework for online combinatorial computation offloading in MEC networks,” IEEE Wireless Commun., vol. 29, no. 1, pp. 170–177, Feb. 2022.

[27] Y. He et al., “Deep-reinforcement-learning-based optimization for cacheenabled opportunistic interference alignment wireless networks,” IEEE Trans. Veh. Technol., vol. 66, no. 11, pp. 10433–10445, Sep. 2017.

[28] M. J. Neely, Stochastic Network Optimization With Application to Communication and Queueing Systems. San Rafael, CA, USA: Morgan & Claypool, 2010.

[29] M. Min, L. Xiao, Y. Chen, P. Cheng, D. Wu, and W. Zhuang, “Learningbased computation offloading for IoT devices with energy harvesting,” IEEE Trans. Veh. Technol., vol. 68, no. 2, pp. 1930–1941, Feb. 2019.

[30] H. Liu and G. Cao, “Deep reinforcement learning-based server selection for mobile edge computing,” IEEE Trans. Veh. Technol., vol. 70, no. 12, pp. 13351–13363, Dec. 2021.

[31] K. Guo, R. Gao, W. Xia, and T. Q. S. Quek, “Online learning based computation offloading in MEC systems with communication and computation dynamics,” IEEE Trans. Commun., vol. 69, no. 2, pp. 1147–1162, Feb. 2021.

[32] L. T. Hoang, C. T. Nguyen, P. Li, and A. T. Pham, “Joint uplink and downlink resource allocation for UAV-enabled MEC networks under user mobility,” in Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops), May 2022, pp. 1059–1064.

[33] S. M. Ross, Introduction to Probability Models, 12th ed. New York, NY, USA: Academic, 2019.

[34] Y. Mao, J. Zhang, S. H. Song, and K. B. Letaief, “Stochastic joint radio and computational resource management for multi-user mobileedge computing systems,” IEEE Trans. Wireless Commun., vol. 16, no. 9, pp. 5994–6009, Sep. 2017.

[35] T. D. Burd and R. W. Brodersen, “Processor design for portable systems,” J. VLSI Signal Process. Syst. Signal, Image Video Technol., vol. 13, no. 2, pp. 203–221, Aug. 1996.

[36] J. Zhang et al., “Stochastic computation offloading and trajectory scheduling for UAV-assisted mobile edge computing,” IEEE Internet Things J., vol. 6, no. 2, pp. 3688–3699, Apr. 2019.

[37] K. Tammer, “The application of parametric optimization and imbedding to the foundation and realization of a generalized primal decomposition approach,” Math. Res., vol. 35, pp. 376–386, 1987. [Online]. Available: http://pascal-francis.inist.fr/ vibad/index.php?action=getRecordDetail&idt=7646720

[38] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, arXiv:1412.6980.

[39] L. Grippo and M. Sciandrone, “On the convergence of the block nonlinear Gauss–Seidel method under convex constraints,” Oper. Res. Lett., vol. 26, no. 3, pp. 127–136, 2000.

<!-- image-->

Linh T. Hoang (Graduate Student Member, IEEE) received the B.E. degree in electronics and telecommunications engineering from the Hanoi University of Science and Technology (HUST), Vietnam, in 2018, and the M.S. degree in computer science and engineering from The University of Aizu, Japan, in 2021, where he is currently pursuing the Ph.D. degree. His study in Japan is funded by the Japanese Government Scholarship (Monbukagakusho). His research interests include unmanned aerial vehicleaided networks, medium access control techniques,

resource management in mobile edge computing, and machine learning in wireless communications.

<!-- image-->

Chuyen T. Nguyen received the B.E. degree in electronics and telecommunications from the Hanoi University of Science and Technology (HUST), Vietnam, in 2006, the M.S. degree in communications engineering from National Tsing Hua University, Taiwan, in 2008, and the Ph.D. degree in informatics from Kyoto University, Japan, in 2013. From September 2014 to November 2014, he was a Visiting Researcher with The University of Aizu, Japan. He is currently an Associate Professor with the School of Electrical and Electronic Engineering (SEEE), HUST. His current research interests include MAC protocol design and reliable transmission in wireless/optical networks. He received the Fellow Award from Hitachi Global Foundation in August 2016, the First Best Paper Award from the 2019 IEEE ICT, and the Best Paper Award from the 2018 KICS/IEEE ICTC and the 2019 IEEE ICC.

<!-- image-->

Anh T. Pham (Senior Member, IEEE) received the B.E. and M.E. degrees in electronics engineering from the Hanoi University of Science and Technology (HUST), Vietnam, in 1997 and 2000, respectively, and the Ph.D. degree in information and mathematical sciences from Saitama University, Japan, in 2005. From 1998 to 2002, he was with NTT Corporation, Vietnam. Since 2005, he has been a Faculty Member with The University of Aizu, where he is currently a Professor and the Head of the Computer Communications Laboratory, Division

of Computer Engineering. He has authored/coauthored more than 200 peerreviewed articles on these topics. His research interests include communication theory and networking with a particular emphasis on modeling, design, and performance evaluation of wired/wireless communication systems and networks. He is a member of IEICE and OSA.