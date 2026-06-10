# Multiagent Deep Reinforcement Learning-Based Hierarchical Scheduling in Heterogeneous UAV-Enabled Vehicular Networks

Tianjiao Du , Xiaolin Gui , Member, IEEE, and Tao Sheng

Abstract—Current vehicular network systems often encounter significant challenges in dynamic task offloading and communication resource optimization, particularly in dynamic environments with varying computational demands. To address the challenges, we propose a novel two-tier unmanned aerial vehicle (UAV)- assisted architecture, where observation rotary-wing UAVs (OUs) are stationed above roadways to collect and transmit vehicle task information to relay rotary-wing UAVs (RUs) hovering over urban areas, which further offload the tasks to remote mobile edge computing (MEC) servers for execution and return the processed results. This approach enhances data transmission rates and optimizes network coverage, enabling seamless computational task offloading from vehicles to edge while reducing latency. To solve the multilayer vehicle task scheduling problem within heterogeneous intelligent UAV architectures, we first formulate an optimization problem aimed at minimizing latency and maximizing load balancing, which is further analyzed by a Markov decision process (MDP). Then, we propose a multiagent deep reinforcement learning-based hierarchical scheduling algorithm in heterogeneous UAV-enabled vehicular network (MAHHV) with the centralized training and decentralized execution (CTDE) framework to pursue the optimal hierarchical offloading policy. In addition, we integrate a gated recurrent unit (GRU) network to enhance the temporal dependency modeling and improve decision-making efficiency in dynamic environments. Numerical results demonstrate that the proposed algorithm outperforms the best baseline multiagent proximal policy optimization (MAPPO) by 0.6% on average across varying numbers of vehicles, validating its substantial potential for advancing intelligent vehicular systems and enhancing the efficacy of distributed computing in real-time applications.

Index Terms—Computational task offloading, mobile edge computing (MEC), multiagent deep reinforcement learning (MADRL), unmanned aerial vehicles (UAVs), vehicular network.

## NOMENCLATURE

Notation Definition T , t, and T Set, index, and number of time slots. V (t) v, and V(t) Set, index, and number of vehicles at t.

Received 1 August 2025; revised 26 September 2025; accepted 10 October 2025. Date of publication 15 October 2025; date of current version 8 December 2025. This work was supported in part by the Natural Science Basic Research Program of Shaanxi Province under Grant 2023-JC-ZD-38 and in part by the National Key Research and Development Program of China under Grant 2022YFB3305502 and Grant 2022YFB3305503. (Corresponding author: Xiaolin Gui.)

The authors are with the School of Electronic and Information Engineering and Shaanxi Province Key Laboratory of Computer Network, Xi’an Jiaotong University, Xi’an 710049, China (e-mail: tianjiaodu@stu.xjtu.edu.cn; xlgui@mail.xjtu.edu.cn; Shengtao@stu.xjtu.edu.cn).

Digital Object Identifier 10.1109/JIOT.2025.3621756

K k, and K Set, index, and number of observation rotary-wing tier unmanned aerial vehicles (OUs).

J  j, and J Set, index, and number of relay rotarywing unmanned aerial vehicles (RUs).

M m, and M Set, index, and number of macro eNodeBs (MeNBs).

$P _ { \nu } ( t )$ Coordinates of vehicle v at t.

$P _ { k }$ Coordinates of OU k.

$P _ { j }$ Coordinates of RU j.

$\pmb { P _ { m } }$ Coordinates of MeNB m.

$D _ { \nu } ( t )$ Task size of vehicle v.

$C _ { \nu } ( t )$ Task required computing resource of vehicle v.

$L _ { \nu } ( t )$ Task tolerable delay of vehicle v.

$d _ { \nu , k } ( t )$ Distance between vehicle v and the OU k.

$d _ { k , j }$ Distance between OU k and the RU j.

$d _ { j , m }$ Distance between RU j and MeNB m.

$B _ { \nu , k } ( t )$ Bandwidth allocated by OU k to vehicle v.

$B _ { k , j } ( t )$ Bandwidth allocated by OU k to RU j. $B _ { j , m } ( t )$ Bandwidth allocated by RU j to MeNB m.

$T _ { i \cdot \cdot } ^ { \mathrm { r e c } }$ Receiving policy of RU j for vehicle v. $T _ { j , \nu } ^ { \mathrm { r t r } }$ Retransmission policy of RU j for vehicle v.

$N _ { i } ^ { \mathrm { r e c } } ( t )$ Amount of tasks received by RU j until t. $N _ { m } ^ { \mathrm { r t r } } ( t )$ Amount of tasks retransmitted to MeNB m until t.

## I. INTRODUCTION

HE rapid development of digital technologies has given T rise to the Internet of Things (IoT), which has facilitated the emergence of smart cities and provided a platform for complex applications, significantly enhancing the quality of life for residents [1], [2]. Various IoT-based technologies collaborate to drive innovation and optimization in transportation systems through real-time data collection and intelligent decision-making, positioning the Internet of Vehicles (IoV) as a competitive paradigm in the future sixth generation (6G) era [3], [4]. The high bandwidth, low latency, and extensive connectivity offered by 6G technology will enable future intelligent vehicles to possess more comprehensive and diverse functionalities, such as advanced communication capabilities.

Moreover, massive sensor data can provide drivers with realtime traffic conditions, video analysis, parking information, and other services [5]. Simultaneously, the development of IoV has led to the proliferation of various computationally intensive applications, which rely on high-performance computing and real-time data processing [6]. Due to the limited computational resources, vehicles are unable to handle compute-intensive or latency-sensitive tasks, resulting in poor user experience and satisfaction.

In response to the above challenge, mobile edge computing (MEC) can provide support for latency, reliability, bandwidth efficiency, and servicing real-time applications by positioning computational and data storage [7]. vehicular edge computing (VEC) further extends the capabilities of MEC by integrating computing resources directly into vehicular scenarios [8]. This allows for real-time processing of vehicle-generated data, enabling applications such as autonomous driving, traffic management, and in-vehicle infotainment systems. Nevertheless, traditional ground-based VEC communication infrastructure may exhibit critical weaknesses in coverage, flexibility, and reliability, particularly in emergencies. Natural disasters can lead to the degradation or complete failure of cellular networks, making it imperative to establish alternative communication pathways [9]. The incorporation of unmanned aerial vehicles (UAVs) into VEC frameworks can mitigate many challenges associated with disaster management, as they can provide line-of-sight (LoS) links to reduce interference, enhance connectivity, and extend coverage, improving accessibility in disaster-affected areas [10]. Furthermore, the flexible deployment of UAVs enables dynamic network topologies to adapt to changing environmental conditions and allows UAVs to serve as temporary relays to deliver tasks for remote computing, thereby alleviating computational load and enhancing vehicular communications in areas experiencing network congestion or surges in computational requirements [11]. Consider the constraints on resources and service capabilities, a single UAV is no longer sufficient to meet the actual mission requirements. In particular, multi-UAV systems have been widely studied for their ability to extend network coverage [12], ensure seamless connectivity [13] and enhance reliability through collaborative resource allocation and dynamic load balancing [14]. However, they also pose several challenges [15], [16]. Practically, the high variability of vehicle operational information, the dynamically evolving traffic environment, the intricate demand of load balancing, and the optimization complexity cannot be ignored. They pose significant difficulties to the real-time relay decision-making, making it essential to design an adaptive and efficient relay algorithm UAV-assisted IoV system.

In previous studies [11], [17], [18], the successive convex approximation algorithm is often used to address the multiconstraint task offloading problem in UAV-assisted MEC scenarios. However, most of these methods address nonconvex optimization problems by decoupling the coupled variables and decomposing the original problem into several subproblems for solution, which generally results in typically exhibit limitations in adaptability and long-term optimization [1]. Deep reinforcement learning (DRL), with its unique advantages in dynamic adaptation, cumulative reward optimization, and real-time decision-making, offers a promising alternative that has the potential to significantly enhance the efficiency and effectiveness of UAV operations within the IoV network [19], [20], [21]. In this article, we propose a multilayer vehicle task scheduling architecture that utilizes heterogeneous UAVs as data bridges. It introduces observation UAVs to collect task information and relay UAVs to extend network coverage while acting as multidecision agents for collaborative task offloading. A multiagent DRL algorithm is employed to enhance the overall system efficiency.

## A. Related Works

The emergence of 6G technology promises to enhance the capabilities and performance of IoV, enabling more advanced vehicular communication systems and applications [22]. However, due to the limited storage and computing resources of vehicles, MEC is expected to enable data processing at roadside units (RSUs), thereby enhancing the overall computational capability and responsiveness of IoV networks [6]. Consequently, the computational offloading problem in vehicular edge networks has attracted significant interest. Wang et al. [23] constructed an MEC-assisted vehicle-toeverything network and proposed an effective offloading scheme based on game theory. Zhao et al. [24] proposed a multihop communication task offloading strategy in RSU-assisted vehicular networks, achieving improved task completion ratio and reduced processing delay. Furthermore, it is imperative to meet the high quality of service (QoS) and quality of experience (QoE) requirements of applications. While inappropriate task offloading may temporarily fulfill immediate requirements, it has the potential to overwhelm limited edge infrastructure resources over time [25]. This can lead to higher overall service latency and jitter, resulting in delays and diminished responsiveness.

However, under conditions of bursty traffic, such as congestion in highways or parking lots, there is a significant surge in demand for vehicular network applications. In addition, in disaster scenarios, the real-time information about on-site conditions must be promptly relayed to remote command centers for appropriate guidance. The resource capacity of edge servers may reach its upper bound [7], and the concentrated transmission can lead to network congestion. In such scenarios, technologies like LoRa, while suitable for low-data-rate and long-range communication, are inadequate due to their inability to handle high data throughput and low-latency requirements. Furthermore, in disaster scenarios, where RSUs along roadways may be damaged, real-time environmental information must be promptly relayed to remote command centers for appropriate guidance, necessitating a robust and high-capacity communication framework that can ensure reliable and timely data transmission even under extreme conditions.

UAVs circumvent ground-based challenges through their superior maneuverability and deployability, delivering multiangle aerial interfaces for the IoV system within the 3-D architecture [3], [16]. By offering real-time aerial data collection, they facilitate network expansion and dynamic routing, emerging as a compelling solution when traditional MEC systems face limitations. Recently, UAVs have been studied as relays to help ground users establish communications with MeNBs whenever necessary, which is particularly critical in urban environments with intensive task computing requirements or during the aftermath of natural disasters. In [26], UAV relays can significantly improve the performance of wireless communications. Zhang et al. [11] and Li et al. [27] have proposed solutions for vehicular networks aimed at minimizing energy consumption while providing communication and task-offloading services. It is necessary to acknowledge the limitations of the aforementioned studies, which focus exclusively on the applications of a single UAV.

TABLE I  
COMPARISON WITH RELATED WORKS
<table><tr><td>Ref.</td><td>UAV-assisted network</td><td>Heterogeneous Service coverage UAVs</td><td>overlapping</td><td>Offloading scheduling</td><td>Latency minimization</td><td>Load balancing</td><td>Algorithm architecture</td><td>MADRL solution</td></tr><tr><td>[1]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[3]</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>CTDE</td><td>√</td></tr><tr><td>[11]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[15]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>distributed</td><td>√</td></tr><tr><td>[16]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>CTDE</td><td>√</td></tr><tr><td>[17]</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[23]</td><td>×</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>distributed</td><td>×</td></tr><tr><td>[25]</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>CTDE</td><td>√</td></tr><tr><td>[7]</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[26]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[27]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>×</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[28]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>×</td><td>×</td><td>distributed</td><td>×</td></tr><tr><td>[29]</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[30]</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>centralized</td><td>×</td></tr><tr><td>[31]</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>distributed</td><td>√</td></tr><tr><td>[32]</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>distributed</td><td>√</td></tr><tr><td>[33]</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>CTDE</td><td>√</td></tr><tr><td>Our work</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>CTDE</td><td>√</td></tr></table>

To address these challenges, deploying multiple UAVs can improve communication efficiency and system reliability in vehicular networks through collaborative optimization, load balancing, and expanded coverage. Oubbati et al. [28] explored routing strategies in UAV-assisted vehicular ad hoc networks (VANETs) and introduced a predictive technique for estimating the expiration time of communication paths. The expression for peak age-of-information was derived, and the nonconvex optimization problem was solved by considering associated soft constraints [29]. Recently, the deployment of multiple UAVs as relays has greatly advanced IoV technologies by enhancing the network coverage, real-time data transmission, dynamic routing optimization, edge computing, and security support. A UAV-assisted data aggregation mechanism has been proposed, utilizing the average data to overhead ratio as a metric to enable vehicular user equipment to efficiently relay data to base stations in space–air–ground integrated networks [30]. While high-performance antennas can extend UAV transmission range, their size and weight exceed typical payload limits, making them impractical without degrading flight performance. Thus, single-layer UAVs performing both observation and transmission still face restricted communication coverage. A heterogeneous multilayer UAV architecture has been proposed in [17], where observation and relay UAVs cooperate to overcome the constraints of single-layer systems and improve task execution efficiency and communication coverage.

Reinforcement learning is widely used to solve nonconvex optimization problems in UAV-assisted IoV due to its strong adaptability to dynamic environments [1]. In multivehicle task offloading scenarios, traditional single-agent reinforcement learning architectures face challenges, such as single-point failure risks, sharply increasing computational and communication loads with vehicle numbers, and high latency and uncertainty in obtaining global information, restricting their applicability in dynamic vehicular networks [15]. To overcome these limitations, researchers have proposed multiagent reinforcement learning algorithms to collaboratively optimize and enhance system performance [31]. [32] investigated the routing problem in a multihop UAV relay network and proposed a multiagent DRL (MADRL)-based algorithm with intra-UAV and inter-UAV training mechanisms to solve problems with large action spaces. This approach helps to enhance network throughput, reduce congestion probability, and shorten transmission time. To highlight the advantages of our work, a comprehensive comparison between our work and existing related research is summarized in Table I.

However, although numerous studies have focused on UAV-assisted vehicular networks, existing approaches have generally overlooked the systematic modeling and optimization of heterogeneous UAVs functioning as multitier relays. Moreover, most existing methods still rely on centralized decision-making mechanisms, which inherently restrict the flexibility and scalability of the system. While some studies have explored multiagent decision-making, they typically adopt a decentralizing training and decentralizing execution (DTDE) paradigm, where each agent only uses its own local observations for training and decision-making. Although DTDE can effectively address scalability issues caused by the increasing number of agents, it is theoretically constrained by nonstationarity and sensitivity to partial observability [34]. Furthermore, centralized training with decentralized execution (CTDE)-based studies leverage a centralized critic to process global states during training but generally ignore overlapping coverage, making them unsuitable for real-world service-overlapping scenarios. Although some studies, such as [1], [25], and [33] have considered complete task offloading in overlapping regions. In [25], vehicles were modeled as agents for decision-making, but they were mostly limited to perceiving local states, which constrained the availability of real-time decision information. Other deep deterministic policy gradient (DDPG)-based approaches rely on continuous action spaces, where task assignment is determined by comparing the magnitudes of network outputs [33]. However, such approaches suffer from output fluctuations that result in slow policy convergence and poor stability. Moreover, the continuous outputs are difficult to map to discrete offloading requirements, thereby limiting performance in multi-UAV cooperation or highly concurrent scenarios. To the best of the authors’ knowledge, we are the first to propose an innovative scheme that leverages multiple heterogeneous UAVs as relays to construct a two-layer, highly interconnected architecture for ultralong-distance task migration. Within this architecture, an MADRL framework is introduced, combined with an integer-based decision mechanism and CTDE paradigm, to enable the cooperative offloading decisions and global environment awareness in UAV swarms over service-overlapping regions. The proposed approach effectively enhances task scheduling performance, reduces latency, and ensures load balancing, making it particularly suitable for ultralong-distance vehicular task offloading scenarios.

## B. Motivation and Contributions

Motivated by the above-mentioned research and challenges, this study aims to address the growing demand for efficient task offloading in complex vehicular networks, particularly in urban environments where communication range and resource constraints limit the capabilities of traditional single-agent reinforcement learning models. Existing approaches often struggle to adapt to large scale, dynamic environments, resulting in inefficiencies. To overcome these challenges, this study leverages heterogeneous UAVs as relay nodes in a dual-layer, highly interconnected architecture for task transfer and offloading. By employing MADRL, cooperative decision-making among UAVs is achieved through a centralized training and decentralized execution (CTDE) mechanism, enabling agents to gain awareness of global environmental information. The objective is to improve the task scheduling performance, reduce latency, and ensure load balancing, particularly in ultralong-distance vehicular network scenarios.

1) This study introduces a novel dual-layer, highly interconnected architecture that leverages heterogeneous UAVs as task relays to dynamically collect and deliver vehicle tasks. By utilizing UAVs with distinct roles, the system enhances coverage, communication reliability, and load balancing across multiple regions, enabling efficient offloading and computation.

2) A multilayer scheduling framework is proposed, where UAVs operate at distinct layers to facilitate efficient task assignment and coordination across the network. The framework integrates multiagent collaborative decisionmaking, enabling the multistage selective task transfer and computational offloading for real-time tasks.

3) The task scheduling problem is modeled as a partially observable Markov decision process (POMDP), and a multiagent reinforcement learning approach is utilized to enable UAVs to interact with the environment and collaboratively optimize task offloading decisions. This formulation improves decision-making in dynamic, high-dimensional environments.

4) This article presents a multiagent DRL-based hierarchical scheduling algorithm in heterogeneous UAV-enabled vehicular network (MAHHV), which builds on the state-of-the-art multiagent proximal policy optimization (MAPPO). The algorithm adopts a CTDE framework, allowing UAVs to learn cooperative policies during training while making autonomous decisions during execution, which enables the system to scale and adapt to dynamic IoV environments effectively.

The rest of this article is summarized as follows. In Section II, we present the system model and the problem formulation. In Section III, we introduce the proposed MADRL-based hierarchical scheduling. The numerical results and comparative analysis are exhibited in Section IV while Section V concludes this article. For ease of reference, we provide a list of important notations in the Nomenclature.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

In this section, we illustrate a heterogeneous UAV-enabled vehicular network and then propose a hierarchical scheduling model. To this end, we formulate the vehicle tasks scheduling optimization problem.

## A. Network Model

Herein, we consider a 5G-/6G-driven vehicular network scenario in the urban environment, where ground vehicles upload tasks through OUs and RUs to the MeNBs mounted with edge servers. As illustrated in Fig. 1, the proposed model includes two types of UAVs that perform their respective missions. Specifically, OUs are stationed above roadways and responsible for collecting and transmitting the location and task information of vehicles within their coverage area. However, due to the limitations of direct communication over long distances and potential obstacles in urban environments, the RUs, which hover above urban areas, are introduced for retransmitting the tasks from OUs to MeNBs, where edge servers perform task calculations and return the final results. This multihop communication strategy avoids the LoS occlusion problem of single-layer UAV relaying and ensures reliable and efficient data transmission, even in a complex urban environment. To timely evaluate the status of the network, a digital twin (DT) layer is deployed at the emergency control center, which is utilized not only to model the entire network but also to support model training and parameter synchronization for learning-based methods [27], [35], [36]. Regarding the energy consumption concerns of UAVs and to extend their flight durations, the MeNBs are considered to serve as charging stations as suggested in [37]. Each UAV takes off with a maximum battery capacity and returns to the charging station before its energy drops below a minimum threshold to ensure continuous operation. In addition, laser-based wireless charging technologies have also garnered significant attention and exploration, as discussed in [38] and [39], to provide UAVs with rapid and targeted energy replenishment, enabling them to maintain continuous hover.

<!-- image-->  
Fig. 1. Heterogeneous UAV-enabled vehicular networks with hierarchical scheduling.

Assuming that the areas covered by UAVs correspond to regions with high-vehicle task densities, the group of vehicles within the coverage of OUs is denoted by $\mathcal { V } ( t ) \ =$ $\{ 1 , \ldots , \nu , \ldots , V ( t ) \}$ . The 3-D coordinates of vehicle v are represented as $P _ { \nu } ( t ) = ( x _ { \nu } ( t ) , y _ { \nu } ( t ) , z _ { \nu } ( t ) )$ . Each vehicle v generates a computational task, denoted by $\{ D _ { \nu } ( t ) , C _ { \nu } ( t ) , L _ { \nu } ( t ) \}$ , where $D _ { \nu } , \ C _ { \nu } ,$ and $L _ { \nu }$ ,are the task size, the needed CPU cycles for computing the task, and the tolerable delay of the task including emergency rescue path planning, vehicle status monitoring, and early warning. The OUs, denoted by $\mathcal { K } = \{ 1 , \ldots , k , \ldots , K \}$ , are fixed above the road area and have , . . . , , . . . ,the coordinates represented by $P _ { k } = ( x _ { k } , y _ { k } , z _ { k } )$ . After collect-, ,ing the location and task information within their coverage, the OUs forward this information to their associated RUs. The total set of the RUs is defined as $\mathcal { T } = \{ 1 , \dots , j , \dots , J \}$ , which consists of the set of intraregion RUs (I-RUs) responsible for relaying tasks in nonoverlapping OU observation areas and the set of cross-region RUs (C-RUs) with additional transmission capabilities for tasks in overlapping areas. Since RUs perform distributed decision-making for task relaying, this classification effectively reduces the decision-making burden on individual RUs. Given this structure, an OU can be connected to multiple RUs. The set and number of RUs connected to OU k are denoted by $\mathcal { T } _ { k } / J _ { k }$ , and the OU associated with RU j is denoted by $k _ { j } .$ / Furthermore, the RUs act as the transmission bridges between OUs and MeNBs, enabling realtime ultralong-distance transfer. Due to the distance to MeNB and the computational capabilities of the MEC server mounted on it affect the decisions of RUs, as well as the computational delay of tasks to be executed at the MeNBs, we represent the set of MeNBs as $\mathcal { M } \ = \ \{ 1 , \dotsc , m , \dotsc , M \}$ , with their , .computational resources denoted by $F _ { m } .$

In the typical network, we divide the task assignment decision problem into a two-stage task transfer mechanism and propose a hierarchical scheduling method to cooperatively accomplish the task using a distributed optimization framework. In the first stage, vehicles on urban roads upload their coordinates and generate task information to the OUs hovering above them. We assume that each OU individually collects task information from vehicles within its designated observation area during each time slot. The collected information is then forwarded to the corresponding RUs, which are responsible for determining task reception strategies and instructing OUs on directional relaying for each specific vehicle task.

Given the overlap in the observation areas of OUs, we assume that tasks observed by multiple OUs are forwarded to C-RUs for collaborative offloading decisions. Let Vk(t) and $V _ { k } ( t )$ be the set and the number of vehicles within the coverage area of the kth OU, respectively. The set of vehicle tasks that require decision-making by RU j is denoted by $\mathcal { V } _ { j } ( t ) \subseteq \mathcal { V } _ { k _ { j } } ( t )$ with $V _ { j } ( t )$ being the corresponding number of tasks. After receiving the task information of vehicle $\nu \in \mathcal { V } _ { j } ( t )$ from its corresponding OU, each RU j determines the task reception policy, denoted by $T _ { j , \nu } ^ { \mathrm { r e c } } ( t )$ . Note that when $T _ { j , \nu } ^ { \mathrm { r e c } } = 1$ , it indicates , ,that RU j chooses to receive a task from vehicle v at time t; otherwise, it is equal to 0. For $\nu \in \mathcal { V } ( t ) \setminus \mathcal { V } _ { j } ( t )$ that are not received by RU j, we set $T _ { j , \nu } ^ { \mathrm { r e c } } ( t ) = 0$ . Since each task can be ,only offloaded to one of the RUs, the sum of $T _ { j , \nu } ^ { \mathrm { r e c } } ( t )$ over all RUs in J of each task v has to be equal to 1.

<!-- image-->  
Fig. 2. Total delay of task transmission and calculation process.

Furthermore, in the second stage, the RUs perform secondary retransmission. Due to limited communication distances, OUs cannot directly forward the collected tasks to distant MeNBs in real time. The re-transmission policy determined by RU j is $T _ { j , \nu } ^ { \mathrm { r t r } } ( t )$ , which represents the index of the ,MeNB chosen to execute the task computation for the vjth vehicle. After finishing calculating the task computation, the MeNB transmits the result back to the vehicle through the task uplink path. Since the downlink result size is small, we hereby omit the delay in result feedback [19]. The total latency for tasks consists of transmission latency and computational latency, as shown in Fig. 2. To meet the low-latency requirements of vehicle delay-sensitive tasks, the objective of our method is to minimize the total latency and ensure the network load balance by enabling OUs and RUs to make intelligent transmission decisions based on the real-time vehicle task information.

## B. Communication Model

To meet the requirements of 6G wireless communication systems for higher spectral efficiency, higher transmission performance, and lower latency, we consider a communication framework, where vehicle-OU and OU-RU communication connections are operating on the mmWave band, capitalizing on the high data rates and low latency enabled by LoS propagation [40]. We assume that a vehicle can establish a connection link with only one OU at each time slot, and each OU is responsible for collecting multiple vehicle tasks and establishing connections with multiple RUs for transmission. After receiving the task information, each RU offloads the tasks to its associated MeNBs. Given the potential for signal blockages in urban environments caused by obstacles such as buildings, trees, and other structures, the communication link between RU and MeNB is established based on the sub-6-GHz band [2]. The MeNB then calculates multiple tasks simultaneously and returns the results. In our network model, we adopt the orthogonal frequency-division multiple access (OFDMA) to manage interference [11], [27]. Interference is ignored due to the exclusive subcarrier allocation, since the existing techniques, such as cell planning, frequency reuse, and beam-forming, have proven effective in significantly mitigating the interference [41], [42]. In the proposed hierarchical scheduling mechanism, the first stage of task transmission includes wireless communication links from vehicles to OUs and from OUs to RUs. Since the wireless signal first propagates through the low-altitude urban environment and then transitions to free space during the task offloading stage, we adopt the probabilistic channel model described in [43].

To establish high-quality communication links, we assume that OUs have high altitudes. Thus, here we only consider the establishment of the LoS links between OUs and vehicles [1]. Let $\mathcal { V } _ { k } ( t ) / V _ { k } ( t )$ be the set/number of vehicles under the /coverage of OU k, and the pathloss from vehicle v to OU k can be written as follows:

$$
\mathrm { P L } _ { \nu , k } \left( t \right) = 2 0 \log { \left( \frac { 4 \pi f _ { c } d _ { \nu , k } \left( t \right) } { c } \right) } + \eta _ { \mathrm { L o S } }\tag{1}
$$

where $d _ { \nu , k } ( t ) = ( ( x _ { \nu } ( t ) - x _ { k } ) ^ { 2 } + ( y _ { \nu } ( t ) - y _ { k } ) ^ { 2 } + ( z _ { \nu } ( t ) - z _ { k } ) ^ { 2 } ) ^ { 1 / 2 }$ ,is the Euclidean distance between the vth vehicle and the kth OU at time slot t and $\eta _ { \mathrm { L o S } }$ is the excessive pathloss ηdue to the LoS connection. Then, the signal-to-noise ratio (SNR) [32] at UAV k from vehicle $\nu _ { j }$ can be expressed as $\Gamma _ { \nu , k } ~ = ~ ( P _ { \mathrm { v h c } } 1 0 ^ { - \mathrm { P L } _ { \nu , k } / 1 0 } ) / ( N _ { 0 } B _ { \nu , k } ( t ) )$ , where $P _ { \mathrm { v h c } }$ is the com-, / ,munication power of vehicles and $N _ { 0 }$ is the noise power spectral density. Let $\begin{array} { r } { N _ { j } ^ { \mathrm { r e c } } ( t ) ~ = ~ \sum _ { \nu = 1 } ^ { V ( t ) } T _ { j , \nu } ^ { \mathrm { r e c } } } \end{array}$ be the amount of tasks received by RU j. We define $B _ { \nu , k } ( \bar { t } ) = B _ { \mathrm { O U } } ^ { \mathrm { r e c } } / \sum _ { j = 1 } ^ { J _ { k } } N _ { j } ^ { \mathrm { r e c } } ( t )$ , /as the bandwidth allocated by OU k to serve each received vehicle. $B _ { \mathrm { O U } } ^ { \mathrm { r e c } }$ is defined as the total bandwidth of each OU to collect the vehicle tasks. The achievable transmission uplink rate from the vth vehicle to the kth OU can be denoted by $R _ { \nu , k } ( t ) \ = \ B _ { \nu , k } ( t ) \log _ { 2 } ( 1 + \Gamma _ { \nu , k } )$ . Thus, the receiving latency , , ,between OU k and the associated vehicle v can be calculated as follows:

$$
L _ { \nu , k } ^ { \mathrm { r e c } } \left( t \right) = \frac { D _ { \nu } \left( t \right) } { R _ { \nu , k } \left( t \right) } .\tag{2}
$$

Due to the long distance from the MeNB with computing resources, after the OU receives the tasks from the vehicles, it needs to establish links with the RUs associated with it for task transmission. Herein, we consider that the hovering height of the UAVs is higher than the obstacles in the urban environment, and the links between the UAVs are LoS in free space [44]. Similar to (1), the path loss from OU k to RU $j \in \mathcal { T } _ { k } \mathrm { \ i s \ P L } _ { k , j } = 2 0 \log ( 4 \pi f _ { c } d _ { k , j } / c ) + \eta _ { \mathrm { L o S } }$ , where $d _ { k , j } ~ =$ $( ( x _ { k } - x _ { j } ) ^ { 2 } + ( y _ { k } - y _ { j } ) ^ { 2 } + ( z _ { k } - z _ { j } ) ^ { 2 } ) ^ { 1 / 2 }$ , / η ,is the distance between the kth OU and the jth RU. $\eta _ { \mathrm { L o S } }$ is the additional attenuation factor ηdue to the LoS connection. The SNR at RU j from OU k can be calculated as $\Gamma _ { k , j } ( t ) = ( P _ { \mathrm { O U } } 1 0 ^ { - \mathrm { P L } _ { k , j } / 1 0 } ) / ( N _ { 0 } B _ { k , j } ( t ) )$ , where $P _ { \mathrm { O U } }$ ,is the transmission power of OUs and $B _ { k , j } ( t )$ , is the bandwidth allocated to transmit each task from OU k to RU j at time slot t, which can be calculated by $B _ { k , j } ( t ) = B _ { \mathrm { O U } } ^ { \mathrm { t r } } / \textstyle \sum _ { j = 1 } ^ { J _ { k } } N _ { j } ^ { \mathrm { r e c } } ( t ) . B _ { \mathrm { O U } } ^ { \mathrm { t r } }$ , /is denoted as the total bandwidth of each OU to transmit the tasks. The transmission rate between OU k and RU j can be written as $R _ { k , j } ( t ) = B _ { k , j } ( t ) \log _ { 2 } ( 1 + \Gamma _ { k , j } ( t ) )$ , and the transmission , , ,latency of the vth vehicle’s task between OU k and RU j can be denoted as follows:

$$
L _ { \nu , k , j } ^ { \mathrm { t r } } ( t ) = \frac { D _ { \nu } \left( t \right) } { R _ { k , j } \left( t \right) } .\tag{3}
$$

In this scenario, we assume that MeNBs are uniformly distributed on the ground in the city, and vehicles and OUs cannot establish direct communication links with MeNBs due to the long distance between them. Therefore, in the second stage of the hierarchical scheduling mechanism, the RUs allocate MeNBs to execute the computational tasks of vehicles and begin to retransmit the tasks. In urban environments, signal blockages caused by buildings, trees, and other obstacles must be considered. Hence, we adopt the widely used probabilistic channel model, which accounts for both LoS and non-LoS (NLoS) [43]. Let $\mathbf { \mathcal { M } } _ { j }$ be the set of MeNBs associated with the jth RU. The probability of the LoS connection between the RU j and MeNB $m \in { \mathcal { M } } _ { j }$ is expressed as $\mathrm { P r } _ { j , m } ^ { \mathrm { L o S } } = 1 / ( 1 + a \exp ( - b [ \theta _ { j , m } - a ] ) )$ , where a and ,b are constants related to the communication environment. $\theta _ { j , m }$ is the elevation angle calculated as $\theta _ { j , m } = \arcsin ( | z _ { j } - z _ { m } | / d _ { j , m } ) .$ and $d _ { j , m } = ( ( x _ { j } - \bar { x _ { m } } ) ^ { 2 } + ( y _ { j } - y _ { m } ) ^ { 2 } + ( \bar { z _ { j } } - z _ { m } ) ^ { 2 } ) ^ { 1 / 2 }$ represents the ,Euclidean distance between RU r and MeNB m. Let $\eta _ { \mathrm { N L o S } }$ be ηthe excessive path loss for NLoS connection, and the average path loss can be calculated as follows:

$$
\mathrm { P L } _ { j , m } = 2 0 \log \left( \frac { 4 \pi f _ { c } d _ { j , m } } { c } \right) + \mathrm { P r } _ { j , m } ^ { \mathrm { L o S } } \eta _ { \mathrm { L o S } } + \mathrm { P r } _ { j , m } ^ { \mathrm { N L o S } } \eta _ { \mathrm { N L o S } }\tag{4}
$$

where $\mathrm { P r } _ { j , m } ^ { \mathrm { N L o S } }$ is the NLoS probability $\mathrm { P r } _ { j , m } ^ { \mathrm { N L o S } } \ = \ 1 - \mathrm { P r } _ { j , m } ^ { \mathrm { N L o S } }$ and $\eta _ { \mathrm { N L o S } }$ , ,is the excessive pathloss due to the NLoS conηnections. Thus, the SNR between RU j and MeNB m can be expressed as $\Gamma _ { j , m } ( t ) = ( P _ { \mathrm { R U } } 1 0 ^ { - \mathrm { P L } _ { j , m _ { j } } / 1 0 } ) / ( N _ { 0 } B _ { j , m } ( t ) )$ , where , / ,PRU is the retransmission power of RU j to offload each task and $B _ { j , m } ( t ) = { B _ { \mathrm { O U } } ^ { \mathrm { r e c } } } / { N _ { j } ^ { \mathrm { r e c } } ( t ) }$ is the bandwidth evenly allocated , /by RU j to each task. The retransmission rate from RU j to MeNB m is $R _ { j , m } ( t ) = B _ { j , m } ( t ) \log _ { 2 } ( 1 + \Gamma _ { j , m } ( t ) )$ . Following this, the RUs complete the second stage of task re-transmission, and the latency of vehicle task v from jth RU to mth MeNB is

$$
L _ { \nu , j , m } ^ { \mathrm { r t r } } \left( t \right) = \frac { D _ { \nu } \left( t \right) } { R _ { j , m } \left( t \right) } .\tag{5}
$$

## C. Computational Model

We assume that each MEC server-amounted MeNB $m ,$ evenly distributes computing resources for the assigned tasks. Herein, we construct the retransmission policy as a one-hot vector $\vec { T } _ { j , \nu } ^ { \mathrm { r t r } } ( t )$ with dimension of M, where the element with ,index equal to $T _ { j , \nu } ^ { \mathrm { r t r } }$ is set to 1, while all other elements are set ,to 0. The total amount of tasks retransmitted to MeNB m at time slot t can be calculated by

$$
N _ { m } ^ { \mathrm { r t r } } \left( t \right) = \sum _ { \nu = 1 } ^ { V ( t ) } \sum _ { j = 1 } ^ { J } \vec { T } _ { j , \nu } ^ { \mathrm { r t r } } \left( t \right) \left[ m - 1 \right] .\tag{6}
$$

Accordingly, the computing resources obtained by the task of each vehicle v from MeNB m is $f _ { \nu , m } ( t ) ~ = ~ F _ { m } / N _ { m } ^ { \mathrm { r t r } } ( t )$

Immediately, the computational latency of the vth vehicle task can be written as follows:

$$
L _ { \nu , m } ^ { \mathrm { c o m } } \left( t \right) = \frac { C _ { \nu } } { f _ { \nu , m } } .\tag{7}
$$

## D. QoS and QoE Collaborative Model

With the advent of autonomous vehicles, along with the increasing demands for safety, efficiency, and satisfaction from users, optimizing vehicular networks has become critical. UAVs have emerged as a key technology to enhance nextgeneration intelligent transportation systems, offering dynamic and flexible solutions to address the remote connection challenges in complex urban environments. In this context, we focus on optimizing vehicular network resources by simultaneously considering QoE and QoS. This integrated approach not only improves user satisfaction and system performance but also ensures the efficient use of limited network resources, thereby promoting reliable and high-quality vehicle services. QoE is a user-centric metric that reflects the level of user satisfaction with the final service, based on individual pReferences [10]. For task offloading in vehicular networks, QoE of vehicles is designed to be related to the total delay of the task [45]. In the hierarchical scheduling method, after OU k finishes receiving and transmitting the vth vehicle task in the first stage, the time consumption for the task transmission of vehicle $\nu \in \mathcal { V } _ { k } ( t )$ during the first stage can be calculated as follows:

$$
L _ { \nu , k , j } \left( t \right) = L _ { \nu , k } ^ { \mathrm { r e c } } \left( t \right) + L _ { \nu , k , j } ^ { \mathrm { t r } } \left( t \right) .\tag{8}
$$

In the second stage, the delay associated with the task of vehicle v, which is retransmitted by RU j and executed by MeNB $T _ { j , \nu } ^ { \mathrm { r t r } }$ , can be written as follows:

$$
L _ { \nu , j , m } \left( t \right) = L _ { \nu , j , m } ^ { \mathrm { r t r } } \left( t \right) + L _ { \nu , m } ^ { \mathrm { c o m } } \left( t \right) .\tag{9}
$$

Considering that the receiving decision given by RUs will affect the task collection behavior of OUs, the binary parameter $T _ { k . \nu } ^ { \mathrm { r e c } } ( t ) = 1$ is defined when $\begin{array} { r } { \sum _ { j } ^ { J _ { k } } T _ { j , \nu } ^ { \mathrm { r e c } } = 1 } \end{array}$ , which indicates , ,that OU k receives the complete task information of vehicle v according to the policy of its corresponding RUs. Thus, the task transmission indicator can be defined as follows:

$$
{ { T } _ { \nu , k , j , m } } \left( t \right) = { { T } _ { k , \nu } ^ { \mathrm { { r e c } } } } \left( t \right) { { T } _ { j , \nu } ^ { \mathrm { { r e c } } } } \left( t \right) \vec { { T } } _ { j , \nu } ^ { \mathrm { { r t r } } } \left( t \right) \left[ m - 1 \right]\tag{10}
$$

and the possible earliness for the task of vehicle v is

$$
\widetilde { L } _ { \nu , k , j , m } \left( t \right) = L _ { \nu } \left( t \right) - L _ { \nu , k , j } \left( t \right) - L _ { \nu , j , m } \left( t \right) .\tag{11}
$$

To sum up, the total earliness can be given by

$$
L _ { \nu } ^ { \mathrm { t o t a l } } \left( t \right) = \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { J } \sum _ { m = 1 } ^ { M } T _ { \nu , k , j , m } \left( t \right) \widetilde { L } _ { \nu , k , j , m } \left( t \right) .\tag{12}
$$

Since numerous devices concurrently provide services within the network, the fairness of resource allocation becomes a critical issue, as it directly impacts the level of QoS [46]. In this context, it is essential to consider load balancing for UAVs acting as relays and resource utilization balancing for MeNBs serving as task executors, as balanced task distribution enhances system robustness and efficiency. It not only mitigates the risk of single-point failures and facilitates efficient

OFDM subcarrier allocation but also reduces the programming and debugging complexity associated with managing largescale parallel task execution. Long-term fairness must also be considered to ensure the rational allocation and utilization of network resources, thus promoting the sustainability and healthy development of the network ecosystem. To evaluate the load balance fairness among UAVs for wireless relaying, we use Jain’s fairness index [47]. In addition, the fairness of task offloading from the OU to its connected RUs must be addressed. Therefore, the fairness of task receiving among RUs up to time slot t is

$$
F ^ { \mathrm { r e c } } \left( t \right) = \frac { \left( \sum _ { j = 0 } ^ { J } N _ { j } \left( t \right) \right) ^ { 2 } } { J \sum _ { j = 0 } ^ { J } N _ { j } \left( t \right) ^ { 2 } }\tag{13}
$$

where $\begin{array} { r } { N _ { j } ( t ) = \sum _ { i = 1 } ^ { t } N _ { j } ^ { \mathrm { r e c } } ( i ) } \end{array}$ . After the RUs receive the tasks transmitted from the corresponding OUs, in the second stage, they need to assign the tasks to the designated associated MeNBs. In this regard, it is essential to consider the fairness of the subsequent task retransmission. Consequently, let $\begin{array} { r } { N _ { m } ( t ) ~ = ~ \sum _ { i = 1 } ^ { T } N _ { m } ^ { \mathrm { r t r } } ( i ) } \end{array}$ , and the computational load balancing among the MeNBs can be formulated as follows:

$$
F ^ { \mathrm { r t r } } \left( t \right) = \frac { \left( \sum _ { m = 0 } ^ { M } { N _ { m } \left( t \right) } \right) ^ { 2 } } { M \sum _ { m = 0 } ^ { M } { N _ { m } \left( t \right) } ^ { 2 } } .\tag{14}
$$

## E. Problem Formulation

We formulate the task offloading problem as minimizing the latency of vehicle tasks through joint optimization of task offloading and transfer strategies among heterogeneous UAVs. Due to the limited bandwidth resources in the network and the computational resources of the MEC servers mounted on MeNBs, and unlike traditional centralized decision-making [48], RUs are designed to make intelligent decisions independently, in order to reduce communication overhead during information synchronization. During the optimization process, we must account for the fact that, after vehicles generate task information at each time slot, RUs independently decide the task receiving strategy in the first stage, as well as the task retransmission strategy in the second stage. The goal of our research is to minimize the total delay of all tasks while maximizing network load balance. Thus, the optimization problem can be mathematically formulated as follows:

$$
\operatorname* { m a x } _ { T ^ { \mathrm { r e c } } ( t ) , } \sum _ { t = 1 } ^ { T } \sum _ { \nu = 1 } ^ { V \left( t \right) } L _ { \nu } ^ { \mathrm { t o t a l } } \left( t \right) \cdot F ^ { \mathrm { r e c } } \left( t \right) \cdot F ^ { \mathrm { r t r } } \left( t \right)\tag{15a}
$$

$$
\mathrm { s . t . } ( 1 0 ) , ( 1 1 ) , ( 1 2 )\tag{15b}
$$

$$
T _ { j , \nu } ^ { \mathrm { r e c } } \in \{ 0 , 1 \} \forall j \in \mathcal { T } \forall \nu \in \mathcal { V } _ { j } ( t )\tag{15c}
$$

$$
T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) = 0 \quad \forall j \in \mathcal { T } \quad \forall \nu \in \mathcal { V } \left( t \right) \backslash \mathcal { V } _ { j } \left( t \right)\tag{15d}
$$

$$
T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) = 1 \quad \forall j \in \mathcal { I } \quad \forall \nu \in \mathcal { V } _ { j } \left( t \right) \backslash \mathcal { V } _ { j } ^ { \prime } \left( t \right)\tag{15e}
$$

$$
\sum _ { j = 1 } ^ { J } T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) = 1 \quad \forall \nu \in \mathcal { V } \left( t \right)\tag{15f}
$$

$$
T _ { j , \nu } ^ { \mathrm { r t r } } \left( t \right) \in { } _ { j } \quad \forall j \in \mathcal { I } \quad \forall \nu \in \mathcal { V } _ { j } \left( t \right)\tag{15g}
$$

$$
T _ { j , \nu } ^ { \mathrm { r t r } } \left( t \right) = 0 \quad \forall j \in \mathcal { I } \quad \forall \nu \in \mathcal { V } \left( t \right) \backslash \mathcal { V } _ { j } \left( t \right) ,\tag{15h}
$$

where $\pmb { T } ^ { \mathrm { r e c } } ( t ) ~ = ~ \{ T _ { i , \nu } ^ { \mathrm { r e c } } ( t ) , j ~ \in ~ \mathcal { I } , \nu ~ \in ~ \mathcal { V } ( t ) \}$ and $\begin{array} { r l } { T ^ { \mathrm { r t r } } ( t ) } & { { } = } \end{array}$ $\{ T _ { j , \nu } ^ { \mathrm { r t r } } ( t ) , j \in \mathcal { I } , \nu \in \dot { \mathcal { V } } ( t ) \}$ ,are the vehicle task receiving and , , ,retransmission matrices of $\mathrm { R U } ~ j \in \mathcal { I }$ for each vehicle $\nu \in \mathcal { V } ( t )$ Moreover, $\begin{array} { r } { \mathcal { V } _ { j } ^ { \prime } ( t ) = \mathcal { V } _ { j } ( t ) \cap \bigcup _ { i \in \mathcal { T } \backslash \{ j \} } \mathcal { V } _ { i } ( t ) } \end{array}$ represents the set of tasks that require RU j, in collaboration with other RUs, to make decisions regarding task reception. The set of tasks delivered to RU j that only require the offloading decision for computation in the second stage is denoted separately. The binary constraint (15c) imposes a restriction on the task transmission in the first stage of hierarchical scheduling. Constraints (15d) and (15e) ensure that the RU j only provides receiving policies for tasks that are also transmitted to other RUs, where $j \in \mathcal { I }$ . Constraint (15f) reflects that each task can be relayed to exactly one RU. Each task is ultimately transmitted to a unique MeNB for computation can be guaranteed in constraint (15g). Constraint (15h) indicates that RUs do not make final task offloading decisions for tasks it has not received from OUs.

It can be readily derived that problem (15b) is a complex nonconvex problem with mixed-integer constraints. The joint optimization problem involving multiple UAVs in the proposed hierarchical task mechanism requires consideration of transmission delays at each stage of task transmission, as well as load balancing among nodes at each network layer. This problem represents a highly dynamic hybrid scheduling challenge, necessitating real-time decision-making at each time slot. As the optimization problem presented in our paper can be regarded as a well-known generalized assignment problem (GAP), which has been proven as an NP-hard problem, so the optimization problem is also NP-hard [36]. Traditional optimization methods, such as linear programming and genetic algorithms, are insufficient to solve this problem efficiently [49]. Since each UAV makes decisions based on its varying local observations while accounting for the stochastic nature of the environment and the actions of other agents, existing decision-making approaches, as discussed in [11] and [17], are limited by their reliance on complete global information, which is often impractical to obtain in real-time due to communication delays and bandwidth constraints.

In this case, the proposed problem can be regarded as a sequential decision-making problem in the time-varying UAVassisted IoV system. Given that each UAV makes decisions based on its local observations and must account for the stochastic nature of the environment and the actions of other agents, the state transition probabilities of the typical MDP quadruple are unknown. Therefore, the optimization problem can be reformulated as a POMDP [25]. Considering the inherent challenges of unknowable transition probabilities and the necessity for deterministic decision-making strategies, DRL is employed to provide an effective solution [50]. As each UAV can be regarded as an independent agent, with the collective objective of maximizing the earliness of task completion while ensuring network load balancing, we adopt a multiagent framework following the current trend of CTDE to enhance system responsiveness and reduce reliance on global information [51]. As a CTDE-based architecture, MAPPO is particularly wellsuited for this purpose due to its clipped surrogate objective inherited from PPO, which ensures stable policy updates and prevents destabilizing large gradient steps, while its critics evaluate the centralized state–value function of the global environment state [35]. It supports high-dimensional discrete action spaces and offers robust performance in cooperative multiagent environments with straightforward implementation [52]. Considering that UAV relay decision-making is characterized as a long-sequence model in training, leveraging past information during training is critical. To address this, we integrate a gated recurrent unit (GRU) into the neural networks of MAPPO, which enhances experience replay buffer utilization to accelerate convergence while reducing computational costs and enabling lightweight model design compared to other recurrent neural network (RNN) variants.

## III. PROPOSED MULTIAGENT DRL APPROACH: MAHHV

The aforementioned problem is a mixed-integer programming problem characterized by high environmental dynamism, as well as the diversity and uncertainty of vehicle tasks. Traditional optimization algorithms, which rely on static snapshots of the environment, struggle to adapt to real-time conditions, leading to inefficiencies in large-scale IoV systems. To address these challenges, and considering the limited coverage and computing capabilities of a single UAV, we leverage an MADRL approach. In this section, we formulate the problem as a POMDP, allowing agents to learn optimal policies through continuous interaction with the IoV environment. We then propose a MAHHV, implementing a CTDE mechanism that utilizes a centralized value function to evaluate the global environmental state. The proposed method enables the determination of decision variables through the collaborative strategies of multiple heterogeneous UAVs.

## A. MDP Formulation for Heterogeneous UAV-Enabled Vehicular Networks

In this network, there are multiple OUs responsible for observing vehicle task information and RUs that make task transmission decisions. The RUs consist of two types: I-RUs and C-RUs. I-RUs are responsible for making receiving and computational assignment decisions for tasks in nonoverlapping coverage regions of the OUs, while C-RUs possess the additional capability to make decisions for tasks that across multiple coverage areas. Each $\mathrm { R U } ~ j \in \mathcal { I }$ can be seen as an agent to make decisions on task reception and offloading based on the task information forwarded by the associated OU.

Typically, the scenario in which multiple UAVs jointly provide task offloading services for ground vehicles can be regarded as a multiagent collaboration sequential decisionmaking problem, generally termed as a POMDP [53]. For the task of distributing optimization in the IoV network, each RU is considered as an agent. Herein, we formally define the POMDP as a seven-tuple hS A O P Z R, i [54]. S is the global state space of the environment and O is the observed states for all RUs, i.e., $\pmb { \mathcal { O } } = \{ \pmb { O } _ { 1 } , \ldots , \pmb { O } _ { j } , \ldots , \pmb { O } _ { J } \} . \pmb { \mathcal { Z } } ( \pmb { o } ( t ) \mid s ( t ) )$ , . . . , , . . . ,represents the mapping from global state s(t) to observation ${ \pmb o } ( t )$ . In addition, the $\pmb { \mathcal { A } } = \{ A _ { 1 } , \dotsc , A _ { j } , \dotsc , A _ { J } \}$ is the joint , . . . , , . . . ,action space. P is the distribution of transition probability $P ( s ( t + 1 ) | | s ( t ) , a ( t ) )$ , which indicates the probability of transitioning to a new state $s ( t + 1 )$ given the current state s(t) and the joint action a(t) taken by all agents. $r ( t ) = \mathcal { R } ( s ( t ) , a ( t ) )$ ,is defined as the immediate rewards received by agents during state transitions. $\gamma \in [ 0 , 1 ]$ is the discount factor, which deterγ ,mines the importance of future rewards relative to immediate rewards. The final goal of the our cooperative multiagent model is to maximize the long-time cumulative discounted reward from all UAV agents, which is calculated by a discount factor $\gamma \in [ 0 , 1 )$ determines the weight of the future reward γ ,and can be described as $\begin{array} { r } { R ^ { \mathrm { { t o t a l } } } = \sum _ { i = 0 } ^ { J } \sum _ { t = 0 } ^ { T } \gamma ^ { t } r ( t ) } \end{array}$

1) Environmental State $\begin{array} { r l r } { s ( t ) } & { { } \in } & { s \colon } \end{array}$ γAccording to the proposed multilayer architecture of the heterogeneous UAVenabled vehicular network, the environment state comprises the information of vehicles in the area-of-interest, the cumulative amount of tasks received by each RU and executed by each MeNB until each time slot. Thus, the environmental state at time slot t can be formalized as follows:

$$
\begin{array} { r l } & { s \left( t \right) } \\ & { = \left\{ P _ { 1 } \left( t \right) , \ldots , P _ { \nu } \left( t \right) , \ldots , P _ { V \left( t \right) } \left( t \right) , D _ { 1 } \left( t \right) , \ldots , \right. } \\ & { \qquad \left. D _ { \nu } \left( t \right) , \ldots , D _ { V \left( t \right) } \left( t \right) , C _ { 1 } \left( t \right) , \ldots , C _ { \nu } \left( t \right) , \ldots , \right. } \\ & { \qquad \left. C _ { V \left( t \right) } \left( t \right) , L _ { 1 } \left( t \right) , \ldots , L _ { \nu } \left( t \right) , \ldots , L _ { V \left( t \right) } \left( t \right) , N _ { 1 } \left( t \right) , \ldots , \right. } \\ & { \qquad \left. N _ { j } \left( t \right) , \ldots , N _ { J } \left( t \right) , N _ { 1 } \left( t \right) , \ldots , N _ { m } \left( t \right) , \ldots , N _ { M } \left( t \right) , t \right\} } \end{array}\tag{16}
$$

where $P _ { \nu } ~ = ~ ( x _ { \nu } ( t ) , y _ { \nu } ( t ) , z _ { \nu } ( t ) )$ is the coordinates of vehicle $\nu \in \mathcal { V } ( t )$ and $D _ { \nu } ( t ) , C _ { \nu } ( t )$ , and $L _ { \nu } ( t )$ are the detailed task information of the vth vehicle, including the task size, the required CPU cycles for computation, and the tolerable latency. $N _ { j } ( t )$ and $N _ { m } ( t )$ are the total amount of tasks received by RU j and executed by MeNB m until time slot t.

2) Observations $o _ { j } ( t ) \in O _ { j } .$ : From the perspective that RU only receives task information from its associated OU, and each OU only collects the task information from vehicles within its coverage region. As there is no information exchange among different OUs, the observation of each RU at time slot t can be described as follows:

$$
\begin{array} { c } { { \sigma _ { j } \left( t \right) = \left\{ P _ { 1 } \left( t \right) , \ldots , P _ { \nu } \left( t \right) , \ldots , P _ { V _ { k _ { j } \left( t \right) } \left( t \right) , D _ { 1 } \left( t \right) , \ldots , } \right. } } \\ { { \left. D _ { \nu } \left( t \right) , \ldots , D _ { V _ { k _ { j } \left( t \right) } \left( t \right) , C _ { 1 } \left( t \right) , \ldots , C _ { \nu } \left( t \right) , \ldots , } \right. } } \\ { { \left. C _ { V _ { k _ { j } \left( t \right) } } \left( t \right) , L _ { 1 } \left( t \right) , \ldots , L _ { \nu } \left( t \right) , \ldots , L _ { V _ { j } \left( t \right) } \left( t \right) \right. } } \\ { { \left. N _ { j } \left( t \right) , N _ { 1 } \left( t \right) , \ldots N _ { m } \left( t \right) , \ldots , N _ { M _ { j } } \left( t \right) , t \right\} . } } \end{array}\tag{17}
$$

Each RU is able to obtain the task information of $\nu \in \mathcal { V } _ { j } ( t )$ from its associated OU at time slot t, which consists of the vehicle coordinates and the generated task size, required computational resource, and tolerable latency. In addition, each RU can update the total number of tasks it has cumulatively received until each time slot, as well as the amount of tasks retransmitted to its associated MeNBs.

3) Actions $a _ { j } ( t ) \in A _ { j } .$ At each time slot t, RU takes an action ${ \pmb a } _ { j } ( t )$ , which includes the selection of task reception from each vehicle and the decision of the target MeNB to which the received tasks are retransmitted for execution. According to the corresponding observation and the current decision policy $\pi _ { j } ,$ the action taken by the jth RU can be πwritten as follows:

$$
\begin{array} { r } { \pmb { a } _ { j } \left( t \right) = \left\{ T _ { j , 1 } ^ { \mathrm { r e c } } \left( t \right) , \ldots , T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) , \ldots , T _ { j , V _ { j } ^ { \prime } \left( t \right) } ^ { \mathrm { r e c } } \left( t \right) \right. } \end{array}
$$

$$
T _ { j , 1 } ^ { \mathrm { r t r } } \left( t \right) , \ldots , T _ { j , \nu } ^ { \mathrm { r t r } } \left( t \right) , \ldots , T _ { j , V _ { j } \left( t \right) } ^ { \mathrm { r t r } } \left( t \right) \Biggr \}\tag{18}
$$

where binary variable $T _ { j , \nu } ^ { \mathrm { r e c } } ( t )$ denotes the task receiving deci-,sion of the j RU, while the integer variable $T _ { j , \nu } ^ { \mathrm { r t r } } ( t ) \ \in \ \mathcal { M } _ { j }$ ,indicates the MeNB selection for task offloading and computation. By combining (15d) and (15e), all values of $T _ { j , \nu } ^ { \mathrm { r e c } } ( t ) \forall j \in$ $\mathcal { I } \ \forall \nu \in \mathcal { V } ( t )$ , can be obtained. To ensure the validity of constraint (15f), we modify the actual execution scheme of the binary action $T _ { j , \nu } ^ { \mathrm { r e c } } ( t )$ . The reconstructed action is defined as follows:

$$
T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } \left\{ 1 - \sum _ { i = 1 } ^ { j - 1 } T _ { i , \nu } ^ { \mathrm { r e c } } \left( t \right) , T _ { j , \nu } ^ { \mathrm { r e c } } \left( t \right) \right\} , } & { j < J } \\ { \displaystyle \operatorname* { m a x } \left\{ 1 - \sum _ { i = 1 } ^ { j - 1 } T _ { i , \nu } ^ { \mathrm { r e c } } \left( t \right) , 1 \right\} , } & { j = J } \end{array} \right.\tag{19}
$$

which indicates that the collection strategy represented by $T _ { j , \nu } ^ { \mathrm { r e c } } ( t ) = 1$ is effective only when the other RUs numbered ,less than j choose not to collect the task data from vehicle v.

4) Rewards r(t): The reward is a feedback signal received by an agent from the environment, which reflects the immediate benefit or utility of an action taken in a given state. Through continuous interaction with the environment, the agent iteratively improves its policy to maximize the cumulative rewards. Based on the optimization problem formulated in (15b), the objective is to maximize the total earliness of task completion while ensuring the even distribution of tasks across the network and maintaining the computational load balancing among MeNBs. In this system, each RU can be regarded as an independent agent, cooperating with each other to pursue the same optimization objective. Therefore, all agents share the immediate reward $r _ { j } ( t )$ and collaboratively transmitting and offloading tasks in the dynamic MEC-assisted IoV network. Accordingly, the reward is composed of three elements: the total earliness of task completion, the fairness of task receiving among RUs, and the fairness of task computation among MeNBs. The immediate reward r(t) is defined as follows:

$$
r \left( t \right) = \sum _ { \nu = 1 } ^ { V \left( t \right) } L _ { \nu } ^ { \mathrm { t o t a l } } \left( t \right) \cdot F ^ { \mathrm { r e c } } \left( t \right) \cdot F ^ { \mathrm { r t r } } \left( t \right)\tag{20}
$$

where $L _ { \nu } ^ { \mathrm { t o t a l } } ( t )$ denotes the earliness of task completion for vehicle $\nu \in \mathcal { V } ( t )$ . The subsequent two parameters denote the distribution balance of tasks across the network and the load balance metric on MeNBs after each time slot, both of which are defined by Jain’s fairness index.

## B. CTDE-Based MAHHV Algorithm Design

We proposed an improved MAPPO-based on-policy scheme to solve the joint optimization problem of task transmission and offloading among multiple UAVs collaborating within an MEC-assisted IoV network. MAPPO is one of the state-of-theart MARL algorithms that is designed to handle multiagent environments effectively. It builds upon the proximal policy optimization (PPO) framework, which is well-known for its stability and efficiency in policy training [55]. The CTDE framework has seen widespread adoption in recent years due to its advantages of stable convergence and fast execution speed.

The proposed MAHHV algorithm profited by the actor–critic architecture and a CTDE framework is adopted as illustrated in Fig. 3. The RUs execute task offloading according to the actions generated by their actor networks in the physical environment, while transmitting their local experiences and synchronized system states to the DT layer. During the training phase, a global state–value function is employed to evaluate the joint state of the environment, incorporating information from all RU agents. This enables the RUs to learn cooperative strategies while considering the entire network state, including task distribution, network conditions, and computational load status. After updating both the actor and critic networks, the parameters of the actor networks are distributed back to RUs. In the execution phase, each RU agent operates independently, making decisions based on its local observation and autonomously adapting to realtime environmental change, which enables it to be suitable for scenarios requiring collaboration, especially in the realworld dynamic and distributed environments. To maintain the memory of past information in the model, two GRUs are introduced before the output layer of each actor and critic network to enable the model to retain memory of historical data, while all other hidden layers are composed of multilayer perceptrons (MLPs). Compared with long short-term memory (LSTM), GRU’s simpler gating structure reduces the number of parameters, accelerates training, and requires fewer data in training, while also outperforming LSTM on nonlanguage modeling tasks with shorter sequences [34], [56], [57]. Each GRU unit has two inputs: the previous hidden state $h _ { t - 1 }$ and the output of the preceding layer. The hidden state output $h _ { t }$ is obtained after processing by the reset gate and update gate. This allows the GRU to propagate historical information from the current time step to the next, effectively capturing longterm dependencies in the sequence model. Specifically, let $W _ { r }$ $W _ { z } ,$ and $W _ { h }$ be the weight matrices for the reset gate, update gate, and the current input, respectively. The key calculation formulas can be expressed as follows:

$$
r _ { t } = \sigma \left( W _ { r } \left[ h _ { t - 1 } , x _ { t } \right] + b _ { r } \right)\tag{21}
$$

$$
z _ { t } = \sigma ( W _ { z } \left[ h _ { t - 1 } , x _ { t } \right] + b _ { z } )\tag{22}
$$

$$
\tilde { h } _ { t } = \operatorname { t a n h } { ( W _ { h } [ r _ { t } \otimes h _ { t - 1 } , x _ { t } ] + b _ { h } ) }\tag{23}
$$

$$
h _ { t } = ( 1 - z _ { t } ) \otimes h _ { t - 1 } + z _ { t } \otimes \tilde { h } _ { t }\tag{24}
$$

where $b _ { r } , \ b _ { z }$ , and $b _ { h }$ are the corresponding biases, and $\sigma$ σis the sigmoid activation function. Thus, the output of the GRU cell is then fed into the rest of the network, while the hidden state is recursively passed to future time steps, enabling agents to effectively model and adapt to dynamic environmental conditions.

In the proposed framework, for agent $j \in \mathcal { I } ,$ let $\theta _ { j }$ and $\omega _ { j }$ be the parameters of its actor network $\pi _ { \theta _ { j } }$ θand critic network $V _ { \omega _ { j } }$ πθ, respectively. At each time slot t, the agents ωinput their own local observations $o _ { j } ( t )$ into actor networks and obtain the $a _ { j } ( t )$ . After receiving and retransmitting the tasks from vehicles, the MeNBs perform the computation, and the UAVs get rewards from the environment. Throughout an episode, the agents continuously store the generated experiences $\{ o _ { j } ( t ) , a _ { j } ( t ) , p r _ { j } ( t ) , r _ { j } ( t ) , s ( t ) , \nu _ { j } \}$ in their separated buffers $\pmb { { \cal B } } _ { j }$ with finite size B, where $p r _ { j } ( t )$ is the log-probability for sampling $a _ { j } ( t )$ . As MAHHV requires experience of all time slots within an episode to train the actor–critic networks at end of an episode, the historical experience samples are stored in chronological order in the replay buffer [58]. Then, in the centralized training phase, the policy and value networks are updated based on the temporal-difference error and the advantage estimates. It is worth noting that in MAHHV, the advantage function is typically approximated using generalized advantage estimation (GAE), which helps to reduce variance in policy gradient updates while maintaining low bias, thereby improving the stability and efficiency of the learning process. GAE can be defined as follows:

<!-- image-->  
Fig. 3. MAHHV framework in the UAV- and MEC-assisted vehicular network.

$$
\begin{array} { l } { \displaystyle \hat { A } _ { j } \left( t \right) = \sum _ { l = 0 } ^ { \infty } \left( \gamma \lambda \right) ^ { l } \left( r _ { j } \left( t + l \right) \right. } \\ { \displaystyle \left. + \gamma V _ { \omega _ { j } } \left( s \left( t + l + 1 \right) \right) - V _ { \omega _ { j } } \left( s \left( t + l \right) \right) \right) } \end{array}\tag{25}
$$

where $\gamma$ is the discount factor and  is the tunable parameter γ λthat controls the bias-variance tradeoff in the advantage estimation. Then, the clipped surrogate objective function from PPO is introduced to stable policy updates and reduce the risk of destabilizing the learning process, which is written as follows:

$$
\begin{array} { r l } & { \mathcal { T } _ { j } ^ { \mathrm { c l i p } } } \\ & { = \mathbb { E } _ { t } \{ \operatorname* { m i n } [ \frac { \pi _ { \theta _ { j } } ( a _ { j } ( t ) \mid o _ { j } ( t ) ) } { \pi _ { \theta _ { j } ^ { \prime } } ( a _ { j } ( t ) \mid o _ { j } ( t ) ) }   } \\ & { \qquad \mathrm { c l i p } (  \frac { \pi _ { \theta _ { j } } ( a _ { j } ( t ) \mid o _ { j } ( t ) ) } { \pi _ { \theta _ { j } } ( a _ { j } ( t ) \mid o _ { j } ( t ) ) } , 1 - \varepsilon , 1 + \varepsilon ) ] \hat { A } _ { j } ( t ) \} } \end{array}\tag{26}
$$

where $\theta _ { j }$ and $\theta _ { j } ^ { \prime }$ are the new and the old parameters of the θ θpolicy network.  is the clip fraction that controls the magεnitude of allowable policy updates. To encourage exploration, a policy entropy $H ( \pi _ { \theta _ { j } } ) = - \mathbb { E } _ { a _ { j } ( t ) \sim \pi _ { \theta _ { i } } } [ \log ( \pi _ { \theta _ { j } } ( a _ { j } ( t ) \mid s _ { j } ( t ) ) ) ]$ is πθ πθ  πθoften included as a regularization term in the policy’s loss function which can be formulated as follows:

$$
L \left( \boldsymbol { \theta } _ { j } \right) = \mathcal { T } _ { j } ^ { \mathrm { c l i p } } + \psi H \left( \boldsymbol { \pi } _ { \boldsymbol { \theta } _ { j } } \right)\tag{27}
$$

where $\psi$ is the entropy coefficient hyperparameter to encourage ψexploration. On the other hand, to minimize the mean squared error of the value function, the gradient descent algorithm is used to update the critic network. The loss function of the critic network is obtained by

$$
L \left( \omega _ { j } \right) = \mathbb { E } _ { t } \left[ \left( r _ { j } \left( t \right) + \gamma V _ { \omega _ { j } } \left( s \left( t + 1 \right) \right) - V _ { \omega _ { j } } \left( s \left( t \right) \right) \right) ^ { 2 } \right] .\tag{28}
$$

The proposed MAHHV is summarized in Algorithm 1. The algorithm begins with initializing the training hyperparameters and the network parameters for both the actor and critic networks of each RU agent. The environment state is reset at the beginning of each episode, and each RU selects an action according to its current observation and policy network in each time slot (lines 3–8). The rewards and the log probabilities can be calculated and stored in the replay buffer associated with the environment state, observations, actions, and state value. Subsequently, the environmental state is updated (lines 10–13). After each episode, samples are extracted from the replay buffer, and the GAE method is employed to compute the estimated advantage for each RU (lines 15–18). Subsequently, the loss of the actor and critic networks is obtained by (27) and (28), and the optimization tool Adam optimizer is adopted.

```perl
Algorithm 1 MAHHV
Initialize :
Initialize the maximum training episodes E,episode
length T,ppo epochs $P e .$
Initialize the actor network $\theta _ { j }$ , critic network $\omega _ { j } .$
replay buffer $\pmb { { \cal B } } _ { j }$ for each agent $j \in \mathcal { I } .$
Algorithm:
foreach episode $e = 1 , 2 , \ldots , E$ do
Initialize the state $s ( t )$ ,reset $r = 0 .$
foreach time slot $t = 1 , 2 , \dots , T$ do
foreach RU $j = 1 , 2 , \dots , J$ do
Obtain the current observation ${ \pmb o } _ { j } ( t ) .$
Executes action $a _ { j } ( t )$ according to
$\pi _ { \theta _ { j } } ( a _ { j } ( t ) \mid o _ { j } ( t ) )$
end
Evaluate reward $r _ { j } ( t )$
Calculate log-probability $p r _ { j } ( t )$
Insert $\{ s ( t ) , o _ { j } ( t ) , a _ { j } ( t ) , r _ { j } ( t ) , p r _ { j } ( t ) , v _ { j } \}$ into
the replay buffer $\pmb { { \cal B } } _ { j }$
The environmental state is updated to $s ( t + 1 )$
end
foreach epoch $e p = 1 , 2 , \ldots , P _ { e }$ do
Obtain experiences from replay buffer.
foreach RU $j = 1 , 2 , \dots , J$ do
Calculate GAE ${ \hat { A } } _ { j } ( t )$ according to (25).
Calculate the loss of the actor network $\pi _ { \theta _ { j } }$
according to (27).
Calculate the loss of the critic network $V _ { \theta _ { j } }$
according to (28).
Update the actor network parameters $\theta _ { j }$ via
gradient descent algorithm by Adam.
Update the critic network parameters $\omega _ { j }$
via gradient descent algorithm by Adam.
end
end
Download actor networks from DT layer to RUs.
end
```

The gradient descent method is applied to update the network parameters (lines 19–22). The updated actor network parameters are then propagated to RUs for decentralized execution (lines 25).

## C. Complexity Analysis

In this framework, the complexity of MAHHV mainly comes from the forward and backward propagations. For an MLP layer, the number of neurons for each MLP layer i is indicated as $n _ { i } ,$ and $n _ { s }$ is the dimension of the state space. In the network training phase, the computational complexity of an I-layer MLP can be calculated by $\begin{array} { r } { \mathcal { O } \left( \sum _ { i = 1 } ^ { I - 1 } n _ { i } n _ { i + 1 } \right) } \end{array}$ Furthermore, the computational complexity of the input layer is given by $\mathcal { O } ( n _ { s } n _ { 1 } )$ . The complexity of GRU is $\mathcal { O } ( n _ { G } ( n _ { G } +$ $n _ { I } ) )$ , where $n _ { G }$ is the hidden state of GRU, which is also equal to the size of the unit’s output, considering that it has two hidden layers. Therefore, considering the actor and critic are constructed with the same architecture, the overall computational complexity for all agents within all E episodes is $\begin{array} { r } { \mathcal { O } \left( 2 E T J \left( n _ { s } n _ { 1 } + \sum _ { i = 1 } ^ { I - 1 } n _ { i } n _ { i + 1 } + 2 n _ { G } ( n _ { G } + n _ { I } ) \right) \right) } \end{array}$

## IV. SIMULATION RESULTS

In this section, the performance of the proposed MAHHV algorithm is evaluated, demonstrating its effectiveness in improving task offloading efficiency and enhancing the collaboration among UAVs in dynamic IoV environments.

## A. Simulation Setting

In the simulation, a specific road segment is carefully selected for experimental validation. The road stretches 500 m in length, with vehicles evenly distributed along it and traveling at an average speed of 25 m/s. The Y-axis of the 3-D coordinate system is established along the road. All UAVs are randomly distributed and hovered at an altitude of 50 m, providing services to the ground vehicles. The number of OUs is $K = 2$ , and they are evenly distributed above the road to observe and collect vehicle tasks, with horizontal coordinates of [150, 0] and [350, 0] m, respectively. Furthermore, the RUs are randomly distributed on both sides away from the road, with the x-coordinate set with $x _ { k } \ \in \ [ 0 ,$ 500] m and the $y -$ coordinate set with $y _ { k } \in [ - 1 0 0 .$ , 100] m. Finally, we assume that each RU is associated with a set of MeNBs that share a microwave band, with their horizontal positions randomly set as xm ∈ [0, 500] and $y _ { m } \in [ - 2 0 0 , 2 0 0 ]$ m. In the subsequent sections, unless otherwise specified, the number of RUs is set as $J \ = \ 6 ,$ with each RU associated with two MeNBs. The simulation is implemented by Python 3.12 and Torch 2.2.2 on a desktop computer with an Intel Core i5-12500 3000-MHz CPU and 16-GB RAM. Other main parameters are summarized in Table II. For specific UAV configurations, please refer to the DJI Inspire 2 equipped with Jetson Nano.

## B. Network Architecture

The specific network is shown in Fig. 3, where both the critic and actor networks consist of a layer of fully connected MLPs and two GRU layers. All agents have the same network structure; the input dimension of the actor network depends on the vehicle information observed by all RUs and the number of associated MeNBs. The output dimension depends on the RU that needs to make the most receiving and offloading decisions. Thus, the input layer of each actor network has $\operatorname* { m a x } \{ 6 V _ { k _ { i } } ( t ) + M _ { j } + 2 \forall j \in \mathcal { I } \forall t \in \mathcal { T } \}$ neurons, and the output layer has max $\{ V _ { i } ^ { \prime } ( t ) + V _ { j } ( t ) ~ \forall j ~ \in ~ \mathcal { T } ~ \forall t ~ \in ~ \mathcal { T } \}$ neurons. Similarly, the input of the critic network of each agent consists of the vehicle information observed by all RUs, the cumulative number of tasks they have received, and the total number of tasks cumulatively executed by MeNBs. The input dimension of the critic network can be calculated as $\operatorname* { m a x } \{ 6 V ( t ) + J + M + 1 \ \forall t \in \mathcal { T } \}$ . To maintain dimensional consistency among the agents, the actor network inputs are zero-padded to match the maximum input dimension, while the outputs are masked so that only the valid action dimensions are preserved and the remaining entries are ignored.

TABLE II SIMULATION PARAMETERS
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Number of vehicles to offload tasks V(t)</td><td>[5, 10,15]</td></tr><tr><td>Number ofRUs J</td><td>[4,6,8]</td></tr><tr><td>NumberofMeNBsM</td><td>[8, 12, 16]</td></tr><tr><td>Transmission power of the vehicles  $P _ { \mathrm { v h c } }$ </td><td>0.2 W[1]</td></tr><tr><td>Transmission power of OUs  $P _ { \mathrm { O U } }$ </td><td>1W[59]</td></tr><tr><td>Transmission power of RUs  $P _ { \mathrm { R U } }$ </td><td>0.2W [60]</td></tr><tr><td>Bandwidth of OUs to receive tasks  $B _ { \mathrm { O U } } ^ { \mathrm { r e c } }$ </td><td>20 MHz [25]</td></tr><tr><td>Bandwidth of OUs to transmit tasks  $B _ { \mathrm { O U } } ^ { \mathrm { t r } }$ </td><td>20 MHz [25]</td></tr><tr><td>Bandwidth ofRUs to re-transmit tasks  $B _ { \mathrm { R U } } ^ { \mathrm { r t r } }$ </td><td>200 MHz [25]</td></tr><tr><td>Task size of vehicle U  $D _ { v }$  TaskCPUcyclesdemandof vehicle  $v C _ { v }$ </td><td>[0.1,0.6]MB[61]</td></tr><tr><td>Task tolerable latency of vehicle U  $L _ { v }$ </td><td>[0.1,0.6] GHz [61]</td></tr><tr><td>LoS transmission pathloss</td><td>[0.2,1] s [62]</td></tr><tr><td> $\eta _ { \mathrm { L o S } }$  NLoS transmission pathloss /NLoS</td><td>1 dB [63]</td></tr><tr><td></td><td>20 dB [63]</td></tr><tr><td>S-curve parameters a, b</td><td>9.6, 0.16 [64]</td></tr><tr><td>Noise power spectral density  $N _ { 0 }$ </td><td>-174dBm/Hz [41]</td></tr><tr><td>Carrier frequencies f</td><td>2GHz [64]</td></tr><tr><td>CPU frequency of MeNB m  $f _ { m }$ </td><td>[18,50] GHz [41]</td></tr><tr><td>Number of time slots in an episode T</td><td>100</td></tr><tr><td>Number of ppo epochs  $P _ { e }$ </td><td>15</td></tr><tr><td>PPO clip parameter ε</td><td>0.2</td></tr><tr><td>Discounted factor y</td><td>0.99</td></tr><tr><td>GAE parameter 入</td><td>0.95</td></tr><tr><td>Entropy coefficient 亚</td><td>0.01</td></tr><tr><td>Number of hidden layers</td><td>3</td></tr><tr><td>Number of neurons in hidden layers</td><td>512</td></tr><tr><td>Learning rate of actor and critic networks</td><td>2e-5</td></tr><tr><td>Learning rate decay factor</td><td>1e-9</td></tr></table>

## C. Performance of the Proposed Algorithm

We compare the proposed MAHHV with four baseline methods, GMAPPO, self-interested PPO, random policy, and evenly policy, as referenced in prior works [53], [65], [66].

1) GMAPPO: The general MAPPO in which both actor and critic networks are constructed by three-layer MLPs.

2) Self-Interested PPO: Self-interested hierarchical scheduling scheme based on PPO, where each RU acts as an independent agent with a decentralized training mode.

3) Random Policy: The RUs randomly generate task collection and computational offloading strategies based on the task information received from OUs.

4) Uniform Policy: Based on the random strategy, the RUs evenly distribute the received tasks to the associated MeNBs in the second stage.

Since the learning rate can significantly affect the training convergence speed, we evaluate the performance of the proposed MAHHV algorithm under different learning rates to find the optimal balance between convergence speed and overall system performance. The models are tested in a UAVassisted IoV system consisting of 10 vehicles, 2 OUs, 6 RUs, and 12 MeNBs. The results, as shown in Fig. 4, indicate that as the number of training episodes increases, the average reward gradually rises while network fluctuations decrease. This is because the agents start to leverage the knowledge they have acquired. Comparatively, when the learning rate for both the actor and critic networks was set to $5 \mathrm { e } ^ { - 5 }$ , a good balance between convergence speed and average system establishment time is achieved, resulting in improved model training efficiency. Thus, in the subsequent experiments, we used a learning rate of $5 \mathrm { e } ^ { - 4 }$ for performance evaluation.

<!-- image-->  
Fig. 4. Convergence of MAHHV under different learning rates.

<!-- image-->  
Fig. 5. Convergence of different algorithms.

Fig. 5 illustrates the learning curves of different scheduling schemes for a UAV-assisted IoV system: 1) the scheme based on MAHHV; 2) the scheme based on general MAPPO; 3) the self-interested scheme based on PPO; 4) the uniform policy; and 5) the random policy. The convergence speed and performance of these five algorithms are compared. It can be observed that PPO reaches convergence first as training episodes increase, primarily because it focuses solely on the behavior of a single agent. However, this leads to poorer performance in cooperative scenarios, characterized by higher fluctuations and a reduced ability to adapt to environmental changes. In contrast, the MAHHV and GMAPPO algorithms achieve higher average rewards, as they enable agents to cooperate effectively. Among these, MAHHV demonstrates superior performance. This advantage is attributed to the use of GRU, a recurrent neural network that can effectively capture dependencies in time-series data, thereby generating more optimal strategies. In addition, GRU’s ability to retain historical information allows it to adapt to dynamic multiagent environments, enhancing the stability and robustness of the strategies. Moreover, MAHHV and GMAPPO can significantly outperform random allocation and uniform distribution by promoting agent cooperation. Although uniform distribution ensures a fairer task offloading process, leading to higher average rewards compared to random allocation, the cooperative nature of MAHHV and GMAPPO ultimately enables them to achieve better overall performance.

<!-- image-->  
Fig. 6. Performance comparison of different number of RUs.

<!-- image-->  
Fig. 7. Total delay of task transmission and computational process.

In Fig. 6, we compare the performance of the five schemes under varying numbers of RU agents, where each RU is associated with two MeNBs. The results demonstrate that the proposed MAHHV algorithm consistently outperforms the baseline algorithms. Compared with the GMAPPO algorithm without GRU, MAHHV exhibits superior capability in observing dynamic environments and robustness, with its advantages becoming more pronounced as the number of agents increases. Compared with PPO, which lacks a centralized evaluation mechanism, MAHHV excels in promoting better cooperation among agents. This shows that MAHHV effectively optimizes strategies to achieve faster task completion in vehicular networks while ensuring load balancing. The uniform policy, which performs more equitable task offloading in the second stage, achieves better fairness, resulting in superior performance compared with the random allocation strategy. In addition, it can be observed that as the number of RUs increases, the average reward slightly decreases. This is because although more UAVs and MeNBs can provide additional computational resources, enabling tasks to be completed more quickly, the growing number of resource providers also complicates the fair distribution of tasks. This may result in higher variability, which in turn reduces the overall fairness with each episode.

We further analyzed the average latency distribution of tasks during the two-stage transmission. Fig. 7 illustrates that the average task latency decreases as the number of RUs increases.

<!-- image-->  
Fig. 8. Load fairness comparison of different algorithms.

<!-- image-->  
Fig. 9. Performance comparison under different number of vehicles.

The change in the number of RUs has little impact on the latency in the first stage, as the transmission delay primarily depends on the link bandwidth allocated to each task by the OUs. Since the total received and forwarded data bandwidth of OUs remains constant, the average transmission delay in the first stage remains unchanged. In the second stage, we can see that as the number of RUs increases, the retransmission and computational delays gradually decrease. This is due to the additional bandwidth and computational resources provided by the increased RUs and MeNBs. Furthermore, the analysis shows that the proposed DRL-based algorithms have a significant decrease in system latency compared with other methods. This can be attributed to the memory capability of GRU, which allows it to retain historical interaction experiences, thereby improving the task offloading performance in multilayer vehicular networks and enhancing QoS for users. The introduction of more agents heightens the dynamic nature of the environment, expands the dimensionality of the observation and decision-making space, and increases the difficulty of the collaborative optimization problem.

Fig. 8 demonstrates that MAHHV ensures a more equitable distribution of system resources, minimizing task congestion in certain network areas. Experimental results show that, compared with GMAPPO, PPO, uniform, and random, the proposed method achieves improvements of 0.895%, 2.31%, 4.72%, and 6.21% in load fairness, respectively, without compromising overall system performance. This enhancement is critical in scenarios where fairness in resource allocation is a key performance metric, such as in multiagent environments or networks with varying load conditions. Therefore, the proposed approach not only reduces latency but also guarantees a more balanced and fair system performance, contributing to the overall efficiency and stability of the system.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

(d）  
<!-- image-->

<!-- image-->  
(f)

<!-- image-->

<!-- image-->

i  
<!-- image-->  
i  
Fig. 10. Accumulation of tasks transmitted to each RU and offloaded to each MeNB in an episode. (a)–(f) Task amount changes under MAHHV, GMAPPO, and PPO, and (g)–(j) task load changes of uniform and random.

In addition, we conducted a comparative analysis of the system performance under varying vehicle task numbers. The results in Fig. 9 show that, as expected, the proposed MAHHV outperforms the others, thanks to the GRU’s ability to effectively capture long-term dependencies, allowing agents to better adapt to the dynamic nature of the environment and manage resources more efficiently. Moreover, the centralized training mechanism helps to coordinate agent decisions more effectively, mitigating the negative impacts of resource contention and improving the overall stability and performance of the system. Compared with GMAPPO, PPO, uniform, and random approaches, our scheme achieves an average improvement in cumulative reward by 0.6%, 1.2%, 7.1%, and 8.8%, respectively.

Fig. 10 further illustrates the dynamic changes in load fairness during one episode with $J ~ = ~ 6 ~ \mathrm { R U s }$ , where the legend on the right defines the color scheme for different task reception and offloading quantities. It can be observed that all algorithms achieve favorable QoE metrics, indicating that tasks are distributed fairly evenly across the network. However, compared with the other four baseline methods, the proposed MAHHV stands out because it incorporates both agent cooperation and temporal dependency awareness of the environment, ensuring that no single node becomes overloaded, thus preventing network congestion and resource bottlenecks. By combining the memory capability of the GRU component with a centralized training and distributed execution mechanism, the system is able to operate more efficiently and stably in complex dynamic environments, enhancing the performance and scalability of the multiagent system.

## V. CONCLUSION

In this article, we have proposed a novel dual-layer architecture where heterogeneous UAVs are exploited to optimize task relaying and computational offloading in vehicular networks. By employing a multilayer scheduling framework, incorporating multiagent reinforcement learning, and modeling the problem as a POMDP, we achieved dynamic and efficient task allocation across the network. The proposed MAHHV algorithm based on the CTDE paradigm demonstrates superior scalability and adaptability to the dynamic IoV environments, enhancing the service coverage, communication reliability, and load balancing. This work lays a foundation for the further development of UAV-enabled vehicular networks and dynamic multiagent task management. It is assumed that the channel condition is perfect in our system. In future research, we will take into account more practical challenges, including severe weather impacts and high-interference environments, and further evaluate and optimize the proposed algorithm under more realistic and extreme conditions. Besides, we will consider dynamic task slicing offloading with highdimensional resource allocation in UAV-assisted hierarchical IoV networks.

## REFERENCES

[1] M. Yan, R. Xiong, Y. Wang, and C. Li, “Edge computing task offloading optimization for a UAV-assisted Internet of Vehicles via deep reinforcement learning,” IEEE Trans. Veh. Technol., vol. 73, no. 4, pp. 5647–5658, Apr. 2024.

[2] M. Dai, Z. Su, Q. Xu, and N. Zhang, “Vehicle assisted computing offloading for unmanned aerial vehicles in smart city,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 3, pp. 1932–1944, Mar. 2021.

[3] A. T. Jawad, R. Maaloul, and L. Chaari, “A multi-agent reinforcement learning-based approach for UAV-assisted vehicle-to-everything network,” in Proc. 9th Int. Conf. Control, Decis. Inf. Technol. (CoDIT), Jul. 2023, pp. 123–129.

[4] X. Dai, Z. Xiao, H. Jiang, and J. C. S. Lui, “UAV-assisted task offloading in vehicular edge computing networks,” IEEE Trans. Mobile Comput., vol. 23, no. 4, pp. 2520–2534, Apr. 2024.

[5] X. Zhou, W. Liang, J. She, Z. Yan, and K. I. Wang, “Two-layer federated learning with heterogeneous model aggregation for 6G supported Internet of Vehicles,” IEEE Trans. Veh. Technol., vol. 70, no. 6, pp. 5308–5317, Jun. 2021.

[6] C. Chen, H. Li, H. Li, R. Fu, Y. Liu, and S. Wan, “Efficiency and fairness oriented dynamic task offloading in Internet of Vehicles,” IEEE Trans. Green Commun. Netw., vol. 6, no. 3, pp. 1481–1493, Sep. 2022.

[7] L. Sun, L. Wan, J. Wang, L. Lin, and M. Gen, “Joint resource scheduling for UAV-enabled mobile edge computing system in Internet of Vehicles,” IEEE Trans. Intell. Transp. Syst., vol. 24, no. 12, pp. 15624–15632, Dec. 2023.

[8] L. Zhao et al., “MESON: A mobility-aware dependent task offloading scheme for urban vehicular edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 5, pp. 4259–4272, May 2024.

[9] K. Jiang, X. Cao, W. Song, and Q. Jiang, “DRL-based multidimensional resource scheduling for intelligent connected vehicles in UAV-assisted VEC systems,” IEEE Sensors J., vol. 25, no. 8, pp. 13871–13883, Apr. 2025.

[10] Y. Zhou, X. Ma, S. Hu, D. Zhou, N. Cheng, and N. Lu, “QoE-driven adaptive deployment strategy of multi-UAV networks based on hybrid deep reinforcement learning,” IEEE Internet Things J., vol. 9, no. 8, pp. 5868–5881, Apr. 2022.

[11] W. Zhang, Z. Lu, M. Ge, and L. Wang, “UAV-assisted vehicular edge ¨ computing system: Min-max fair offloading and position optimization,” IEEE Trans. Consum. Electron., vol. 70, no. 4, pp. 7412–7423, Nov. 2024.

[12] J. Xiao et al., “A deep reinforcement learning based distributed multi-UAV dynamic area coverage algorithm for complex environment,” Neurocomputing, vol. 595, Aug. 2024, Art. no. 127904. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231224006751

[13] S. Gu, X. Sun, Z. Yang, T. Huang, W. Xiang, and K. Yu, “Energyaware coded caching strategy design with resource optimization for satellite-UAV-vehicle-integrated networks,” IEEE Internet Things J., vol. 9, no. 8, pp. 5799–5811, Apr. 2022.

[14] Y. Su, M. Liwang, Z. Chen, and X. Du, “Toward optimal deployment of UAV relays in UAV-assisted Internet of Vehicles,” IEEE Trans. Veh. Technol., vol. 72, no. 10, pp. 13392–13405, Oct. 2023.

[15] B. Hazarika, K. Singh, A. Paul, and T. Q. Duong, “Hybrid machine learning approach for resource allocation of digital twin in UAV-aided Internet-of-Vehicles networks,” IEEE Trans. Intell. Vehicles, vol. 9, no. 1, pp. 2923–2939, Jan. 2024.

[16] Y. Liu, P. Lin, M. Zhang, Z. Zhang, and F. R. Yu, “Mobile-aware service offloading for UAV-assisted IoV: A multiagent tiny distributed learning approach,” IEEE Internet Things J., vol. 11, no. 12, pp. 21191–21201, Jun. 2024.

[17] N. Khan, A. Ahmad, A. Wakeel, Z. Kaleem, B. Rashid, and W. Khalid, “Efficient UAVs deployment and resource allocation in UAV-relay assisted public safety networks for video transmission,” IEEE Access, vol. 12, pp. 4561–4574, 2024.

[18] X. Liu, B. Lai, B. Lin, and V. C. M. Leung, “Joint communication and trajectory optimization for multi-UAV enabled mobile Internet of Vehicles,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 9, pp. 15354–15366, Sep. 2022.

[19] X. Zhu, Y. Luo, A. Liu, M. Z. A. Bhuiyan, and S. Zhang, “Multiagent deep reinforcement learning for vehicular computation offloading in IoT,” IEEE Internet Things J., vol. 8, no. 12, pp. 9763–9773, Jun. 2021.

[20] M. Wu, K. Guo, Z. Lin, X. Li, K. An, and Y. Huang, “Joint optimization design of RIS-assisted hybrid FSO SAGINs using deep reinforcement learning,” IEEE Trans. Veh. Technol., vol. 73, no. 3, pp. 3025–3040, Mar. 2024.

[21] W. Zhao et al., “A survey on DRL based UAV communications and networking: DRL fundamentals, applications and implementations,” IEEE Commun. Surveys Tuts., early access, Jun. 23, 2025, doi: 10.1109/ COMST.2025.3581912.

[22] X. Zhou et al., “Edge computation offloading with content caching in 6G-enabled IoV,” IEEE Trans. Intell. Transp. Syst., vol. 25, no. 3, pp. 2733–2747, Mar. 2024.

[23] H. Wang, Z. Lin, K. Guo, and T. Lv, “Computation offloading based on game theory in MEC-assisted V2X networks,” in Proc. IEEE Int. Conf. Commun. Workshops, Jun. 2021, pp. 1–6.

[24] W. Zhao, T. Weng, Y. Cheng, Z. Liu, and N. Kato, “Deep reinforcement learning for optimizing multi-hop distributed collaborative task offloading in R2X,” IEEE Trans. Veh. Technol., vol. 74, no. 6, pp. 9533–9548, Jun. 2025.

[25] L. Zhu, Z. Zhang, L. Liu, L. Feng, P. Lin, and Y. Zhang, “Online distributed learning-based load-aware heterogeneous vehicular edge computing,” IEEE Sensors J., vol. 23, no. 15, pp. 17350–17365, Aug. 2023.

[26] Y. Ji, Z. Yang, H. Shen, W. Xu, K. Wang, and X. Dong, “Multicell edge coverage enhancement using mobile UAV-relay,” IEEE Internet Things J., vol. 7, no. 8, pp. 7482–7494, Aug. 2020.

[27] B. Li, W. Xie, Y. Ye, L. Liu, and Z. Fei, “FlexEdge: Digital twin-enabled task offloading for UAV-aided vehicular edge computing,” IEEE Trans. Veh. Technol., vol. 72, no. 8, pp. 11086–11091, Aug. 2023.

[28] O. S. Oubbati, N. Chaib, A. Lakas, P. Lorenz, and A. Rachedi, “UAVassisted supporting services connectivity in urban VANETs,” IEEE Trans. Veh. Technol., vol. 68, no. 4, pp. 3944–3951, Apr. 2019.

[29] R. Han, Y. Wen, L. Bai, J. Liu, and J. Choi, “Age of information aware UAV deployment for intelligent transportation systems,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 3, pp. 2705–2715, Mar. 2022.

[30] L. Bai, J. Liu, J. Wang, R. Han, and J. Choi, “Data aggregation in UAVaided random access for Internet of Vehicles,” IEEE Internet Things J., vol. 9, no. 8, pp. 5755–5764, Apr. 2022.

[31] B. Hazarika and K. Singh, “AFL-DMAAC: Integrated resource management and cooperative caching for URLLC-IoV networks,” IEEE Trans. Intell. Vehicles, vol. 9, no. 6, pp. 5101–5117, Jun. 2024.

[32] R. Ding, J. Chen, W. Wu, J. Liu, F. Gao, and X. Shen, “Packet routing in dynamic multi-hop UAV relay network: A multi-agent learning approach,” IEEE Trans. Veh. Technol., vol. 71, no. 9, pp. 10059–10072, Sep. 2022.

[33] H. Peng and X. Shen, “Multi-agent reinforcement learning based resource management in MEC{-} and UAV-assisted vehicular networks,” IEEE J. Sel. Areas Commun., vol. 39, no. 1, pp. 131–141, Jan. 2021.

[34] Y. Liang, H. Wu, and H. Wang, “ASM-PPO: Asynchronous and scalable multi-agent PPO for cooperative charging,” in Proc. 21st Int. Conf. Auton. Agents Multiagent Syst., 2022, pp. 798–806.

[35] W. Liu, B. Li, W. Xie, Y. Dai, and Z. Fei, “Energy efficient computation offloading in aerial edge networks with multi-agent cooperation,” IEEE Trans. Wireless Commun., vol. 22, no. 9, pp. 5725–5739, Sep. 2023.

[36] L. Qin, H. Lu, Y. Chen, B. Chong, and F. Wu, “Toward decentralized task offloading and resource allocation in user-centric MEC,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 11807–11823, Dec. 2024.

[37] Y.-J. Chen, W. Chen, and M.-L. Ku, “Trajectory design and link selection in UAV-assisted hybrid satellite-terrestrial network,” IEEE Commun. Lett., vol. 26, no. 7, pp. 1643–1647, Jul. 2022.

[38] M.-M. Zhao, Q. Shi, and M.-J. Zhao, “Efficiency maximization for UAV-enabled mobile relaying systems with laser charging,” IEEE Trans. Wireless Commun., vol. 19, no. 5, pp. 3257–3272, May 2020.

[39] Q. Zhang, W. Fang, Q. Liu, J. Wu, P. Xia, and L. Yang, “Distributed laser charging: A wireless power transfer approach,” IEEE Internet Things J., vol. 5, no. 5, pp. 3853–3864, Oct. 2018.

[40] L. Bai, Z. Huang, L. Cui, and X. Cheng, “A non-stationary multi-UAV cooperative channel model for 6G massive MIMO mmWave communications,” IEEE Trans. Wireless Commun., vol. 22, no. 12, pp. 9233–9247, Dec. 2023.

[41] L. Zhang, A. Celik, S. Dang, and B. Shihada, “Energy-efficient trajectory optimization for UAV-assisted IoT networks,” IEEE Trans. Mobile Comput., vol. 21, no. 12, pp. 4323–4337, Dec. 2022.

[42] W. Nam, D. Bai, J. Lee, and I. Kang, “Advanced interference management for 5G cellular networks,” IEEE Commun. Mag., vol. 52, no. 5, pp. 52–60, May 2014.

[43] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP altitude for maximum coverage,” IEEE Wireless Commun. Lett., vol. 3, no. 6, pp. 569–572, Dec. 2014.

[44] X. Huang and X. Fu, “Fresh data collection for UAV-assisted IoT based on aerial collaborative relay,” IEEE Sensors J., vol. 23, no. 8, pp. 8810–8825, Apr. 2023.

[45] J. Lin, S. Huang, H. Zhang, X. Yang, and P. Zhao, “A deep reinforcement learning based computation offloading with mobile vehicles in vehicular edge computing,” IEEE Internet Things J., vol. 10, no. 17, pp. 15501–15514, Sep. 2023.

[46] G. Wang, F. Xu, and C. Zhao, “QoS-enabled resource allocation algorithm in Internet of Vehicles with mobile edge computing,” IET Commun., vol. 14, no. 14, pp. 2326–2333, Aug. 2020.

[47] R. K. Jain, D.-M. W. Chiu, and W. R. Hawe, “A quantitative measure of fairness and discrimination for resource allocation in shared computer systems,” Eastern Res. Lab., Digit. Equip. Corp., Hudson, MA, USA, Tech. Rep. DEC-TR-301, 1984, vol. 38.

[48] H. Peng and X. S. Shen, “DDPG-based resource management for MEC/UAV-assisted vehicular networks,” in Proc. IEEE 92nd Veh. Technol. Conf. (VTC-Fall), Nov. 2020, pp. 1–6.

[49] T. Zhang, K. Zhu, and J. Wang, “Energy-efficient mode selection and resource allocation for D2D-enabled heterogeneous networks: A deep reinforcement learning approach,” IEEE Trans. Wireless Commun., vol. 20, no. 2, pp. 1175–1187, Feb. 2021.

[50] G. He, S. Zhang, M. Feng, S. Li, and T. Jiang, “Age of incorrect information-aware data dissemination for distributed multiagent systems,” IEEE Trans. Wireless Commun., vol. 23, no. 10, pp. 15705–15718, Oct. 2024.

Authorized licensed use limited to: SUN YAT-SEN UNIVERSITY. Downloaded on May 19,2026 at 07:42:33 UTC from IEEE Xplore. Restrictions apply.

[51] B. Li, R. Yang, L. Liu, J. Wang, N. Zhang, and M. Dong, “Robust computation offloading and trajectory optimization for multi-UAV-assisted MEC: A multiagent DRL approach,” IEEE Internet Things J., vol. 11, no. 3, pp. 4775–4786, Feb. 2024.

[52] Q. Wang, S. Zou, Y. Sun, M. Liwang, X. Wang, and W. Ni, “Toward intelligent and adaptive task scheduling for 6G: An intent-driven framework,” IEEE Trans. Cognit. Commun. Netw., vol. 10, no. 5, pp. 1975–1988, Oct. 2024.

[53] H. Kang, X. Chang, J. Misiˇ c, V. B. Mi ´ siˇ c, J. Fan, and Y. Liu, ´ “Cooperative UAV resource allocation and task offloading in hierarchical aerial computing systems: A MAPPO based approach,” IEEE Internet Things J., vol. 10, no. 12, pp. 10497–10509, Jun. 2023.

[54] M. Cheng, C. Zhu, M. Lin, and W.-P. Zhu, “A MAPPO based scheme for joint resource allocation in UAV assisted MEC networks,” in Proc. IEEE/CIC Int. Conf. Commun. China (ICCC), Aug. 2024, pp. 42–47.

[55] C. Yu et al., “The surprising effectiveness of ppo in cooperative multiagent games,” in Proc. Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 24611–24624.

[56] P. Bohm, P. Pounds, and A. C. Chapman, “Feature extraction for ¨ effective and efficient deep reinforcement learning on real robotic platforms,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2023, pp. 7126–7132.

[57] A. Kaur, J. Thakur, M. Thakur, K. Kumar, A. Prakash, and R. Tripathi, “Deep recurrent reinforcement learning-based distributed dynamic spectrum access in multichannel wireless networks with imperfect feedback,” IEEE Trans. Cognit. Commun. Netw., vol. 9, no. 2, pp. 281–292, Apr. 2023.

[58] X. Ning, M. Zeng, M. Hua, and Z. Fei, “Multiple reconfigurable intelligent surfaces aided vehicular edge computing networks: A MAPPO-based approach,” IEEE Trans. Veh. Technol., vol. 73, no. 11, pp. 17496–17509, Nov. 2024.

[59] Z. Yu, Y. Gong, S. Gong, and Y. Guo, “Joint task offloading and resource allocation in UAV-enabled mobile edge computing,” IEEE Internet Things J., vol. 7, no. 4, pp. 3147–3159, Apr. 2020.

[60] H. Hellaoui, A. Chelli, M. Bagaa, and T. Taleb, “Towards mitigating the impact of UAVs on cellular communications,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), Dec. 2018, pp. 1–7.

[61] D. Zheng, L. Wang, C. Kai, and M. Peng, “Resource optimization for task offloading with real-time location prediction in pedestrian-vehicle interaction scenarios,” IEEE Trans. Wireless Commun., vol. 22, no. 11, pp. 7331–7344, Nov. 2023.

[62] S. Raza, S. Wang, M. Ahmed, M. R. Anwar, M. A. Mirza, and W. U. Khan, “Task offloading and resource allocation for IoV using 5G NR-V2X communication,” IEEE Internet Things J., vol. 9, no. 13, pp. 10397–10410, Jul. 2022.

[63] A. Hourani, S. Kandeepan, and A. Jamalipour, “Modeling air-to-ground path loss for low altitude platforms in urban environments,” in Proc. IEEE Global Telecommun. Conf. (GLOBECOM), Austin, TX, USA, Dec. 2014, pp. 2898–2904.

[64] N. Nouri, J. Abouei, A. R. Sepasian, M. Jaseemuddin, A. Anpalagan, and K. N. Plataniotis, “Three-dimensional multi-UAV placement and resource allocation for energy-efficient IoT communication,” IEEE Internet Things J., vol. 9, no. 3, pp. 2134–2152, Feb. 2022.

[65] W. Wang, X. Xu, M. Bilal, M. Khan, and Y. Xing, “UAV-assisted content caching for human-centric consumer applications in IoV,” IEEE Trans. Consum. Electron., vol. 70, no. 1, pp. 927–938, Feb. 2024.

[66] Z. Li, H. Zhang, X. Li, H. Ji, and V. C. M. Leung, “Distributed task scheduling for MEC-assisted virtual reality: A fully-cooperative multiagent perspective,” IEEE Trans. Veh. Technol., vol. 73, no. 7, pp. 10572–10586, Jul. 2024.

<!-- image-->

Tianjiao Du received the B.S. degree in Internet of Things from Central South University, Changsha, China, in 2019. She is currently pursuing the Ph.D. degree with Shaanxi Province Key Laboratory of Computer Network, Xi’an Jiaotong University, Xi’an, China.

Her current research interests include trusted computing, mobile edge computing, resource optimization, and information security of the Internet of Things.

<!-- image-->

Xiaolin Gui (Member, IEEE) received the Ph.D. degree in computer science from Xi’an Jiaotong University, Xi’an, China, in 2001.

community networks.

He has been a Professor and the Deputy Dean of the School of Electronic and Information, Xi’an Jiaotong University, since 2013. He has also been the Director of Shaanxi Province Key Laboratory of Computer Network, Xi’an, since 2008. His recent research covers high-performance computing, secure computation of open network systems, dynamic trust management theory, and development of

Dr. Gui was a recipient of the New Century Excellent Talents in Universities of China.

<!-- image-->

Tao Sheng was born in 1998. He received the B.S. degree and the M.S. degree from Huaibei Normal University, Huaibei, China, in 2020 and 2023, respectively. He is currently pursuing the Ph.D. degree with the School of Computer Science and Technology, Xi’an Jiaotong University, Xi’an, China.

His research areas include cloud edge collaboration, secure computing, and data aggregation.