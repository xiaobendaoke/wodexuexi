Remote ID based UAV collision avoidance optimization for low-altitude air‐ space safety

Ziye JIA, Yian ZHU, Qihui WU, Yao WU, Lei ZHANG, Sen YANG, Zhu HAN

PII: S1000-9361(25)00447-9

DOI: https://doi.org/10.1016/j.cja.2025.103841

Reference: CJA 103841

<!-- image-->

To appear in: Chinese Journal of Aeronautics

Revised Date: 16 September 2025

Accepted Date: 16 September 2025

Please cite this article as: Z. JIA, Y. ZHU, Q. WU, Y. WU, L. ZHANG, S. YANG, Z. HAN, Remote ID based UAV collision avoidance optimization for low-altitude airspace safety, Chinese Journal of Aeronautics (2025), doi: https://doi.org/10.1016/j.cja.2025.103841

This is a PDF file of an article that has undergone enhancements after acceptance, such as the addition of a cover page and metadata, and formatting for readability, but it is not yet the definitive version of record. This version will undergo additional copyediting, typesetting and review before it is published in its final form, but we are providing this version to give early visibility of the article. Please note that, during the production process, errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

© 2025 Published by Elsevier Ltd on behalf of Chinese Society of Aeronautics and Astronautics.

# Remote ID based UAV collision avoidance optimization for low-altitude airspace safety

Ziye JIA a, Yian ZHU a,∗, Qihui WU a, Yao WU a, Lei ZHANG a, Sen YANG b, Zhu HAN c

a Key Laboratory of Dynamic Cognitive System of Electromagnetic Spectrum Space, Ministry of Industry and Information Technology, Nanjing University of Aeronautics and Astronautics, Nanjing, 211106, China

b Institute of War Studies, Military Academy of Sciences, Beijing, 100080, China

c Department of Electrical and Computer Engineering, University of Houston, Houston, TX 77004, USA

## KEYWORDS

Unmanned Aerial Vehicle (UAV);   
Remote Identification   
(Remote ID);   
Bluetooth;   
Wireless Fidelity (Wi-Fi); Airspace safety;   
Collision Avoidance;   
Deep Reinforcement   
Learning (DRL).

Abstract With the rapid development of Unmanned Aerial Vehicles (UAVs), it is paramount to ensure safe and efficient operations in open airspaces. The Remote Identification (Remote ID) is deemed an effective real-time UAV monitoring system by the federal aviation administration, which holds potentials for enabling inter-UAV communications. This paper deeply investigates the application of Remote ID for UAV collision avoidance while minimizing communication delays. First, we propose a Remote ID based Distributed Multi-UAV Collision Avoidance (DMUCA) framework to support the collision detection, avoidance decision-making, and trajectory recovery. Then, the average transmission delays for Remote ID messages are analyzed, incorporating the packet reception mechanisms and packet loss due to interference. The optimization problem is formulated to minimize the long-term average communication delay, where UAVs can flexibly select the Remote ID protocol to enhance the collision avoidance performance. To tackle the problem, we design a multi-agent deep Q-network based adaptive communication configuration algorithm, allowing UAVs to autonomously learn the optimal protocol configurations in dynamic environments. Finally, numerical results verify the feasibility of the proposed DMUCA framework, and the proposed mechanism can reduce the average delay by 32% compared to the fixed protocol configuration.

## 1. Introduction

The rapid advancement of Unmanned Aerial Vehicles (UAVs) has driven significant economic growth and innovations1, 2, 3. In commercial applications, UAVs are increasingly utilized in daily operations due to their efficiency and flexibility, including tasks such as package delivery4, low-altitude monitoring5, and urban inspections 6. However, how to ensure the safe operation of UAVs, particularly in the densely populated urban airspace, is a critical challenge 7. In such complex environments, various factors such as signal interference, unpredictable flight paths and airspace congestion aggravate the collision risks of UAVs 8. The main factors of UAV collisions can be grouped into three key aspects. First, UAVs generally have limited situational awareness since they lack reliable state information from nearby UAVs, especially in environments without the direct sensing or centralized control. Besides, the communication delays and unstable data exchange make it hard for UAVs to respond in time when other UAVs are nearby. Moreover, many existing avoidance strategies are static and cannot adapt to fastchanging environments with multiple UAVs. These problems are more serious in the low-altitude urban airspace, where many UAVs from different operators fly close to each other without the coordination. Therefore, it is urgent to design a safe, reliable, and efficient collision avoidance model for the low-altitude airspace safety9, 10.

Several sensor-based approaches have been explored for UAV collision avoidance, with environmental awareness as a precondition. For example, the red-green-blue cameras are used in Ref.11 to detect dynamic obstacles in unknown tunnel environments. In Ref. 12, a Light Detection and Ranging (LiDAR)-based system collects obstacle data by capturing detailed Three-Dimensional (3D) spatial information of the environment, while Ref.13 integrates radar to provide distance and speed data for collision avoidance. However, in the open airspace scenarios with UAVs operated by various operators, these solutions are costly and unsuitable for large-scale implementation 14, 15.

To address the limitations of these sensor-based methods, the communication-based approaches such as Remote Identification (Remote ID) have gained attentions as a more scalable solution. In particular, Remote ID is a regulatory framework first introduced by the Federal Aviation Administration (FAA) and later adopted by aviation authorities worldwide 16. Its purpose is to enable the real-time identification and tracking of UAVs during flights. The concept is straightforward that UAVs must transmit essential flight information, such as the identity, position, velocity, and timestamp. The standards of Remote ID define two main types 17. The first is the broadcast Remote ID, where information is sent using local radio links, typically Bluetooth Low Energy (BLE) or Wireless Fidelity (Wi-Fi). Another is the network Remote ID, which leverages the Internet connectivity through cellular networks. We focus on the broadcast Remote ID in this paper since it is better suited for the environments without ground communication infrastructures and for the decentralized airspace monitoring.

Although wireless technologies such as long range radio can offer advantages such as long range and low power, the Remote ID standard from FAA focuses on the compatibility with common consumer devices. BLE and Wi-Fi are chosen since they allow most smartphones, tablets, and other personal devices to receive broadcast signals directly. This removes the need for extra hardwares and supports wider public access, making it easier for users to access Remote ID information. The public can then identify and locate these UAVs using mobile devices such as smartphones or ground stations 18. Such a broadcast-based approach also supports environmental sensing among UAVs with multiple advantages: (A) As a widely adopted surveillance mechanism19, Remote ID supports the large-scale integration of UAVs. (B) UAVs can share onboard sensor data to improve the state information accuracy20. (C) This method alleviates the need for direct UAV connections, reducing the communication cost and providing an effective alternative.

However, while Remote ID offers a promising communication solution for the UAV collision avoidance, its practical implementation faces challenges of communication delays. Many studies on collision avoidance algorithms regard fixed delays as communication parameters, without considering the delay in dynamic environments 21. However, in real-world scenarios, the communication delay is influenced by various factors. In particular, Remote ID operates over the frequency band used for Industrial, Scientific, and Medical (ISM) applications by BLE 4, BLE 5, and Wi-Fi to broadcast identification information22. Each protocol has different broadcasting mechanisms, transmission ranges, and data rates, corresponding to varying transmission delays. Moreover, the implementation of Remote ID does not specify a particular transmission protocol 18, leading to potential interferences among different protocols, which can further increase delays. Therefore, for the delay sensitive demands to avoid collisions, it is critical to make decisions across different communication protocols.

## 1.1. Related works

## 1.1.1. Multi-UAV collision avoidance

The collision avoidance for multi-UAVs is extensively investigated. For instance, in $\mathrm { R e f . } ^ { 2 3 }$ , an adaptive collision avoidance framework is proposed, by combining a Deep Reinforcement Learning (DRL) model and a conflict resolution pool to manage 3D pairwise conflicts with reduced computational complexity. The authors in Ref. 24 address the obstacle avoidance for fixed-wing UAVs using a curriculum based multi-agent DRL approach, which effectively learns collision avoidance strategies in cluttered environments. A realtime reactive collision avoidance algorithm based on low-resolution cameras in $\mathrm { R e f . } ^ { 2 5 }$ is developed, for the integration into low-cost UAVs without inter-robot communications. In Ref.26, the authors propose an energy-efficient cooperative collision avoidance scheme for UAV swarms, with each UAV detecting obstacles by LiDAR and sharing environmental data within the swarm, allowing all UAVs to process information and enhance collision avoidance.

While the above studies mainly rely on high-frequency sensors or centralized communication for situational awareness, there exist some methods use wireless signals for obstacle detection and airspace monitoring. For example, Ref.27 applies the automatic dependent surveillance-broadcast system, a standard wireless technology in civil aviation, to detect the nearby aircrafts and avoid collisions. The authors divide the airspace into smaller sub-regions and apply a PSO-RRT planning algorithm to improve the safety and efficiency of UAV trajectories in low-altitude environments. Ref. 28 develops a Remote ID based UAV traffic management monitoring system with a new teardrop-shaped detection area. The proposed algorithm improves the awareness of flight direction and increases the safe distance between UAVs. Ref.21 studies the use of Wi-Fi messaging for the UAV traffic management. The authors present a unified framework for UAV separation and experiments show that the Wi-Fi communication can effectively support safe UAV flights. However, most existing methods based on wireless signals do not analyze the communication protocols in detail. They also fail to consider how communication delay may affect the time-critical tasks.

## 1.1.2. Remote ID performance and analysis

The authors in Ref. 18 analyze the impact of packet loss rate in Remote ID on maintaining safe separation between UAVs. By quantifying the packet loss rate, the study evaluates the transmission reliability of Remote ID messages. Ref. 29 provides a mathematical modeling and simulation-based analysis of Remote ID on unmanned aerial systems to enhance the airspace safety, focusing on the impact of Remote ID broadcast range, signal latency, and UAV deceleration capabilities on safe operational limitations. Some studies explore the communication protocols related to Remote ID applications. For instance, Ref. 30 evaluates the advanced neighbor discovery process performance of BLE 5.0 in terms of signal collisions, discovery delays, and energy consumption. Ref. 3 proposes a BLE frequency hopping scheme that avoids collisions with Wi-Fi beacons by scheduling transmissions in the time domain, achieving better performance in access point-dense environments. The authors of Ref. 32 evaluate the Wi-Fi communication performance between UAVs, considering the interference from remote controllers, and test the packet loss rates with varying distances and interference.

Most existing studies on Remote ID rarely focus on modeling specific communication protocols and lack detailed analyses of communication delays in BLE 4, BLE 5, and Wi-Fi. To the best of the authors’ knowledge, the existing researches on Remote ID mainly focus on the system compatibility, message formats, and general transmission performance. In contrast, the adaptive protocol selection or the use of protocol-level information to support real-time control in UAV operations remain underexplored.

## 1.1.3. DRL-based optimization in UAV communication

The DRL algorithms are widely used to optimize the communication performance of UAV networks 33, 34, 35. In $\mathrm { R e f . } ^ { 3 6 }$ , the authors formulate a Markov Decision Process (MDP) model to address latency issues in a two-layer UAV-enabled mobile edge computing network, and employ Deep Q-Networks (DQNs) for discrete decision-making and Deep Deterministic Policy Gradient (DDPG) for continuous spaces. In Ref. 37, an RF signal detector based on RF fingerprinting and machine learning is proposed to identify UAV controller signals under Wi-Fi and bluetooth interferences. The authors in $\mathrm { R e f . } ^ { 3 8 }$ explore a hybrid UAV-assisted communication network by integrating BLE, long term evolution, Wi-Fi, and long range technologies, and design a reinforcement learning-based algorithm to optimize the link allocation and energy consumption. In Ref. 39, the authors propose a Multi-Agent Deep Deterministic Policy Gradient (MADDPG)-based approach to jointly optimize the UAV trajectory, bandwidth allocation, and ground user access control. Ref. 40 proposes a hybrid communication architecture for V2X applications that selects the best combination of radio access technologies using a double deep Q-learning algorithm. This approach improves reliability and reduces the channel load. $\mathrm { R e f . } ^ { 4 1 }$ considers a UAV-assisted covert edge computing framework, where DRL is used to jointly optimize the transmission power, task offloading ratios, and UAV altitude to minimize the task delay under covertness constraints.

Although many studies apply DRL in UAV communication networks, few have focused on reducing delays in the Remote ID communication systems. Most works focus on the general resource allocation or edge computing tasks. In contrast, this work focuses on adapting and managing multiple Remote ID communication protocols to reduce communication delays and enhance collision avoidances in low-altitude UAV operations.

## 1.2. Contributions

Inspired by the aforementioned analyses, we explore Remote ID to enhance the UAV situational awareness for the distributed collision avoidance. To achieve this, we propose a Remote ID based Distributed Multi-UAV Collision Avoidance (DMUCA) framework, where Remote ID enables key state information exchange between $\mathrm { U A V s } .$ The effectiveness of the framework relies on the timeliness and accuracy of information sharing, which are essential for collision avoidance performance. Therefore, we conduct detailed analyses of Remote ID based communication protocols to evaluate the effects of transmission delays. Then, we propose a DRL-based approach to optimize the transmission delay to improve the collision avoidance performance. The key contributions of this work are summarized as follows.

(1) We design a DMUCA framework based on Remote ID, in which UAVs obtain the environmental awareness and independently execute collision avoidance without centralized control.

(2) We model and analyze the transmission delay of Remote ID messages for three communication protocols, i.e., BLE 4, BLE 5, and Wi-Fi. The model explicitly considers the interference caused by the protocol coexistence in overlapping frequency bands and the impact of message transmission rates on delay. Then, we formulate the optimization problem to minimize the long-term average delay and enhance the collision avoidance performance.

(3) We propose a DRL algorithm, termed as MADQN-based Adaptive Transmission Mode Configuration (MADQN-ATMC), to tackle the formulated problem. The algorithm empowers UAVs to autonomously determine communication strategies, including selecting communication protocols and adjusting message transmission rates.

(4) Simulation results verify the feasibility of the DMUCA framework and highlight the critical role of delay optimization in collision avoidance. The results also validate the effectiveness of the proposed MADQN-ATMC algorithm for delay optimization in dynamic environments.

The rest of the paper is organized as follows. Section 2 proposes the DMUCA framework. Section 3 presents the system model and problem formulation. Section 4 designs the MADQN-ATMC algorithm. Numerical results are provided in Section 5, and finally conclusions and future perspectives are drawn in Section 6.

## 2. DMCUA Framework

We design the DMUCA framework, as shown in Fig. 1, which operates in three phases, i.e., observe, orient, and decide, detailed as follows.

(1) Observe: UAVs broadcast their identities, positions, and velocities using Remote ID. The position and velocity are determined via the onboard Global Navigation Satellite System (GNSS), enabling each UAV to independently acquire situational awareness in a decentralized mode. UAVs continuously transmit Remote ID data using the protocols of BLE 4, BLE 5, or Wi-Fi. During this phase, UAVs configure their Remote ID communication protocols, message transmission rates. In addition, the transmission rate, defined as the number of Remote ID messages sent per GNSS sampling cycle, affects the situational update frequency, while the transmission power affects the communication range.

(2) Orient: As shown in Fig. 1, the situational awareness may be affected by information propagation delay. For instance, during the collision avoidance, a UAV (e.g., UAV A) updates its environment information periodically. If UAV A receives Remote ID messages from UAVs B and C, the delay $\Delta t _ { \mathrm { D e l a y } }$ is composed of the system delay from GNSS data sampling to the successful reception of the packet, and the packet collision delay caused by simultaneous transmissions. To mitigate delays and improve the decision-making accuracy, a trajectory prediction mechanism estimates the future positions of nearby UAVs. $\Delta t _ { \mathrm { D e l a y } }$ is defined as $\Delta t _ { \mathrm { D e l a y } } = t _ { \mathrm { c u r r e n t } } - t _ { \mathrm { u p d a t e } } ,$ where $t _ { \mathrm { c u r r e n t } }$ is the current time, and ?? is the timestamp of the last update. Under the assumption of constant velocity, the position and velocity of the UAV are predicted as $p _ { \mathrm { c u r r e n t } } = p _ { \mathrm { u p d a t e } } + \Delta t _ { \mathrm { D e l a y } } \nu _ { \mathrm { u p d a t e } } ,$ and $\nu _ { \mathrm { c u r r e n t } } = \nu _ { \mathrm { u p d a t e } } ,$ , where ??update and ??update represent the last known position and velocity of the UAV, respectively.

(3) Decide: As shown in Fig. 1, UAVs can calculate collisionfree velocities by the Optimal Reciprocal Collision Avoidance (ORCA) method42. To guarantee alignments with the intended flight path post-avoidance, a path recovery mechanism based on ORCA is designed. In particular, a set of UAVs indexed as $i = \{ 1 , 2 , \dots , N \}$ , transmits Remote ID messages within the communication range of a UAV A. The relative velocity obstacle for UAV A to avoid collisions with $\mathrm { U A V } i \in \{ 1 , 2 , \dotsc , N \}$ within a time window ?? is

$$
\begin{array} { r } { \operatorname { V O } _ { A \mid i } ^ { \tau } = \left\{ \nu _ { A \mid i } ~ \mid ~ \exists t \in [ 0 , \tau ] , \left. \nu _ { A \mid i } t - ( p _ { i } - p _ { A } ) \right. \leq r _ { A } + r _ { i } \right\} } \end{array}\tag{1}
$$

<!-- image-->  
Fig. 1 DMUCA framework utilizing Remote ID communication.

where $\nu _ { A | i }$ is the relative velocity of UAV ?? with respect to UAV A, $p _ { A }$ and $\pmb { p } _ { i }$ are the positions of UAV A and ??, respectively. $r _ { A }$ and $r _ { i }$ are their respective radii. If $\pmb { \nu } _ { A | i } = \pmb { \nu } _ { A } - \pmb { \nu } _ { i }$ falls within $\mathrm { V O } _ { A \mid i } ^ { \tau } , \mathsf { a }$ collision is predicted, necessitating a velocity adjustment $u _ { A | i } , \mathrm { i . e . , }$

$$
{ \pmb u } _ { A | i } = \left( \underset { { \pmb v } \in \partial \mathrm { V O } _ { A | i } ^ { \tau } } { \mathrm { a r g m i n } } ~ \left\| { \pmb \nu } - { \pmb \nu } _ { A | i } \right\| \right) - { \pmb \nu } _ { A | i }\tag{2}
$$

Here, ${ \pmb { u } } _ { A | i }$ is the vector from $\nu _ { A | i }$ to the nearest point on the boundary of $\mathrm { v } \dot { 0 } _ { A \vert i } ^ { \tau }$ . Therefore, the collision-free velocity half-space defines the permissible velocities for UAV A to avoid collisions with UAV ?? is

$$
\operatorname { O R C A } _ { A | i } ^ { \tau } = \left\{ \nu \mid \left( \nu - \left( \nu _ { A } ^ { \mathrm { O p t } } + 0 . 5 u _ { A | i } \right) \right) { \pmb \mu } \geq 0 \right\}\tag{3}
$$

where ?? is the outward normal at $\pmb { \nu } _ { A | i } + \pmb { u } _ { A | i }$ within $\mathrm { v o } _ { A \mid i } ^ { \tau }$ . For all $\mathrm { U A V s } ,$ the collision-free velocity space ORCA???? for UAV A is the intersection of relevant half-spaces, denoted as

$$
\mathrm { O R C A } _ { A } ^ { \tau } = C \left( 0 , \nu _ { A } ^ { \mathrm { m a x } } \right) \cap \bigcap _ { i = 1 } ^ { N } \mathrm { O R C A } _ { A | i } ^ { \tau }\tag{4}
$$

where $C ( 0 , \nu _ { A } ^ { \mathrm { m a x } } )$ indicates a closed ball of radius $\nu _ { A } ^ { \mathrm { m a x } }$ centered at the origin, representing the maximum permissible velocity for UAV A. Then, the optimal collision-free velocity $\nu _ { A } ^ { \mathrm { { O p t } } }$ is determined by minimizing the Euclidean distance between the optional velocity from the $\mathrm { O R C A } _ { A } ^ { \tau }$ and the current velocity $\nu _ { A } .$ , i.e.,

$$
\boldsymbol { \nu } _ { A } ^ { \mathrm { O p t } } = \operatorname * { a r g m i n } _ { \boldsymbol { \nu } \in \mathrm { O R C A } _ { A } ^ { \tau } } \| \boldsymbol { \nu } - \boldsymbol { \nu } _ { A } \|\tag{5}
$$

If no collision risks $( \mathrm { i . e . , } \nu _ { A | i } \notin \bigcup _ { i = 1 } ^ { N } \mathrm { V O } _ { A | i } ^ { \tau } )$ last for ??noncollide cycles of period $T _ { \mathrm { O R C A } } .$ , UAV A proceeds to path recovery by minimizing deviations from the predefined trajectory $p _ { \mathrm { t r a j } } ( t )$ with velocity constraints, where $n _ { \mathrm { n o n c o l l i d e } }$ denotes the number of consecutive cycles without detected collision risks. Based on the current position $p _ { A }$ and nearest trajectory point $p _ { \mathrm { t r a j } } ( t _ { \mathrm { n e a r e s t } } )$ , UAV A adjusts its velocity in two cases, i.e.,

(1) If ??nearest > ??current, the UAV computes the velocity aimed at reaching ??nearest by ??nearest:

$$
\nu _ { A } ^ { \mathrm { d e s i r e d } } = \frac { p _ { \mathrm { t r a j } } ( t _ { \mathrm { n e a r e s t } } ) - p _ { A } } { t _ { \mathrm { n e a r e s t } } - t _ { \mathrm { c u r r e n t } } }\tag{6}
$$

(2) If $t _ { \mathrm { n e a r e s t } } ~ \leq$ ??current, the UAV targets the trajectory point corresponding to ??current:

$$
\nu _ { A } ^ { \mathrm { d e s i r e d } } = \frac { { \pmb p } _ { \mathrm { t r a j } } ( t _ { \mathrm { c u r r e n t } } ) - { \pmb p } _ { A } } { T }\tag{7}
$$

where ?? represents a predefined interval allocated for the path recovery.

The desired velocity is constrained by $\nu _ { A } ^ { \mathrm { m a x } }$ , yielding the final adjusted velocity as

$$
\nu _ { A } ^ { \mathrm { O p t } } = \operatorname* { m i n } \left( \left\| \nu _ { A } ^ { \mathrm { d e s i r e d } } \right\| , \nu _ { A } ^ { \mathrm { m a x } } \right) \frac { \nu _ { A } ^ { \mathrm { d e s i r e d } } } { \left\| \nu _ { A } ^ { \mathrm { d e s i r e d } } \right\| }\tag{8}
$$

Here, ∥·∥ denotes the Euclidean norm of a vector. $\operatorname { I f } \left\| \nu _ { A } ^ { \mathrm { d e s i r e d } } \right\|$ exceeds $\nu _ { A \_ A } ^ { \mathrm { m a x } }$ , the expected arrival time and trajectory of UAV A are dynamically updated.

## 3. System model and problem formulation

In this section, the packet reception model is firstly established to analyze the system delays of BLE 4, BLE 5, and Wi-Fi. Then, the packet collision model is proposed to quantify the interference effects, followed by an average transmission delay model to evaluate the Remote ID delays. Specifically, we conduct the delay modeling at the medium access control layer for protocol mechanisms. It is assumed that the UAV is equipped with standard communication modules. To facilitate understanding, the key notations for modeling of Remote ID communication protocols are summarized in Table 1. Moreover, the optimization problem is formulated to minimize the long-term average transmission delay of Remote ID messages.

## 3.1. Packet reception model

We leverage a discrete-time slot model to evaluate the packet reception delay performance of the three transmission protocols of BLE 4, BLE 5, and Wi-Fi for Remote ID. A time duration is divided into multiple slots with length of Δ. TX (Transmitter) represents the device broadcasting Remote ID packets, and RX (Receiver) refers to the device capturing these packets by scanning the broadcast channels.

$$
3 . I . I . \ B L E { \ 4 }
$$

BLE operates in the 2.4 GHz ISM band (2 400 MHz to 2 483.5 $\mathrm { M H z } ) ^ { \dot { 4 } 3 }$ , which is divided into 40 channels. BLE 4 uses channels

Remote ID based UAV collision avoidance optimization for low-altitude airspace safety

Table 1 Key notations for Remote ID communication protocol.
<table><tr><td>Type</td><td>Parameter Definition</td><td></td></tr><tr><td>General</td><td>△</td><td>Discrete time slot length.</td></tr><tr><td></td><td> $t _ { 0 }$ </td><td>Start time of an advertisement or beacon event.</td></tr><tr><td></td><td> $T _ { \mathrm { G N S S } }$ </td><td>GNSS synchronization period.</td></tr><tr><td>BLE</td><td> $A _ { P }$ </td><td>Transmission duration of a single PDU packet transmitted on the BLE primary</td></tr><tr><td></td><td> $P _ { I }$ </td><td>channel. Interval between successive PDU trans- missions.</td></tr><tr><td></td><td> $A _ { I }$ </td><td>Interval between consecutive advertise- ment events.</td></tr><tr><td></td><td> $\hat { A } _ { I }$ </td><td>Approximatedadvertisement interval ·</td></tr><tr><td></td><td> $S _ { I }$ </td><td>satisfying gcd  $\ \hat { ( A _ { I } , 3 S _ { I } ) } = 1$  BLE scan interval.</td></tr><tr><td></td><td> $S _ { W }$ </td><td>BLE scan window duration per channel.</td></tr><tr><td></td><td> $A _ { \mathrm { O f f s e t } }$ </td><td>Time offset from BLE 5 pointer PDU to auxiliarypacket.</td></tr><tr><td></td><td> $T _ { \mathrm { A U X } }$ </td><td>Transmission duration of BLE 5 auxil- iary packet.</td></tr><tr><td></td><td> $O _ { U }$ </td><td>Timing uncertainty in BLE 5 auxiliary packet scheduling.</td></tr><tr><td></td><td> $\psi _ { \mathrm { B L E } 4 }$ </td><td>Message transmission rate for BLE 4.</td></tr><tr><td></td><td> $\psi _ { \mathrm { B L E } } \varsigma$ </td><td>Message transmission rate for BLE 5.</td></tr><tr><td>Wi-Fi</td><td> $B _ { D }$ </td><td>Duration of beacon packet transmission.</td></tr><tr><td></td><td> $B _ { I }$ </td><td>Interval between successive beacon packets.</td></tr><tr><td></td><td> ${ \hat { B } } _ { I }$ </td><td>Approximated beacon interval satisfying</td></tr><tr><td></td><td> $T _ { S }$ </td><td> $\mathrm { g c d } ( \hat { B } _ { I } , 3 ( T _ { S } + C _ { T } ) ) = 1$  Scan time per channel.</td></tr><tr><td></td><td> $C _ { T }$ </td><td>Channel switching time.</td></tr><tr><td></td><td></td><td>Time slot index range for channel c.</td></tr><tr><td></td><td> $I _ { c }$   $\psi _ { \mathrm { W i - F i } }$ </td><td>Message transmission rate for Wi-Fi.</td></tr></table>

37, 38, and 39 for broadcasting. As illustrated in Fig. 2(a), a BLE 4 TX periodically generates an advertisement event (Adv_Event), which represents a Remote ID message. Each Adv_Event consists of three identical Protocol Data Units (PDUs), transmitted sequentially on channels 37, 38, and 39. Each PDU is regarded as a packet carrying the actual Remote ID payload, which consists of the data representing Remote ID information, such as identity, position, and velocity. The transmission duration for each packet is denoted as $A _ { P } ,$ with an interval $P _ { I }$ between successive transmissions. The interval between consecutive Adv_Events is $A _ { I } ,$ , which includes a pseudo-random delay $R _ { D }$ . RX periodically scans channels 37, 38, and 39, with a scanning duration of $S _ { W }$ per channel and a scanning cycle of $S _ { I }$ . All time-related parameters are discretized as $\begin{array} { r } { X _ { d } = \left\lfloor \frac { X } { \Delta } \right\rfloor } \end{array}$ , where ⌊·⌋ is the floor function.

Let the scanning process begin at $t = 0 \mathbf { s } .$ , and with the Adv_Event starting at ??0 $\left( t _ { 0 } \geq 0 \mathrm { s } \right)$ . The reception is successful if RX receives at least one PDU packet during a scanning cycle. For example, in Fig. 2(a), TX transmits a PDU packet on channel 37 starting at ??0. The reception is successful if the broadcast interval $\left[ t _ { 0 } , t _ { 0 } + A _ { P } \right]$ fully overlaps with RX scan window [0, ????].

Considering the periodic transmission and scanning intervals, we aim to determine whether RX can match the packets within the Adv_Event on the corresponding channels. It is achieved by the

<!-- image-->  
(a) BLE 4 operational mode

<!-- image-->  
Fig. 2 Operational modes of BLE 4, BLE 5, and Wi-Fi communication protocols.

Chinese Remainder Theorem (CRT) T $( t _ { 1 } , t _ { 2 } , S _ { 1 } , S _ { 2 } )$ , a mathematical tool for modular arithmetic equations that identifies the time slots when two periodic events coincide 44, calculated as

$$
\mathcal { T } \left( t _ { 1 } , t _ { 2 } , S _ { 1 } , S _ { 2 } \right) = \left\{ \Delta _ { o } + k S _ { 1 } S _ { 2 } \ | \ k \in \mathbb { N } \right\}
$$

where

$$
\Delta _ { o } = \left( t _ { 1 } S _ { 2 } \left[ S _ { 2 } ^ { - 1 } \right] _ { S _ { 1 } } + t _ { 2 } S _ { 1 } \left[ S _ { 1 } ^ { - 1 } \right] _ { S _ { 2 } } \right) \bmod ( S _ { 1 } S _ { 2 } )\tag{9}
$$

in which $t _ { 1 }$ and $t _ { 2 }$ are the start times of two periodic events, $S _ { 1 }$ and $s _ { 2 }$ are the event periods, which must be relatively prime positive integers. $[ \cdot ] _ { . s } ^ { - 1 }$ is the modular multiplicative inverse of S. $\mathcal { T } ( \cdot )$ yields the matching time slots between the events, corresponding to periodic operations such as broadcasting or scanning in communication protocols. Based on Eq. (9), the set of successful matching time slots of the PDU packets on channel 37 during a single GNSS update cycle is derived as

$$
\begin{array} { r l } & { M _ { \mathrm { B L E } 4 } ^ { 3 7 } ( t _ { 0 } ) = \{ t \mid t \in \mathcal { T } ( t _ { 0 } , i , \hat { A } _ { I } , 3 S _ { I } ) , } \\ & { ~ \quad \quad 0 \leq i \leq S _ { W } - A _ { P } , t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \} } \end{array}\tag{10}
$$

Specifically, $\begin{array} { r } { A _ { I } = \frac { T _ { \mathrm { G N S S } } } { \psi _ { \mathrm { B L E } 4 } } } \end{array}$ , where ??GNSS is the GNSS update period, $\psi _ { \mathrm { B L E } 4 }$ is the Remote ID message transmission rate. $\hat { A } _ { I }$ is defined as an approximation of $A _ { I }$ that is relatively prime to $3 S _ { I }$

Similarly, the matching time slots for channels 38 and 39 are

respectively

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { B L E } 4 } ^ { 3 8 } ( t _ { 0 } ) = \left\{ t \mid t \in \mathcal { T } \left( t _ { 0 } + A _ { P } + P _ { I } , i , \hat { A } _ { I } , 3 S _ { I } \right) , \right. \qquad } \\ { \left. S _ { I } \leq i \leq S _ { I } + S _ { W } - A _ { P } , t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \right\} } \end{array}\tag{11}
$$

and

$$
\begin{array} { r l } & { \mathcal { M } _ { \mathrm { B L E } 4 } ^ { 3 9 } ( t _ { 0 } ) = \left\{ t \mid t \in \mathcal { T } \left( t _ { 0 } + 2 ( A _ { P } + P _ { I } ) , i , \hat { A } _ { I } , 3 S _ { I } \right) , \right. } \\ & { ~ \left. 2 S _ { I } \leq i \leq 2 S _ { I } + S _ { W } - A _ { P } , t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \right\} } \end{array}\tag{12}
$$

The set of matching time slots for all successfully received PDU packets within a single GNSS update cycle is

$$
N _ { \mathrm { B L E } 4 } ( t _ { 0 } ) = \left\{ \mathcal { M } _ { \mathrm { B L E } 4 } ^ { 3 7 } ( t _ { 0 } ) , \mathcal { M } _ { \mathrm { B L E } 4 } ^ { 3 8 } ( t _ { 0 } ) , \mathcal { M } _ { \mathrm { B L E } 4 } ^ { 3 9 } ( t _ { 0 } ) \right\}\tag{13}
$$

Hence, the reception delay for each successfully received PDU packet is

$$
D _ { \mathrm { B L E } 4 } ( \delta ) = \delta - t _ { 0 } + A _ { P } , \quad \delta \in N _ { \mathrm { B L E } 4 } ( t _ { 0 } )\tag{14}
$$

## 3.1.2. BLE 5

As depicted in Fig. 2(b), BLE 5 employs secondary channels 0-36 for message broadcasting, while primary channels 37, 38, and 39 transmit pointers to secondary channels for Remote ID messages. TX sends ADV_EXT_IND events, each comprising three pointer PDUs carrying Aux_Ptr pointers to the same auxiliary packet. The auxiliary packet contains the Remote ID payload and is transmitted via a secondary channel determined by a channel hopping algorithm 45

The interval between the start of the ADV_EXT_IND event and the auxiliary packet is denoted as $A _ { \mathrm { O f f s e t } } ,$ while $T _ { \mathrm { A U X } }$ represents the transmission time of the auxiliary packet. The predefined time interval $O _ { U }$ ensures the auxiliary packets are transmitted between $t _ { 0 } + A _ { \mathrm { O f f s e t } }$ and $t _ { 0 } + A _ { \mathrm { O f f s e t } } + O _ { U }$ , to mitigate the transmission delays or timing uncertainties 43.

The matching time slots for pointer PDU packets on primary channels 37, 38, and 39 of BLE 5 are similar to BLE 4. The interval of ADV_EXT_IND event is defined as $\begin{array} { r } { A _ { I } = \frac { T _ { \mathrm { G N S S } } } { \psi _ { \mathrm { B L E } } _ { 5 } } } \end{array}$ , where ??BLE 5 denotes the message transmission rate. The matching time slots of the packets are expressed as

$$
\begin{array} { r l } & { M _ { \mathrm { B L E } 5 } ^ { 3 7 } ( t _ { 0 } ) = \{ t \mid t \in \mathcal { T } ( t _ { 0 } , i , \hat { A } _ { I } , 3 S _ { I } ) , } \\ & { \quad \quad \quad 0 \leq i \leq S _ { W } - A _ { P } , t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \} } \end{array}\tag{15}
$$

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { B L E } 5 } ^ { 3 8 } ( t _ { 0 } ) = \big \{ t \mid t \in \mathcal { T } ( t _ { 0 } + A _ { P } + P _ { I } , i , \hat { A } _ { I } , 3 S _ { I } ) , } \\ { S _ { I } \leq i \leq S _ { I } + S _ { W } - A _ { P } , \ t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \big \} } \end{array}\tag{16}
$$

and

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { B L E } 5 } ^ { 3 9 } ( t _ { 0 } ) = \left\{ t \mid t \in \mathcal { T } ( t _ { 0 } + 2 ( A _ { P } + P _ { I } ) , i , \hat { A } _ { I } , 3 S _ { I } ) , \right. } \\ { \left. 2 S _ { I } \leq i \leq 2 S _ { I } + S _ { W } - A _ { P } , \ t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \right\} } \end{array}\tag{17}
$$

The set of matching time slots for BLE 5 across all primary channels is defined as

$$
N _ { \mathrm { B L E } } \varsigma ( t _ { 0 } ) = \left\{ \mathcal { M } _ { \mathrm { B L E } } ^ { 3 7 } \varsigma ( t _ { 0 } ) , \mathcal { M } _ { \mathrm { B L E } \varsigma } ^ { 3 8 } ( t _ { 0 } ) , \mathcal { M } _ { \mathrm { B L E } \varsigma } ^ { 3 9 } ( t _ { 0 } ) \right\}\tag{18}
$$

If $\delta _ { \xi } \in \mathcal { N } _ { \mathrm { B L E } } \varsigma \left( t _ { 0 } \right)$ satisfies $t _ { 0 } + ( \xi - 1 ) A _ { I } \ \le \ \delta _ { \xi } \ < \ t _ { 0 } + \xi A _ { I } ,$ the ??-th auxiliary packet is successfully received, with its reception delay of

$$
D _ { \mathrm { B L E } 5 } ( \xi ) = ( \xi - 1 ) A _ { I } + A _ { \mathrm { O f f s e t } } + O _ { U } + T _ { \mathrm { A U X } }\tag{19}
$$

Conversely, if no such $\delta _ { \xi }$ exists in $N _ { \mathrm { B L E } 5 } ( t _ { 0 } )$ , it indicates that the ??-th auxiliary packet is not successfully received.

## 3.1.3. Wi-Fi

In Remote ID, Wi-Fi utilizes 802.11 beacon frames to broadcast messages 46. Operating within [2.402, 2.472] GHz, Wi-Fi is divided into 11 channels, and each having a 20 MHz bandwidth. To minimize the adjacent channel interference, channels 1, 6, and 11 are commonly employed in practice47

The Wi-Fi broadcast mechanism, depicted in Fig. 2(c), involves UAVs transmitting on a designated fixed channel $( \mathrm { e . g . }$ , channel 6 in this case). The beacon packet, carrying the Remote ID payload, is transmitted over a duration of $B _ { D }$ . Given the message transmission rate $\psi _ { \mathrm { { W i - F i } } }$ , the interval between consecutive beacons is $\begin{array} { r } { B _ { I } = \frac { T _ { \mathrm { G N S S } } } { \psi _ { \mathrm { W i - F i } } } } \end{array}$

RX employs passive scanning, listening sequentially on channels 1, 6, and 11 for a duration of $T _ { S }$ per channel before switching, with a channel switching time of $C _ { T } .$ At time $t _ { 0 } ,$ the UAV updates its GNSS data and begins beacon broadcasting, where ??0 ∈ $[ 0 , 3 ( T _ { S } + C _ { T } ) )$ . According to Eq. (9), the set of matching time slots for beacon packets on channel $c \in \{ 1 , 6 , 1 1 \}$ is

$$
\begin{array} { r l } & { M _ { \mathrm { W i - F i } } ^ { c } ( t _ { 0 } ) = \{ t  { \mid t \in \mathcal { T } ( t _ { 0 } , i , \hat { B } _ { I } , 3 ( T _ { S } + C _ { T } ) ) , i \in I _ { c } , } } \\ & { \phantom { \hat { = } } t _ { 0 } \leq t \leq t _ { 0 } + T _ { \mathrm { G N S S } } \} } \end{array}\tag{20}
$$

where ${ \hat { B } } _ { I }$ approximates $B _ { I }$ and is relatively prime to $3 ( T _ { S } + C _ { T } )$ The range $I _ { c }$ for time slots on channel ?? is defined as

$$
I _ { c } = \left\{ \begin{array} { l l } { 0 \leq i \leq T _ { S } - B _ { D } , } & { c = 1 } \\ { T _ { S } + C _ { T } \leq i \leq 2 T _ { S } + C _ { T } - B _ { D } , } & { c = 6 } \\ { 2 \left( T _ { S } + C _ { T } \right) \leq i \leq 3 T _ { S } + 2 C _ { T } - B _ { D } , } & { c = 1 1 } \end{array} \right.\tag{21}
$$

As such, the reception delay for a Wi-Fi beacon packet on channel ?? is

$$
D _ { \mathrm { W i \mathrm { - } F i } } ( \delta ) = \delta - t _ { 0 } + B _ { D } , \quad \delta \in { \cal M } _ { \mathrm { W i \mathrm { - } F i } } ^ { c } ( t _ { 0 } )\tag{22}
$$

## 3.2. Packet collision model

The protocol interference incorporates Self-Technology Interference (STI) and Cross-Technology Interference (CTI).

## 3.2.1. BLE 4 – STI and CTI

We define the set of ?? UAVs as $\mathcal { U } = \{ u _ { 1 } , u _ { 2 } , . . . , u _ { M } \}$ , and the available communication protocols as $\mathcal { E } = \{ \mathrm { B L E } 4 ,$ BLE 5, Wi-Fi}. To model the communication between UAVs, let $\mathcal { R } _ { \epsilon } ^ { u _ { i } }$ denote the set of $\mathrm { U A V s } u _ { j } \in \mathcal { U }$ capable of reaching UAV $u _ { i } \in \mathcal { U }$ through broadcast using protocol $\epsilon \in { \mathcal { E } }$ , specifically,

$$
\mathcal { R } _ { \epsilon } ^ { u _ { i } } = \left\{ u _ { j } \ | \ u _ { j } \in \mathcal { U } , \lambda _ { u _ { j } } ^ { \epsilon } \left( P _ { u _ { j } } ^ { \epsilon , \mathrm { t x } } - \mathrm { P L } _ { u _ { j } , u _ { i } } \right) \geq \Theta _ { \epsilon } \right\}\tag{23}
$$

where $\lambda _ { u _ { i } } ^ { \epsilon }$ is the binary variable that indicates whether UAV $u _ { j }$ selects protocol $\epsilon , P _ { u _ { i } } ^ { \epsilon , \mathrm { t x } }$ is the transmission power of UAV $u _ { j }$ with protocol $\epsilon , \mathrm { P L } _ { u _ { j } , u _ { i } }$ is the path loss between $\mathrm { U A V s } u _ { j }$ and $u _ { i } ,$ , and $\Theta _ { \epsilon }$ is the receiver sensitivity. Thus, the probability of UAV ???? ∈ U to successfully receive a PDU packet from UAV $u _ { j } \in \mathcal { R } _ { \mathrm { B L E } 4 } ^ { u _ { i } }$ without STI is

$$
P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E 4 , S T I } } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { B L E 4 } } ^ { u _ { i } } } \left( 1 - \frac { 2 A _ { P } ^ { \mathrm { B L E 4 } } \psi _ { \mathrm { B L E 4 } } ^ { u _ { k } } } { T _ { \mathrm { G N S S } } } \right)\tag{24}
$$

where $A _ { P } ^ { \mathrm { B L E } 4 }$ is the duration of the PDU packet transmission within an Adv_Event of BLE 4. $\psi _ { \mathrm { B L E } 4 } ^ { u _ { k } }$ represents the number of Remote ID messages transmitted by UAV $u _ { k }$ within a GNSS update cycle, and $u _ { k }$ refers to other UAVs in $\mathcal { R } _ { \mathrm { B L E } 4 } ^ { u _ { i } } .$

The operation of BLE 5 on channels 37, 38, and 39 brings CTI to BLE 4. The probability that $\mathrm { U A V } u _ { i } \in \mathcal { U }$ successfully receives a PDU packet from $u _ { j } \in \mathcal { R } _ { \mathrm { B L E } 4 } ^ { u _ { i } }$ with BLE 5 interference is

$$
P _ { \bar { c } , u _ { i } } ^ { { \mathrm { B L E } } 4 , { \mathrm { B L E } } 5 } = \prod _ { u _ { k } \in \mathcal { R } _ { { \mathrm { B L E } } 5 } ^ { u _ { i } } } \left( 1 - \frac { \left( A _ { P } ^ { { \mathrm { B L E } } 4 } + A _ { P } ^ { { \mathrm { B L E } } 5 } \right) \psi _ { { \mathrm { B L E } } 5 } ^ { u _ { k } } } { T _ { \mathrm { G N S } } } \right)\tag{25}
$$

where $\psi _ { \mathrm { B L E } 5 } ^ { u _ { k } }$ is the message transmission rate of UAV $u _ { k } .$ , and $A _ { P } ^ { \mathrm { B L E } 5 }$ is the BLE 5 Adv_Event packet duration.

## 3.2.2. BLE 5 – STI and CTI

The STI of BLE 5 is analyzed for primary and secondary channels. For primary channels 37, 38, and 39, the probability for $u _ { i } \in \mathcal { U }$ to receive a pointer PDU packet from UAV $u _ { j } \in \mathcal { R } _ { \mathrm { B L E } } ^ { u _ { i } } $ without collision is

$$
P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E } 5 , \mathrm { S T I , P r i } } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { B L E } 5 } ^ { u _ { i } } } \left( 1 - \frac { 2 A _ { P } ^ { \mathrm { B L E } 5 } \psi _ { \mathrm { B L E } 5 } ^ { u _ { k } } } { T _ { \mathrm { G N S S } } } \right)\tag{26}
$$

For secondary channels, BLE 5 uses a channel-hopping mechanism that selects one $\mathbf { o f } 3 7$ secondary channels with equal probability. The probability of successful reception of an auxiliary packet by UAV $u _ { i } \in \mathcal { U }$ from UAV $u _ { j } \in \mathcal { R } _ { \mathrm { B L E } } ^ { u _ { i } } $ is

$$
P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E ~ 5 , S T I , S e c } } = \prod _ { \scriptstyle u _ { k } \in \mathcal { R } _ { \mathrm { B L E } 5 } ^ { u _ { i } } } \left( 1 - \frac { 2 T _ { \mathrm { A U X } } \psi _ { \mathrm { B L E } 5 } ^ { u _ { k } } } { 3 7 T _ { \mathrm { G N S S } } } \right)\tag{27}
$$

The operation of BLE 4 on primary channels brings CTI. The probability of successful reception of a pointer PDU packet by UAV $u _ { i } \in \mathcal { U }$ from UAV $u _ { j } \in \mathcal { R } _ { \mathrm { B L E } } ^ { u _ { i } } 5$ is

$$
P _ { \bar { c } , u _ { i } } ^ { \mathrm { B L E } 5 , \mathrm { B L E } 4 , \mathrm { P r i } } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { B L E } 4 } ^ { u _ { i } } } \left( 1 - \frac { \left( A _ { P } ^ { \mathrm { B L E } 4 } + A _ { P } ^ { \mathrm { B L E } 5 } \right) \psi _ { \mathrm { B L E } 4 } ^ { u _ { k } } } { T _ { \mathrm { G N S S } } } \right) ( 2 8 )
$$

The Wi-Fi channels overlap with the secondary channels of BLE 5, causing CTI. The probability of UAV $u _ { i } \in \mathcal { U }$ successfully receiving an auxiliary packet from $\dot { \mathrm { U A V } } u _ { j } \in \mathcal { R } _ { \mathrm { B L E } 5 } ^ { u _ { i } }$ under Wi-Fi interference is

$$
P _ { \bar { c } , u _ { i } } ^ { { \mathrm { B L E } } ~ 5 , \mathrm { W i - F i } , \mathrm { S e c } } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { W i - F i } } ^ { u _ { i } } } \left( 1 - \frac { 9 \left( T _ { \mathrm { A U X } } + B _ { D } \right) \psi _ { \mathrm { W i - F i } } ^ { u _ { k } } } { 3 7 T _ { \mathrm { G N S S } } } \right)\tag{29}
$$

where $\psi _ { \mathrm { W i - F i } } ^ { u _ { k } }$ is the message transmission rate of UAV $u _ { k }$

## 3.2.3. Wi-Fi – STI and CTI

To model STI of Wi-Fi, UAVs within the communication range of $u _ { i } \in \mathcal { U }$ transmitting on distinct Wi-Fi channels $c \in \{ 1 , 6 , 1 1 \}$ are denoted as $u _ { j } \in \mathcal { R } _ { \mathrm { W i - F i } } ^ { u _ { i } , c }$ . The probability of successful reception of a Wi-Fi beacon packet by UAV $u _ { i }$ from $u _ { j }$ under STI is

$$
P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { W i - F i , S T I } } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { W i - F i } } ^ { u _ { i } , c } } \left( 1 - \frac { 2 B _ { D } \psi _ { \mathrm { W i - F i } } ^ { u _ { k } } } { T _ { \mathrm { G N S S } } } \right)\tag{30}
$$

The operation of BLE 5 on secondary channels may result in CTI with Wi-Fi. The probability of successful reception of a Wi-Fi beacon packet by $\mathrm { U } \bar { \mathbf { A } } \mathrm { V } u _ { i } \in \bar { \mathcal { U } }$ is

$$
P _ { \bar { c } , u _ { i } } ^ { \mathrm { W i - F i , B L E } 5 } = \prod _ { u _ { k } \in \mathcal { R } _ { \mathrm { B L E } 5 } ^ { u _ { i } } } \left( 1 - \frac { 9 \left( T _ { \mathrm { A U X } } + B _ { D } \right) \psi _ { \mathrm { B L E } 5 } ^ { u _ { k } } } { 3 7 T _ { \mathrm { G N S } 5 } } \right)\tag{31}
$$

## 3.3. Average transmission delay model

## 3.3.1. BLE 4

Let $S _ { \epsilon } ^ { u _ { j } }$ denote the set of $\mathrm { U A V s } u _ { i } \in \mathcal { U }$ that are capable of receiving messages from UAV $u _ { j } \in \mathcal { U }$ using protocol $\epsilon \in { \mathcal { E } } ,$ i.e.,

$$
\begin{array} { r } { S _ { \epsilon } ^ { u _ { j } } = \left\{ u _ { i } \ | \ u _ { i } \in \mathcal { U } , \lambda _ { u _ { j } } ^ { \epsilon } \left( P _ { u _ { j } } ^ { \epsilon , \mathrm { t x } } - \mathrm { P L } _ { u _ { j } , u _ { i } } \right) \ge \Theta _ { \epsilon } \right\} } \end{array}\tag{32}
$$

For UAV $u _ { j } ~ \in ~ \mathcal { U }$ broadcasting via BLE 4, the set of successfully received PDU packets within a GNSS update cycle at time $t _ { 0 }$ is $N _ { \mathrm { B L E } 4 } ^ { u _ { j } } \left( t _ { 0 } \right)$ in Eq. (13). Let $\delta _ { u _ { j } , n } ^ { \mathrm { B L E } 4 }$ represent the ??-th earliest packet reception time: $\delta _ { u _ { j } , n } ^ { \mathrm { B L E } 4 } = \operatorname* { m i n } \{ { N } _ { \mathrm { B L E } 4 } ^ { u _ { j } } ( t _ { 0 } ) \} _ { n }$ . Considering the packet reception delay of Eq. (14) and the collision-free probability of BLE 4 packets, the BLE 4 transmission delay from UAV $u _ { j }$ to a neighboring UAV $u _ { i } \in S _ { \mathrm { B L E } 4 } ^ { u _ { j } }$ is

$$
\begin{array} { r l } { d _ { \mathrm { B L E } 4 } ^ { u _ { j } , u _ { i } } \left( t _ { 0 } \right) = \displaystyle \sum _ { k = 0 } ^ { \infty } \sum _ { n = 1 } ^ { \left| N _ { \mathrm { B L E } 4 } ^ { u _ { j } } \left( t _ { 0 } \right) \right| } \left( 1 - P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } 4 } \right) ^ { n - 1 + \left| N _ { \mathrm { B L E } 4 } ^ { u _ { j } } \left( t _ { 0 } \right) \right| k } } & { } \\ { \cdot P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } 4 } \left( D \left( \delta _ { u _ { j } , n } ^ { \mathrm { B L E } 4 } \right) + k T _ { \mathrm { G N S S } } \right) } & { } \end{array}\tag{33}
$$

where

$$
P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } 4 } = P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E } 4 , \mathrm { S T I } } P _ { \bar { c } , u _ { i } } ^ { \mathrm { B L E } 4 , \mathrm { B L E } 5 }\tag{34}
$$

Due to the stochastic nature of message transmission, the average delay for a Remote ID message from $u _ { j }$ to $u _ { i } \in \mathring { S } _ { \mathrm { B L E } 4 } ^ { u _ { j } }$ is

$$
\hat { d } _ { \mathrm { B L E } 4 } ^ { u _ { j } , u _ { i } } = \frac { 1 } { 3 S _ { I } } \sum _ { t _ { 0 } = 0 } ^ { 3 S _ { I } - 1 } d _ { \mathrm { B L E } 4 } ^ { u _ { j } , u _ { i } } \left( t _ { 0 } \right)\tag{35}
$$

## 3.3.2. BLE 5

As for the broadcast communication of BLE 5, let $N _ { \mathrm { B L E } 5 } ^ { u _ { j } } \left( t _ { 0 } \right)$ in Eq. (18) denote the set of time slots in which the pointer PDU packets transmitted by $\mathrm { U A V } u _ { j } \in \mathcal { U }$ are successfully received, and $\delta _ { u _ { j } , n } ^ { \mathrm { B L E } 5 }$ indicate the ??-th smallest matching time slot in $N _ { \mathrm { B L E } 5 } ^ { u _ { j } } \left( t _ { 0 } \right)$ expressed as $\delta _ { u _ { j } , n } ^ { \mathrm { B L E } 5 } \ =$ min $\left( \mathcal { N } _ { \mathrm { B L E } } ^ { u _ { j } } \left( t _ { 0 } \right) \right) _ { n }$ . Assuming UAV ?? ?? transmits the Remote ID messages of BLE 5 at the rate of $\psi _ { \mathrm { B L E } 5 } ^ { u _ { j } } ,$ so $\begin{array} { r } { n _ { \mathrm { B L E } 5 } ^ { u _ { j } , \xi } = \sum _ { \forall n } I _ { \xi } \left( \delta _ { u _ { j } , n } ^ { \mathrm { B L E } 5 } \right) } \end{array}$ represents the number of PDU packets successfully received corresponding to the ??-th auxiliary data packet, where $\xi \in \left\{ 1 , 2 , \dots , \psi _ { \mathrm { B L E } 5 } ^ { u _ { j } } \right\}$ , and $I _ { \xi } ( X )$ is defined as

$$
I _ { \xi } ( X ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { ~ i f ~ } t _ { 0 } + ( \xi - 1 ) A _ { I } \leq X < t _ { 0 } + \xi A _ { I } } \\ { 0 , } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right.\tag{36}
$$

According to Eq. (19) and the packet collision probability model of BLE 5, the transmission delay of a Remote ID message from UAV $u _ { j } \in \mathcal { U }$ to UAV $u _ { i } \in S _ { \mathrm { B L E } } ^ { u _ { j } } { } _ { 5 }$ at time $t _ { 0 }$ is

$$
d _ { \mathrm { B L E } 5 } ^ { u _ { j } , \nu } ( t _ { 0 } ) = \sum _ { k = 0 } ^ { \infty } \sum _ { \xi = 1 } ^ { \psi _ { \mathrm { B L E } 5 } ^ { u _ { j } } } G _ { k } H _ { \xi } P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } 5 } ( \xi ) \left( D _ { \mathrm { B L E } 5 } ( \xi ) + k T _ { \mathrm { G N S S } } \right) ( 3 7 )
$$

where

$$
G _ { k } = \prod _ { n = 1 } ^ { k \psi _ { \mathrm { B L E } } ^ { u _ { j } } } \left( 1 - P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } \ 5 } \left( \left( \left( n - 1 \right) \mathrm { m o d } \psi _ { \mathrm { B L E } \ 5 } ^ { u _ { j } } \right) + 1 \right) \right)\tag{38}
$$

$$
H _ { \xi } = \prod _ { m = 1 } ^ { \xi - 1 } \left( 1 - P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } \ 5 } ( m ) \right)\tag{39}
$$

and

$$
\begin{array} { r l } & { P _ { \mathrm { s u c c } } ^ { \mathrm { B L E } ~ 5 } ( \xi ) = \left( 1 - \left( 1 - P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E } ~ 5 , \mathrm { S T I } , \mathrm { P r i } } P _ { \bar { c } , u _ { i } } ^ { \mathrm { B L E } ~ 5 , \mathrm { B L E } ~ 4 , \mathrm { P r i } } \right) ^ { n _ { \mathrm { B L E } } ^ { u _ { j } , \xi } } \right) } \\ & { ~ \cdot P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { B L E } ~ 5 , \mathrm { S T I } , \mathrm { S e c } } P _ { \bar { c } , u _ { i } } ^ { \mathrm { B L E } ~ 5 , \mathrm { W i } - \mathrm { F i } , \mathrm { S e c } } } \end{array}\tag{40}
$$

Therefore, the average transmission delay for a Remote ID message sent by UAV $u _ { j } \in \mathcal { U }$ to $\mathrm { U A V } u _ { i } \in S _ { \mathrm { B L E } 5 } ^ { u _ { j } ^ { * } } { \mathrm { i s } }$

$$
\bar { d } _ { \mathrm { B L E } 5 } ^ { u _ { j } , u _ { i } } = \frac { 1 } { 3 S _ { I } } \sum _ { t _ { 0 } = 0 } ^ { 3 S _ { I } - 1 } d _ { \mathrm { B L E } 5 } ^ { u _ { j } , u _ { i } } ( t _ { 0 } )\tag{41}
$$

## 3.3.3. Wi-Fi

Based on Eq. (20), the set of matching time slots is $M _ { \mathrm { W i - F i } } ^ { c , u _ { j } } \left( t _ { 0 } \right)$ where the beacon packet transmitted by $u _ { j } \in \mathcal { U }$ is successfully received. Let $\delta _ { u _ { j } , n } ^ { \mathrm { W i - F i } }$ represent the ??-th smallest time slot in $M _ { \mathrm { W i - F i } } ^ { c , u _ { j } } \left( t _ { 0 } \right)$ By combining the packet reception delay in Eq. (22) with the packet collision probability model, the transmission delay of a Wi-Fi message from UAV $u _ { j }$ at time $t _ { 0 }$ to UAV $u _ { i } \in S _ { \mathrm { W i - F i } } ^ { u _ { j } }$ is

$$
d _ { \mathrm { W i - F i } } ^ { u _ { j } , u _ { i } } \left( t _ { 0 } \right) = \sum _ { k = 0 } ^ { \infty } { \sum _ { n = 1 } ^ { \left| \mathcal { M } _ { \mathrm { W i - F i } } ^ { c , u _ { j } } \left( t _ { 0 } \right) \right| } \left( 1 - P _ { \mathrm { s u c c } } ^ { \mathrm { W i - F i } } \right) ^ { n - 1 + \left| \mathcal { M } _ { \mathrm { W i - F i } } ^ { c , u _ { j } } \left( t _ { 0 } \right) \right| k } }\tag{42}
$$

where

$$
P _ { \mathrm { s u c c } } ^ { \mathrm { W i - F i } } = P _ { \bar { c } , u _ { j } , u _ { i } } ^ { \mathrm { W i - F i , S T I } } P _ { \bar { c } , u _ { i } } ^ { \mathrm { W i - F i , B L E } \ 5 }\tag{43}
$$

The average delay for UAV $u _ { i } \in S _ { \mathrm { W i - F i } } ^ { u _ { j } }$ to receive a Remote ID message from $u _ { j } \in \mathcal { U }$ is

$$
\vec { d } _ { \mathrm { W i - F i } } ^ { u _ { j } , u _ { i } } = \frac { 1 } { 3 \left( T _ { S } + C _ { T } \right) } \sum _ { t _ { 0 } = 0 } ^ { 3 \left( T _ { S } + C _ { T } \right) - 1 } d _ { \mathrm { W i - F i } } ^ { u _ { j } , u _ { i } } \left( t _ { 0 } \right)\tag{44}
$$

## 3.4. Problem formulation

The objective is to minimize the long-term average message transmission delay for all UAVs, by jointly optimizing the transmission protocol selection $\Lambda ( t ) = \left\{ \lambda _ { u _ { j } } ^ { \epsilon } ( t ) \mid \forall u _ { j } \in \mathcal { U } , \epsilon \in \mathcal { E } \right\}$ and the message transmission rate $\Psi ( t ) = \left\{ \psi _ { \epsilon } ^ { u _ { j } } ( t ) \mid \forall u _ { j } \in { \mathcal { U } } , \epsilon \in { \mathcal { E } } \right\}$

$$
\mathrm { P 0 } \mathrm { : } \operatorname* { m i n } _ { \Lambda ( t ) , \Psi ( t ) } \frac { 1 } { T _ { \operatorname* { m a x } } } \sum _ { t \in \mathcal { T } } \sum _ { u _ { j } \in \mathcal { U } } \frac { 1 } { | S _ { u _ { j } } | } \sum _ { u _ { i } \in S _ { u _ { j } } } \sum _ { \epsilon \in \mathcal { E } } \left( \bar { d } _ { \epsilon } ^ { u _ { j } , u _ { i } } ( t ) \lambda _ { u _ { j } } ^ { \epsilon } \right)\tag{45}
$$

$$
\mathrm { s . t . } \sum _ { \epsilon \in { \mathcal E } } \lambda _ { u _ { j } } ^ { \epsilon } ( t ) = 1 , \quad \forall u _ { j } \in { \mathcal U } ,\tag{45a}
$$

$$
\lambda _ { u _ { j } } ^ { \epsilon } ( t ) \in \{ 0 , 1 \} , \quad \forall u _ { j } \in \mathcal { U } , \epsilon \in \mathcal { E } ,\tag{45b}
$$

$$
\psi _ { \epsilon } ^ { u _ { j } } ( t ) \leq \psi _ { \epsilon , \operatorname* { m a x } } , \quad \forall u _ { j } \in \mathcal { U } , \epsilon \in \mathcal { E } ,\tag{45c}
$$

$$
\psi _ { \epsilon } ^ { u _ { j } } ( t ) \in \mathbb { Z } ^ { + } , \quad \forall u _ { j } \in \mathcal { U } , \epsilon \in \mathcal { E } .\tag{45d}
$$

Wherein, $T _ { \mathrm { m a x } }$ denotes the total observation duration, and T represents the set of all time points, $S _ { u _ { j } }$ is defined as the union of communication sets for $\mathrm { U A V } u _ { j }$ , and $S _ { u _ { j } } = \left\{ S _ { \mathrm { B L E } 4 } ^ { u _ { j } } \cup S _ { \mathrm { B L E } 5 } ^ { u _ { j } } \cup S _ { \mathrm { W i - F i } } ^ { u _ { j } } \right\}$

Constraint (45a) ensures that each $\mathrm { U A V } u _ { j }$ selects exactly one communication protocol from the set E. Constraint (45b) enforces a binary decision for the communication protocol selection. Constraint (45c) limits the maximum message transmission rate $\psi _ { \epsilon , \mathrm { m a x } }$ for each communication protocol ??. Constraint (45d) ensures that the message transmission rate $\psi _ { \epsilon } ^ { u _ { j } } ( t )$ is a positive integer. Note that the transmission delay is directly related with collision avoidance performance in the DMUCA framework.

P0 is a non-linear integer optimization with dynamic, decentralized decision-making. Due to the dynamic environment and limited local observations, traditional optimization methods are unavailable. The multi-agent DRL algorithms can efficiently manage decentralized decision-making, which facilitates the autonomous coordination among UAVs, adapting to real-time network changes and ensuring robust performance in environments with incomplete information and unpredictable dynamics.

## 4. Algorithm design

To deal with P0, it is initially transformed as an MDP, and then a DRL-based algorithm MADQN-ATMC is designed. The MADQN framework is selected since it supports the decentralized learning in dynamic and partially observable environments. Each UAV acts as an independent agent that learns to select communication protocols based on its own observations and feedback from the environment. This structure is well-suited for real-time UAV systems, where the centralized control is generally impractical due to the limited connectivity or latency.

## 4.1. MDP transformation

To model the problem as an MDP, we define the key components, i.e., the state space, action space, and reward functions as follows.

## 4.1.1. State space

At time ??, each $\mathrm { U A V } u _ { j } \in \mathcal { U }$ obtains a local observation $o _ { u _ { j } } ( t )$ from the environment, which includes information from nearby UAVs. In detail,

$$
o _ { u _ { j } } ( t ) \triangleq \left( \mathbf { A } _ { u _ { j } } ^ { s } ( t ) , \boldsymbol { \Psi } _ { u _ { j } } ^ { s } ( t ) , \boldsymbol { D } ^ { \mathcal { R } ^ { u _ { j } } } ( t ) , \mathbf { A } ^ { \mathcal { R } ^ { u _ { j } } } ( t ) , \boldsymbol { \Psi } ^ { \mathcal { R } ^ { u _ { j } } } ( t ) \right)\tag{46}
$$

where $\Lambda _ { u _ { j } } ^ { s } ( t ) = \left\{ \lambda _ { u _ { j } } ^ { \epsilon } ( t ) \ | \ \epsilon \in \mathcal { E } \right\}$ are the communication protocols selected by $\mathrm { U A V } u _ { j }$ at time ??. $\Psi _ { u _ { j } } ^ { s } ( t ) = \left\{ \psi _ { \epsilon } ^ { u _ { j } } ( t ) \vert \epsilon \in \mathcal { E } \right\}$ denotes the message transmission rates corresponding to the selected protocols. $\begin{array} { r } { D ^ { \mathcal { R } ^ { u _ { j } } } ( t ) = \left\{ d _ { u _ { k } , u _ { j } } ( t ) ~ | ~ u _ { k } \in \bigcup _ { \epsilon \in \mathcal { E } } \mathcal { R } _ { \epsilon } ^ { u _ { j } } \right\} } \end{array}$ are the distances between $\mathrm { U A V } u _ { j }$ and ????. $\begin{array} { r } { \mathbf { \Delta } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathcal { R } ^ { u _ { j } } } \left( t \right) = \left\{ \lambda _ { u _ { k } } ^ { \epsilon } ( t ) \mid \epsilon \in \mathcal { E } , u _ { k } \in \bigcup _ { \epsilon \in \mathcal { E } } \mathcal { R } _ { \epsilon } ^ { u _ { j } } \right\} } \end{array}$ is the set of protocols used by $\mathrm { U A V } _ { u _ { k } }$ transmitting messages to UAV ?? ?? . $\begin{array} { r } { \Psi ^ { \mathcal { R } ^ { u _ { j } } } \left( t \right) = \left\{ \psi _ { \epsilon } ^ { u _ { k } } \left( t \right) \mid \epsilon \in \mathcal { E } , u _ { k } \in \bigcup _ { \epsilon \in \mathcal { E } } \mathcal { R } _ { \epsilon } ^ { u _ { j } } \right\} } \end{array}$ indicates the message transmission rates of $\mathrm { U A V } u _ { k }$ ）

The overall environment state at time ?? is then defined as $s ( t ) \triangleq$ $\left\{ o _ { u _ { 1 } } ( t ) , o _ { u _ { 2 } } ( t ) , \ldots , o _ { u _ { M } } ( t ) \right\}$ . The state space is represented as $s =$ $\{ s ( t ) , t \in \mathcal { T } \}$

## 4.1.2. Action space

Based on the observation $o _ { j } ( t )$ , the action of UAV $u _ { j }$ at time ?? is defined as

$$
a _ { u _ { j } } ( t ) \triangleq \left( \mathbf { \Lambda } \mathbf { { \Lambda } } \mathbf { { \Lambda } } _ { u _ { j } } ^ { a } ( t ) , \mathbf { \Psi } \mathbf { { \Psi } } _ { u _ { j } } ^ { a } ( t ) \right)\tag{47}
$$

where $\Lambda _ { u _ { j } } ^ { a } ( t ) = \left\{ \lambda _ { u _ { j } } ^ { \epsilon , a } ( t ) \vert \epsilon \in \mathcal { E } \right\}$ denotes the selected communi-

$$
\Psi _ { u _ { j } } ^ { a } ( t ) = \left\{ \psi _ { \epsilon } ^ { u _ { j } , a } ( t ) \mid \epsilon \in \mathcal { E } \right\}
$$

The joint actions of all UAVs at time ?? are expressed as $\ \mathbf { \boldsymbol { a } } ( t ) =$ $\left\{ a _ { u _ { 1 } } ( t ) , a _ { u _ { 2 } } ( t ) , \dotsc , a _ { u _ { M } } ( t ) \right\}$ and the action space is $\mathcal { A } = \{ \pmb { a } ( t ) , t \in$ $\mathcal { T } \}$

## 4.1.3. Reward function

The reward function is designed to optimize the transmission delays of Remote ID message by considering both local transmission performance and global delay, which is defined as

???? ?? (??) = ????local?? ?? (??) + ????global?? ?? (??) (48) where $\begin{array} { r } { r _ { u _ { j } } ^ { \mathrm { l o c a l } } ( t ) = - \frac { 1 } { \left| S _ { u _ { j } } \right| } \sum _ { u _ { i } \in S _ { u _ { j } } } \bar { d } ^ { u _ { j } , u _ { i } } ( t ) } \end{array}$ motivates UAV $u _ { j }$ to minimize the message delays to its neighboring $\mathrm { U A V s } ,$ enhancing local communication efficiency. $\begin{array} { r } { r _ { u _ { j } } ^ { \mathrm { g l o b a l } } ( t ) = \frac { 1 } { \left| S _ { u _ { j } } \right| } \sum _ { u _ { i } \in S _ { u _ { j } } } ( \bar { d } _ { \mathrm { g l o b a l } } ( t ) - } \end{array}$ $\bar { d } ^ { u _ { j } , u _ { i } } ( t ) )$ prevents UAVs from making suboptimal decisions due to limited local observations, ensuring alignment with the global transmission average. The global average transmission delay $\bar { d } _ { \mathrm { g l o b a l } } ( t ) =$ 1?? $\begin{array} { r } { \sum _ { \boldsymbol { u _ { j } } \in \mathcal { U } } \frac { 1 } { \left| S _ { \boldsymbol { u _ { j } } } \right| } \sum _ { \boldsymbol { u _ { i } } \in S _ { \boldsymbol { u _ { j } } } } \bar { d } ^ { \boldsymbol { u _ { j } } , \boldsymbol { u _ { i } } } \left( t \right) } \end{array}$ . ?? and $\beta$ are weighting factors to balance the impact of local and global delays.

## 4.2. MADQN-ATMC algorithm

In dynamic environments, it is challenging for UAVs to obtain global information, so the distributed decisions are necessary. We design the MADQN algorithm to provide distributed training and execution. In particular, each agent in MADQN has two Q-networks: the primary Q-network $Q _ { \theta _ { j } } \left( o _ { u _ { j } } , a _ { u _ { j } } \right)$ and the target Q-network $Q _ { \theta _ { j } ^ { \prime } } \left( o _ { u _ { j } } , a _ { u _ { j } } \right)$ . Here, $o _ { u _ { j } }$ denotes the local observation of UAV $u _ { j } ,$ , and $a _ { u _ { j } }$ represents the action of $u _ { j }$ . The target Q-network stabilizes training by periodically updating its parameters $\theta _ { j } ^ { \prime }$ to match the parameters of the primary network $\theta _ { j }$ . The Q-value function is updated by the Temporal Difference (TD) learning, and the TD target is computed as

$$
y _ { j } ( t ) = r _ { j } ( t ) + \operatorname* { m a x } _ { a _ { u _ { j } } ( t + 1 ) } Q _ { \theta _ { j } ^ { \prime } } \left( o _ { u _ { j } } ( t + 1 ) , a _ { u _ { j } } ( t + 1 ) \right)\tag{49}
$$

where $r _ { j } ( t )$ is the reward received by $\mathrm { U A V } u _ { j }$ at time ??, and ?? is the discount factor. The TD target guides the Q-value updates through the loss function ${ \mathcal { L } } ( \theta _ { j } ) , { \mathrm { i . e . } }$

$$
\mathcal { L } \left( \theta _ { j } \right) = \frac { 1 } { \left| \mathcal { B } \right| } \sum _ { i = 1 } ^ { | \mathcal { B } | } \left[ y _ { j } ^ { i } ( t ) - \mathcal { Q } _ { \theta _ { j } } \left( o _ { u _ { j } } ^ { i } ( t ) , a _ { u _ { j } } ^ { i } ( t ) \right) \right] ^ { 2 }\tag{50}
$$

where $\mathcal { B }$ represents a batch of experiences sampled from the replay buffer, and |B| is the batch size.

To enhance the stability, the target Q-network is updated by the following soft update mechanism:

$$
\theta _ { j } ^ { \prime }  \tau \theta _ { j } + ( 1 - \tau ) \theta _ { j } ^ { \prime }\tag{51}
$$

where $\tau \in \ [ 0 , 1 ]$ controls the update rate, with smaller ?? values ensuring slower and more stable updates.

Based on the MDP framework, we integrate the MADQN training process to optimize the long-term transmission delay. During the training, each UAV selects actions based on the exploration rate, and adjusts the communication mode accordingly. The UAV then receives a reward from the environment and transfers to the next state. When sufficient experience is accumulated in the replay buffer, the training proceeds by updating the Q-network parameters through loss minimization, followed by gradually updating the target Q-network parameters. The MADQN-ATMC algorithm is outlined in Algorithm 1, which summarizes the entire process.

Algorithm 1 MADQN-ATMC algorithm.   
1: Input: Maximum steps per episode $T _ { \mathrm { m a x } } .$ , max-episode, explo  
ration settings (initial exploration rate $e _ { \mathrm { i n i t } } ,$ final exploration   
rate $e _ { \mathrm { f i n a l } }$ and maximum exploration episodes $E _ { \mathrm { m a x } } )$ , batch size   
$| { \mathcal { B } } | ,$ and experience replay buffer ?? with maximum capacity   
$D _ { \mathrm { m a x } } .$   
2: Output: Trained Q-networks for each UAV.   
3: Initialize: The number of UAVs ??, Q-networks (primary and   
target) for each UAV.   
4: for episode = 1:max-episode do   
5: if episode $< E _ { \mathrm { m a x } }$ max thenepisode(??init −??final )??max . then   
6: $\begin{array} { r } { e = e _ { \mathrm { i n i t } } - \frac { \mathrm { e p i s o d e } ( e _ { \mathrm { i n i t } } - e _ { \mathrm { f i n a l } } ) } { E _ { \mathrm { m a x } } } . } \end{array}$   
7: else   
8: $e = e _ { \mathrm { f i n a l } } .$   
9: end if   
10: for $t = 1 : T _ { \operatorname* { m a x } }$ do   
11: for $j = 1 : M$ do   
12: Observe state $o _ { u _ { j } } ( t ) .$   
13: Generate a random number ?? ∼ Uniform(0, 1).   
14: if $\cdot \ r < e$ then   
15: ?????? (??) = random action.   
16: else   
17: $\begin{array} { r } { \mathbf { \tilde { \sigma } } ^ { a } { } _ { u _ { j } } ( t ) = \arg \operatorname* { m a x } _ { a _ { u _ { j } } } \mathcal { Q } _ { \theta _ { j } } \left( o _ { u _ { j } } { } ( t ) , a _ { u _ { j } } \right) . } \end{array}$   
18: end if   
9: Execute action $a _ { u _ { j } } ( t )$ , obtain reward $r _ { u _ { j } } ( t )$ and ob  
serve next state $o _ { u _ { j } } ( t + 1 )$   
20: Store experience $\langle \dot { o } _ { u _ { j } } ( t ) , a _ { u _ { j } } ( t ) , r _ { u _ { j } } ( t ) , o _ { u _ { j } } ( t + 1 ) \rangle$ in   
replay buffer $D .$   
21: end for   
22: if $| D | \geq D _ { \operatorname* { m a x } }$ then   
23: for $j = 1 : M$ do   
24: Sample mini-batch B from the replay buffer ??.   
25: Update the Q-network using the sampled mini-batch   
via Eq. (50).   
26: Update the target Q-network via Eq. (51).   
27: end for   
28: end if   
29: Randomly update UAV positions in the environment.   
30: end for   
31: end for

## 5. Simulation results

In this section, we evaluate the performance of the proposed DMUCA framework, as well as communication modes of BLE 4, BLE 5, and Wi-Fi, and validate the effectiveness of the MADQN-ATMC algorithm.

## 5.1. Performance of DMUCA

To validate the DMUCA framework, we consider the scenario of a 1 km×1 km airspace, where five UAVs follow pre-planned trajectories generated using the waypoint trajectory toolbox of MATLAB. By specifying custom waypoints including starting positions, intermediate points, and destinations, this method produces UAV trajectories with proper kinematic constraints to simulate realistic mission scenarios. Each trajectory corresponds to a flight duration of 500 s, with all UAVs converging within the designated 500 m×500 m×300 m area at 250 s. Each UAV has a conflict radius of 5 m, a physical radius of 1 m, and a maximum flight speed of 5m/s. UAVs employ Remote ID to broadcast real-time positions and velocity updates. To evaluate the impact of transmission delays on collision avoidance, Remote ID message transmission delays are sampled from three intervals: [0, 1] s, [1, 2] s, and [2, 3] s, with the GNSS update cycle set to 1s. Simulation results highlight the performance of DMUCA with these conditions.

Fig. 3 presents the collision avoidance results with the DMUCA framework, with transmission delays sampled from [0, 1] s. Fig. 3(a) illustrates the 2D flight trajectories of the UAVs, with the lines representing the UAV flight paths and the points indicating their positions at specific times. Fig. 3(b) provides a 3D view of their positions before and after potential conflicts. The trajectories confirm that UAVs consistently maintain safe separation distances at critical moments. Figs. 3(c) and 3(d) shows the real-time separation distances between UAV pairs over the entire flight duration, with lines indicating the distance between each pair of UAVs at each time point. Under the delay condition of [0, 1] s, all UAVs maintain a minimum separation distance greater than 10m, which corresponds to the combined conflict radii of two UAVs in proximity. The results confirm that DMUCA not only avoids collisions but also maintains a conservative safety margin under low-delay conditions, which is critical for ensuring safe UAV operations in the real-time airspace management.

Fig. 4 analyzes the effect of increasing delays on the collision avoidance by comparing the minimum separation distances between UAVs with delay intervals of [0, 1] s, [1, 2] s, and [2, 3] s. The minimum separation distance is the closest proximity achieved between any two UAVs during the flight. As the delay increases, the predictive accuracy declines, aggravating the collision avoidance performance. As for delays within [0, 1] s, all UAVs maintain a minimum separation above the 10 m conflict threshold. However, for delays of [1, 2] s, UAV1-UAV3 and UAV2-UAV3 achieve minimum separations of 9.75 m and 9.94 m, respectively. For delays in the range of [2, 3] s, six UAV pairs have minimum separations below the conflict threshold.

Notably, even with the highest delay, the minimum separation distance of all UAV pairs remains above 2 m, ensuring no physical collisions. It highlights that although longer delays reduce the prediction accuracy, DMUCA still ensures a basic safety margin, demonstrating its practical robustness for real-time UAV applications where delays cannot always be avoided.

## 5.2. Performance of Remote ID modes

To establish the performance benchmarks for the adaptive communication protocol selection, we verify the differences among BLE 4, BLE 5, and Wi-Fi under fixed protocol modes of Remote ID. The simulation involves 10 UAVs, operating at altitudes between 30 m and 120 m, with random flight paths across spatial environments of varying sizes: $1 0 0 ~ \mathrm { m } ^ { 2 } , 5 0 \overline { { { 0 } } } ~ \mathrm { m } ^ { 2 } , 1 ~ 0 0 0 ~ \mathrm { m } ^ { 2 } , \dot { 3 } ~ 0 0 0 ~ \mathrm { m } ^ { 2 } , 5 0 0 0 ~ \mathrm { m } ^ { 2 } .$ and $1 0 0 0 0 \mathrm { m } ^ { 2 }$ . The UAVs have a maximum flight speed of 20 m/s. Given the scenario involving urban UAV-to-UAV signal transmission, we adopt the log-normal shadowing model to account for the shadow attenuation caused by obstacles such as buildings in city environments. The detailed protocol parameters are listed in Table $2 ^ { 4 6 , 4 8 }$ . Fig. 5 illustrates how the transmission delay varies for BLE 4, BLE 5, and Wi-Fi with the variation of message transmission rates across different spatial sizes with fixed protocol modes. Generally, an increase in message transmission rate leads to a reduction in transmission delays for all protocols, as higher rates enhance the probability of successful reception within each GNSS update cycle. However, the delays of BLE 4 and Wi-Fi face sharp increase at certain message transmission rates, primarily due to the temporary mismatches between the transmission and reception cycles 49. Specifically, at transmission rates of 5 and 10 for BLE 4, and 4, 7, and 8 for Wi-Fi, the misalignment between packet transmission and reception leads to intermittent packet losses, causing sudden increases in packet reception delays. In contrast, BLE 5 demonstrates a smooth decrease in transmission delay as the message transmission rate increases, owing to its dual-channel design, which uses pointer packets on the primary channel and data packets on the secondary channel, ensuring stable and reliable message delivery despite timing mismatches. Overall, with the current protocol configurations, BLE 4 performs optimally at the transmission rate of 9, while BLE 5 and Wi-Fi perform optimally at the transmission rate of 10. Fig. 5(a) demonstrates that in smaller and high-density environments $( 1 0 0 \mathrm { m } ^ { 2 } , 5 0 0 \mathrm { m } ^ { 2 }$ , and $1 0 0 0 \mathrm { m } ^ { 2 } )$ , the optimal Wi-Fi transmission mode yields the lowest transmission delay while BLE 5 with the highest delay. Fig. 5(b) illustrates that in larger and lowdensity environments (3 000 m2, 5 000 m2, and $1 0 \ 0 0 0 \ \mathrm { m } ^ { 2 } )$ , the BLE 4 mode achieves the lowest delay.

Table 2 Parameter settings for BLE 4, BLE 5, and Wi-Fi.
<table><tr><td>Parameter</td><td>BLE 4</td><td>BLE 5</td><td>Wi-Fi</td></tr><tr><td>Data rate (Mbit/s)</td><td>1</td><td>0.125</td><td>1</td></tr><tr><td>Transmit power (dBm)</td><td>18</td><td>18</td><td>18</td></tr><tr><td>Receiver sensitivity(dBm)</td><td>-85</td><td>-97</td><td>-105</td></tr><tr><td>Path loss exponent</td><td>2.1</td><td>2.1</td><td>2.1</td></tr><tr><td>Shadowing variance (dB)</td><td>6</td><td>6</td><td>6</td></tr><tr><td>4(ms)</td><td> $0 . 1 2 5$ </td><td>0.125</td><td>0.125</td></tr><tr><td> $A _ { P } \ ( \mathrm { m s } )$ </td><td>0.376</td><td>1.152</td><td></td></tr><tr><td> $P _ { I } \ ( \mathrm { m s } )$ </td><td>0.125</td><td>0.125</td><td></td></tr><tr><td> $T _ { \mathrm { A U X } } ( \mathrm { m s } )$ </td><td></td><td>3.328</td><td></td></tr><tr><td> $A _ { \mathrm { O f f s e t } } ( \mathrm { m s } )$ </td><td></td><td>5</td><td></td></tr><tr><td> $R _ { D } \ \mathrm { ( m s ) }$ </td><td>528</td><td>5</td><td></td></tr><tr><td> $S _ { W } \left( \mathrm { m s } \right)$ </td><td></td><td>2</td><td></td></tr><tr><td>S1 (ms)</td><td></td><td>8</td><td></td></tr><tr><td> $B _ { D } \ \mathrm { ( m s ) }$ </td><td></td><td></td><td>0.632</td></tr><tr><td> $T _ { S } ~ ( \mathrm { m s } )$ </td><td></td><td></td><td>6</td></tr><tr><td> $C _ { T } \ \mathrm { ( m s ) }$ </td><td></td><td>，</td><td>1</td></tr><tr><td>Broadcast channel</td><td>Fixed</td><td>Fixed</td><td>Random from {1,6,11}</td></tr></table>

Fig. 6 explains the performance differences observed in Fig. 5. In particular, in Figs. 6(a), 6(b), and 6(c), the packet loss rates for BLE 5 and Wi-Fi remain unchanged in smaller environments due to their long transmission ranges (BLE 5: up to 1 000 m, Wi-Fi: up to 2 000 m), indicating that UAVs remain within the communication range of other UAVs. In contrast, BLE 4, with a shorter range (up to 250 m), shows a significant reduction in packet loss rate as the spatial size increases. Notably, even with high packet loss, Wi-Fi maintains a lower average transmission delay, demonstrating superior resistance to interference and better performance in smaller airspace sizes. However, BLE 5 exhibits higher packet loss due to its lower transmission rate, leading to increased STI. In Figs. 6(d), 6(e), and 6(f), the packet loss for BLE 5 and Wi-Fi decreases in larger spatial sizes. However, as the spatial size increases, the packet loss for BLE 4 approaches zero, leading to lower transmission delays in such conditions.

These results provide guidances for selecting communication protocols in practical urban UAV operations. BLE 4 performs best in large and less crowded areas. It shows low delay and low packet loss when the communication space is wide. Wi-Fi is the most effective choice in small and dense environments. It maintains the lowest delay even when packet loss is high. BLE 5 provides stable delay performance, but it suffers from higher packet loss in crowded conditions, which can affect communication reliability. These analyses can help define the baseline performance for each protocol and also support the proposed DMUCA framework, which allows UAVs to switch protocols based on the size and density of their environments.

<!-- image-->  
(a) 2D UAV f light trajectories

<!-- image-->  
(b) 3D UAV f light trajectories

<!-- image-->  
(c) Real-time pairwise UAV separation distance

<!-- image-->  
(d) Enlarged view

Fig. 3 UAV trajectories and conflict avoidance performance using DMUCA framework (random Remote ID message transmission delays sampled from [0, 1] s).  
<!-- image-->  
Fig. 4 Impact of transmission delays on UAV collision avoidance performance: Minimum separation distances under varying delay conditions.

## 5.3. Performance of MADQN-ATMC

We select 10 UAVs for the algorithm validation, since this scale sufficiently captures the interaction complexity within typical communication and density constraints. In real-world deployments, the required training swarm size is determined by two physical factors: the maximum operational communication range and the regulated UAV density in the airspace. By ensuring that every decision made by a UAV relies solely on the neighboring agents within its communication coverage area, this algorithm inherently has the ability to be applied and work effectively in larger UAV swarms. The training parameters to evaluate MADQN-ATMC are outlined in Table 3. Simulations are conducted using Python 3.9 and TensorFlow 2.10.

Table 3 Parameter settings for the MADQN-ATMC algorithm.
<table><tr><td>Parameter Description</td><td>Value</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Replay buffer size</td><td>25000</td></tr><tr><td>Discount factor</td><td>0.95</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>0.000 1</td></tr><tr><td>Soft update rate</td><td>0.999</td></tr><tr><td>Initial/Final exploration rate</td><td>1/0.1</td></tr><tr><td>Exploration decay period</td><td>500</td></tr><tr><td>Total number of episodes</td><td>1000</td></tr><tr><td>Number of steps in each episode</td><td>100</td></tr><tr><td>Number of hidden layers</td><td>2</td></tr><tr><td>Number of neurons in each layer</td><td>256/128</td></tr><tr><td>Weight for local/global reward</td><td>1/1</td></tr><tr><td>UAV flight range</td><td>100 m² to 10 000 m²</td></tr></table>

<!-- image-->  
(a) Transmission delay for small spatial sizes

：： BLE 4 (1 000 m2 ) BLE 5 (1 000 m2 ) Wi-Fi (1 000 m2 )  
<!-- image-->  
(b) Transmission delay for large spatial sizes  
BLE 4 (10 000 m2 ) BLE 5 (10 000 m2 ) Wi-Fi (10 000 m2 )  
Fig. 5 Average transmission delay across different spatial sizes using fixed protocol modes.

To validate the proposed MADQN-ATMC algorithm, we compare it with two representative baseline methods: $\mathbf { \bar { M A D D P G } } ^ { 5 0 }$ and Independent Actor-Critic $( \mathrm { I A C } ) ^ { 5 1 }$ . In detail, MADDPG leverages the centralized training with the decentralized execution. However, it needs to collect global action information from all agents during the training, which limits its scalability in real UAV networks with constrained communication resources. IAC is a fully distributed method based on the policy-based design. It maps the observations directly to actions without learning value functions. MADQN-ATMC adopts a value-based framework that evaluates Q-values from local observations. This design avoids centralized components, supports stable learning in discrete action spaces, and is better suited for the decentralized, real-time UAV environments with partial observabilities.

Fig. 7(a) compares the system-wide convergence performance of the proposed MADQN-ATMC algorithm with baseline methods. All algorithms begin training only after the replay buffer is filled with 250 episodes of experiences. Before this point, the agents collect data using randomly initialized policies, which are stored but not yet used for learning. This setting leads to a sharp increase in performance once the training begins. The results show that both MADQN-ATMC and MADDPG achieve convergence, with MADQN-ATMC converging within 500 episodes, while MADDPG requires approximately 800 episodes. In contrast, IAC fails to converge even after 1 000 episodes, highlighting the superior convergence speed of the proposed algorithm.

Figs. 7(b), 7(c), and 7(d) illustrate the convergence trends for individual UAVs across different algorithms. The proposed algorithm demonstrates consistent convergence across all $1 0 \mathrm { U A V s } ,$ , with each UAV rapidly and stably reaching the optimal performance. In comparison, MADDPG exhibits greater variance, with two UAVs failing to reach the optimal performance by the final episode. IAC performs the worst, with most UAVs failing to converge. These results emphasize the advantages of MADQN-ATMC in terms of the convergence speed, consistency, and overall quality.

Fig. 8 evaluates the system-wide average transmission delay for 10 UAVs across varying airspace densities. The ”high airspace density” refers to operations within $1 0 0 \mathrm { m } ^ { 2 } , 5 0 0 \mathrm { m } ^ { 2 } .$ and $1 0 0 0 \mathrm { { m } } ^ { 2 }$ and ”low airspace density” corresponds to 3 000 $\mathrm { m } ^ { 2 } , 5 \ 0 0 0 \ \mathrm { m } ^ { 2 }$ and $1 0 0 0 0 \mathrm { m } ^ { \hat { 2 } } .$ The ”dynamic airspace density” represents scenarios spanning both high and low density environments. Results are compared with fixed transmission modes, where each protocol uses its optimal message transmission rate. In high density scenarios, the MADQN-ATMC algorithm achieves the lowest average delay of 1 855.28 ms, closely matching the optimal fixed Wi-Fi configuration at 1 865.07 ms, indicating the ability of MADQN-ATMC to autonomously select the Wi-Fi protocol and adjust the message transmission rate. In contrast, MADDPG and IAC exhibit higher delays, failing to effectively optimize communication. BLE 5 and random transmission methods result in significantly higher delays. In low density scenarios, the MADQN-ATMC algorithm achieves an average delay of 246.82 ms, near the optimal delay of 232.38 ms for the fixed BLE 4 configuration, while the fixed Wi-Fi configuration incurs a higher delay of 1 487.69 ms. Although MADDPG and IAC perform better than fixed ${ \mathrm { W i - F i } } .$ , the proposed algorithm outperforms all approaches. In dynamic-density environments, the MADQN-ATMC algorithm achieves an average delay of 1 050.46 ms, outperforming the optimal fixed BLE 4 and Wi-Fi configurations (1 544.65 ms and 1 671.77 ms respectively) and the MADDPG approach (1 302.18 ms). Such results show a delay reduction of approximately 32% compared to BLE 4 and 19% compared to MAD-DPG, underscoring the advantages of dynamic protocol switching. This improvement demonstrates the practical advantage of adaptive protocol selection in reducing communication delay, which directly contributes to improving the UAV collision avoidance performance and operational safety.

<!-- image-->  
Fig. 8 Average system-wide transmission delays for different communication methods across various airspace densities.

Fig. 9 illustrates the individual message transmission delay for 10 UAVs in the dynamic-density environment. With the proposed algorithm, the average delay for each UAV remains consistently around 100 ms. In contrast, MADDPG shows higher delays for UAVs 5 and 6, approximately 200 ms, indicating incomplete convergence during training. This is because the critic network in MADDPG needs to optimize the joint action values of all agents simultaneously during training. It leads to the individual policy gradients of specific agents being affected by the actions of the dominant agent in the gradient update process, making it impossible to effectively learn the optimal strategy. For IAC, only UAVs 2 and 4 achieve lower delays than those observed with fixed transmission protocols. Compared to fixed protocols and random transmission methods, the proposed algorithm significantly reduces Remote ID message transmission delays across all UAVs.

<!-- image-->  
Fig. 9 Individual transmission delays of 10 UAVs in dynamic airspace density across different communication methods.

The average protocol switching time of the proposed algorithm is 3.84 ms, representing the decision-making delay after that the neural network processes the input data. Such a short duration has little effect on the collision avoidance. UAVs maintain communications using the prior protocol during switching, ensuring seamless operations. Additionally, the low switching frequency minimizes disruptions, maintaining system stability and decision-making performance.

## 6. Conclusions and future directions

In this paper, we propose a real-time distributed collision avoidance DMCUA framework for multi-UAVs based on Remote ID. To improve the collision avoidance performance, we analyze the message transmission delays of the three Remote ID communication protocols from the perspectives of packet reception and collision, and formulate a long-term optimization problem to minimize the transmission delay. Then, we design the MADQN-ATMC algorithm, enabling UAVs to autonomously and dynamically select communication modes. Simulation results validate the effectiveness of the proposed framework and emphasize the critical importance of delay optimization. The proposed algorithm reduces the average transmission delay by 32% compared to the optimal fixed communication mode.

In future works, we will extend the proposed framework to more complex and realistic scenarios, including non-cooperative UAVs and dynamic obstacles, and system uncertainties. We also plan to test the Remote ID platform to study how hardware differences influence the system performance in real-world deployments.

## Acknowledgements

This work was co-supported by the National Natural Science Foundation of China (Nos. 62231015 and 62301251), Natural Science Foundation of Jiangsu Province of China (No. BK20220883), Aeronautical Science Foundation of China (No. 2023Z071052007), and Young Elite Scientists Sponsorship Program by CAST, China (No. 2023QNRC001).

## References

1. Cheng N, Wu S, Wang X, et al. AI for UAV-assisted IoT applications: a comprehensive review. IEEE Internet Things J 2023;10(16):14438-61.

2. Cao X, Su X, Yang P, et al. Survey on near-space information networks: channel modeling, transmission, and networking perspectives. IEEE Commun Surv Tutorials 2025. Early access.

3. Jia Z, Cui C, Dong C, et al. Distributionally robust optimization for aerial multi-access edge computing via cooperation of UAVs and HAPs. IEEE Trans Mob Comput 2025. Early access.

4. Dai Y, Lyu L, Cheng N, et al. A survey of graph-based resource management in wireless networks -part II: learning approaches. IEEE Trans Cognit Commun Networking 2025;11(4):2078-100.

5. Yang D, Wang J, Wu F, et al. Energy efficient transmission strategy for mobile edge computing network in UAV-based patrol inspection system. IEEE Trans Mob Comput 2024;23(5):5984-98.

6. Cao Y, Cheng X, Mu J. Concentrated coverage path planning algorithm of UAV formation for aerial photography. IEEE Sens J 2022;22(11):11098-111.

7. Liao Y, Jia Z, Dong C, et al. Interference analysis for coexistence of UAVs and civil aircrafts based on automatic dependent surveillancebroadcast. IEEE Trans Veh Technol 2024;73(10):15911-5.

8. Rezaee MR, Hamid NAWA, Hussin M, et al. Comprehensive review of drones collision avoidance schemes: challenges and open issues. IEEE Trans Intell Transp Syst 2024;25(7):6397-426.

9. Wei Z, Meng Z, Lai M, et al. Anti-collision technologies for unmanned aerial vehicles: recent advances and future trends. IEEE Internet Things J 2022;9(10):7619-38.

10. Jia Z, You J, Dong C, et al. Cooperative cognitive dynamic system in UAV swarms: reconfigurable mechanism and framework. IEEE Veh Technol Mag 2024;19(3):90-101.

11. Xu Z, Chen B, Zhan X, et al. A vision-based autonomous UAV inspection framework for unknown tunnel construction sites with dynamic obstacles. IEEE Rob Autom Lett 2023;8(8):4983-90.

12. Lindqvist B, Mansouri SS, Haluška J, et al. Reactive navigation of an unmanned aerial vehicle with perception-based obstacle avoidance constraints. IEEE Trans Control Syst Technol 2022;30(5):1847-62.

13. Hügler P, Roos F, Schartel M, et al. Radar taking off: new capabilities for UAVs. IEEE Microwave Mag 2018;19(7):43-53.

14. NT SK, Yadav S, Rajalakshmi P. A sliding window technique-based radar and camera fusion model for object detection in adverse weather condition. IEEE Sens Lett 2024;8(6):1-4.

15. Wu Y, Jia Z, Wu Q, et al. Adaptive QoE-aware SFC orchestration in UAV networks: a deep reinforcement learning approach. IEEE Trans Network Sci Eng 2024;11(6):6052-65.

16. Tedeschi P, Al Nuaimi FA, Awad AI, et al. Privacy-aware remote identification for unmanned aerial vehicles: current solutions, potential threats, and future directions. IEEE Trans Ind Inf 2024;20(2):1069-80.

17. Vinogradov E, Pollin S. Reducing safe UAV separation distances with U2U communication and new Remote ID formats. IEEE globecom workshops. 2022.

18. Raheb R, James S, Hudak A, et al. Impact of communications quality of service (QoS) on Remote ID as an unmanned aircraft (UA) coordination mechanism. IEEE/AIAA 40th digital avionics systems conference. 2021.

19. Tedeschi P, Al Nuaimi FA, Awad AI, et al. Privacy-aware remote identification for unmanned aerial vehicles: current solutions, potential threats, and future directions. IEEE Trans Ind Inf 2024;20(2):1069-80.

20. Javaid S, Saeed N, Qadir Z, et al. Communication and control in collaborative UAVs: recent advances and future trends. IEEE Trans Intell Transp Syst 2023;24(6):5719-39.

21. Vinogradov E, Minucci F, Pollin S. Wireless communication for safe UAVs: from long-range deconfliction to short-range collision avoidance. IEEE Veh Technol Mag 2020;15(2):88-95.

22. Chang S, Shin H. 2.4-GHz CMOS bluetooth RF receiver with improved IM2 distortion tolerance. IEEE Trans Microwave Theory Tech 2020;68(11):4589-98.

23. Zhang J, Zhang H, Zhou J, et al. Adaptive collision avoidance for multiple UAVs in urban environments. Drones 2023;7(8):491.

24. Yan C, Wang C, Xiang X, et al. Collision-avoiding flocking with multiple fixed-wing UAVs in obstacle-cluttered environments: a taskspecific curriculum-based MADRL approach. IEEE Trans Neural Netw Learn Syst 2024;35(8):10894-908.

25. Estevez J, Nuñez E, Lopez-Guede JM, et al. A low-cost vision system for online reciprocal collision avoidance with UAVs. Aerosp Sci Technol 2024;150:109190-9.

26. Huang S, Zhang H, Huang Z. E2CoPre: energy efficient and cooperative collision avoidance for UAV swarms with trajectory prediction. IEEE Trans Intell Transp Syst 2024;25(7):6951-63.

27. Dong C, Zhang Y, Jia Z, et al. Three-dimension collision-free trajectory planning of UAVs based on ADS-B information in low-altitude urban airspace. Chin J Aeronaut 2025;38(2):103170.

28. Ruseno N, Lin CY. Development of UTM monitoring system based on network Remote ID with inverted teardrop detection algorithm. Unmanned Systems 2025;13(01):105-20.

29. Attin A, Bonnedahl S, Wang Z, et al. Secure Remote ID and detect-andavoid in unmanned aerial systems: modeling the maximum safe speed. IEEE onternational conference on aerospace and signal processing. 2024.

30. Shan G, Roh Bh. Performance model for advanced neighbor discovery process in bluetooth low energy 5.0-enabled Internet of things networks. IEEE Trans Ind Electron 2020;67(12):10965-74.

31. Park W, Eo S, Joo C, et al. B-hop: time-domain adjustment of BLE frequency hopping against Wi-Fi beacon interference. IEEE Internet Things J 2024;11(7):12853-63.

32. Fabra F, Calafate CT, Cano JC, et al. On the impact of inter-UAV com-

munications interference in the 2.4 GHz band. International wireless communications and mobile computing conference. 2017.

33. Fu S, Su Y, Zhang Z, et al. A deep reinforcement learning framework and its implementation for UAV-aided covert communication. Chin J Aeronaut 2025;38(2):103257.

34. Fu S, Feng X, Sultana A, et al. Joint power allocation and 3D deployment for UAV-BSs: A game theory based deep reinforcement learning approach. IEEE Trans Wireless Commun 2024;23(1):736-48.

35. Jia Z, Cao Y, He L, et al. Service function chain dynamic scheduling in space-air-ground integrated networks. IEEE Trans Veh Technol 2025;74(7):11235-48.

36. Liu Y, Yan J, Zhao X. Deep reinforcement learning based latency minimization for mobile rdge computing with virtualization in maritime UAV communication network. IEEE Trans Veh Technol 2022;71(4):4225-36.

37. Ezuma M, Erden F, Kumar Anjinappa C, et al. Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference. IEEE Open J Commun Soc 2020;1:60-76.

38. Nelson WA, Yeduri SR, Jha A, et al. RL-based energy-efficient data transmission over hybrid BLE/LTE/Wi-Fi/LoRa UAV-assisted wireless network. IEEE ACM Trans Netw 2024;32(3):1951-66.

39. Lu Z, Jia Z, Wu Q, et al. Joint trajectory planning and communication design for multiple UAVs in intelligent collaborative air-ground communication systems. IEEE Internet Things J 2024;11(19):31053-67.

40. Yacheur BY, Ahmed T, Mosbah M. Efficient DRL-based selection strategy in hybrid vehicular networks. IEEE Trans Netw Serv Manage 2023;20(3):2400-11.

41. Wei W, Fu S, Tang Y, et al. Joint optimization of UAV aided covert edge computing via a deep reinforcement learning framework. Chin J Aeronaut 2025;38(10);103218.

42. Zu C, Yang C, Wang J, et al. Simulation and field testing of multiple vehicles collision avoidance algorithms. IEEE/CAA J Autom Sin 2020;7(4):1045-63.

43. Woolley M. Bluetooth core specification v5.1. Kirkland: Bluetooth Special Interest Group, 2019. Specification No. 5.1.

44. Dutta P, Culler D. Practical asynchronous neighbor discovery and rendezvous for mobile sensing applications. Proceedings of the 6th ACM conference on embedded network sensor systems. 2008.

45. Karoliny J, Blazek T, Springer A, et al. Predicting the channel access of bluetooth low energy. IEEE International Conference on Communications. 2023.

46. ASTM International. Standard specification for Remote ID and tracking. West Conshohocken: ASTM International, 2022. Report No. F3411-22a.

47. Gil-Martínez A, Poveda-García M, López-Pastor JA, et al. Wi-Fi direction finding with frequency-scanned antenna and channel-hopping scheme. IEEE Sens J 2022;22(6):5210-22.

48. IEEE standard for information technology–telecommunications and information Exchange between systems - local and metropolitan area networks–specific requirements - part 11: wireless LAN medium access control (MAC) and physical layer (PHY) specifications. Piscataway: IEEE Press 2021. Standard No. 80211-2020.

49. Jeon WS, Dwijaksara MH, Jeong DG. Performance analysis of neighbor discovery process in bluetooth low-energy networks. IEEE Trans Veh Technol 2017;66(2):1865-71.

50. Lowe R, Tamar A, Harb J, et al. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems. 2017.

51. Christianos F, Schäfer L, Albrecht S. Shared experience actor-critic for multi-agent reinforcement learning. Advances in neural information processing systems. 2020.

<!-- image-->  
(a) Packet loss at 100 m2

<!-- image-->

<!-- image-->  
(c) Packet loss at 1 000 m2

<!-- image-->  
(d) Packet loss at 3 000 m2

<!-- image-->  
(e) Packet loss at 5 000 m2

<!-- image-->  
(f ) Packet loss at 10 000 m2  
Fig. 6 Average packet loss rates across different spatial sizes using fixed protocol modes.

<!-- image-->  
(a) Overall reward convergence of proposed algorithm and baseline methods

<!-- image-->  
(b) Individual reward convergence for 10 agents using proposed algorithm

<!-- image-->  
10 agents using MADDPG

<!-- image-->  
(d) Individual reward convergence for  
10 agents using IAC  
Fig. 7 Convergence performance of different algorithms.

## Declaration of Interest Statement

☒ The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

☐ The author is an Editorial Board Member/Editor-in-Chief/Associate Editor/Guest Editor for this journal and was not involved in the editorial review or the decision to publish this article.

☐ The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: