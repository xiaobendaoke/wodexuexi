# Reinforcement Learning-Based Optimal Formation Tracking for UAVs With Safety Constraints

Ping Wang , Member, IEEE, Chengpu Yu , Senior Member, IEEE, Fang Deng , Fellow, IEEE, and Jie Chen , Fellow, IEEE

Abstract—This article develops a scheme to tackle the safe optimal formation tracking issue for multiple fixed-wing uncrewed aerial vehicles (UAVs) with external disturbances and asymmetric control constraints. To ensure safety constraints in collision avoidance, a safe set is first constructed by a super level set of a continuously differential function, following a novel control barrier function (CBF) to characterize the safety. Subsequently, we transform the safe optimal formation tracking control into a constrained zero-sum (ZS) differential game to mitigate the destabilizing effects of the disturbances, where the cost function is constructed in a nonquadratic form to cope with asymmetric input constraints. Particularly, the designed CBF is integrated into the cost function to penalize the unsafe behavior, and a damping coefficient is included to balance the optimality and safety. Afterwords, a critic-only reinforcement learning (RL) strategy is developed to learn the robust safe Nash policy, where the critic weights are updated by applying experience replay technology, thus avoiding the requirement for persistence of excitation condition. Moreover, the stability and forward invariance of the safe set of the presented scheme are also verified. Finally, simulation examples are provided to substantiate the validity of the control scheme.

Index Terms—Control barrier function (CBF), control constraints, formation tracking, safe reinforcement learning (RL), uncrewed aerial vehicle (UAV).

## I. INTRODUCTION

HE formation tracking control for multiple uncrewed aerial vehicles (UAVs) has gained significant attention, owing to its promising applications in areas like forest fire monitoring [1], target tracking [2], and aerial transportation [3]. Formation tracking control seeks to drive each UAV to move in the desired geometric shape while tracking a common reference trajectory, allowing for better adaptation to the mission and environment. Numerous effective formation control schemes have emerged, including behavior-based method, virtual structure, and leader–follower strategies [4], [5], [6]. However, the evolving requirements of intelligent UAV systems, as well as high flight speeds and complex aerodynamic characteristics, present new challenges in designing formation controllers with greater autonomy and intelligence.

In recent years, advancements in artificial intelligence technology have inspired innovative exploration in reinforcement learning (RL) for designing control schemes [7]. This approach has not only shown success in addressing unknown features in both linear and nonlinear systems, but also offered optimal control performance. Particularly, it learns optimal behavior by observing how the environment responds to suboptimal control strategies, thus bypassing the closed-form solution of the Hamilton–Jacobi–Bellman (HJB) equation [8], [9]. Given these advantages, RL-based formation control of UAVs has gained increasing attention [10], [11], [12], [13]. However, in order to obtain the optimal behavior strategy in complex high-altitude flying environments, RL agents must explore a sufficient number of states to gain comprehensive experience, which may involve unsafe states [14]. This is clearly inadvisable for safety-critical systems like UAVs, as unsafe strategies may lead to dangerous behaviors such as deviating from the planned route, collisions, or even crashes. Therefore, ensuring that the designed controller has strict safety guarantees is key to the safe and efficient flight of UAVs.

The concept of safety in control systems is typically formalized by specifying the forward invariance of the safe sets [15]. To ensure that learning-enabled systems meet safety while optimizing performance, safe RL (SRL) has emerged in the control systems community [16]. Classical methods of SRL include reachability analysis [17], [18] and control barrier functions (CBFs) [19], [20]. It is particularly emphasized that CBFs offer a Lyapunov-like framework that enables the analysis of set invariance without the requirement to calculate the system’s reachable set, making them more appealing to researchers. Especially in recent years, several interesting studies have successfully combined CBF with model-based RL to develop a safe learning exploration framework. In detail, by integrating CBFs into the value function, an off-policy RL-based safe optimal scheme of nonlinear systems was investigated in [21], by treating the safety a control objective, whereas Liu et al. [22] focused on the safe optimal problems in affine nonlinear discrete-time systems involving state constraints. Addressing the infinite-horizon safe optimal stabilization of nonlinear systems, an online joint learning safe exploration scheme was proposed in [23]. For the second-order integrator multiagent system, the safe optimal formation control was addressed in [24] by combining CBF and off-policy RL without requiring the followers’ dynamics. Additionally, Kokolakis and Vamvoudakis [25] developed an effective finitetime SRL mechanism for pursuit-evasion differential game, achieving both the optimal performance and finite-time capture. From the above research, it is evident that precise definition of safety may differ based on the learning task and is typically represented in various forms, including collision avoidance, full state constraints, and stability assurance.

It is noteworthy that most of the above results in SRL do not account for disturbances and usually focus on single nonlinear affine systems or simple second-order integrator models, with limited research on complex nonlinear systems such as multi-UAV. There are two potential challenges in achieving the safe and optimal performance in multi-UAV formation tracking flight. First, UAV systems are often subject to external disturbances and model uncertainties, which can severely degrade the system’s control performance. More importantly, disturbances may compromise the forward invariance of safety constraints enforced via CBFs, and destabilize the learning process in RL architectures. Static safety-optimal tradeoffs embedded in the cost function may also fail under such conditions, making previously feasible strategies unsafe or ineffective. Second, the actual flight environment is both dynamic and complex, with safety concerns not only related to the flight state constraints of individual UAVs but also involving collision avoidance among the aircrafts. These challenges hinder the applicability of existing SRL methods to complex and diverse multi-UAV formation control. Therefore, how to characterize the various safety of multi-UAV with disturbances during formation tracking as an appropriate safe set, and to ensure its invariance using CBFs within the framework of SRL, are urgent problems to be addressed.

In addition to system states posing safety threats to UAV flight, the input signals/actuators of the UAV are another critical consideration for ensuring safety. Ignoring input safety constraints may lead to excessively rapid or aggressive maneuvers during flight, increasing the risk of loss of control and accidents. Therefore, to enhance the safety and responsiveness of UAVs in complex environments, it is important to consider the safety or physical limitations of input signals/actuators in the design of stable controllers. In recent years, while researchers have proposed many meaningful optimal control schemes [26], [27], [28], [29] using RL strategies for nonlinear systems with input constraints, the development of an effective SRL strategy that integrates CBFs for complex nonlinear systems with both state and input safety requirements is still in the exploratory stage. Currently, only a few results [30] have achieved optimal control of single affine nonlinear systems with asymmetric input constraints using SRL approach, but it is limited to single affine nonlinear systems and does not address the challenges arising from dynamic coupling or largescale multiagent coordination.

Building on the previous considerations, this article aims to developing an optimal safety-oriented formation tracking scheme based on SRL and CBFs for fixed-wing multi-UAV systems with external disturbances and asymmetric input constraints. This framework allows UAVs to effectively avoid collisions while optimizing their control strategies and ensuring that the control constraints remain within a safe range. The following summarizes the main contributions.

1) A novel CBF-based safe learning optimal formation tracking scheme is presented for fixed-wing multi-UAVs, differing from the existing SRL controls [21], [22], [23], [25], [30] that focus solely on individual agent. The concerned safety encompasses both the position staterelated collision avoidance and input-related actuator constraints. To tackle this complexity, a safety set is first constructed with regard to the collision avoidance, which is subsequently characterized by designing a new CBF. Additionally, to manage asymmetric input constraints, a nonquadratic function is incorporated into the cost function, with the designed CBF serving as a penalty, thereby enhancing safety assurance in real flight scenarios.

2) To address the impact of disturbances, the safetyoptimal formation tracking problem is reformulated as a two-player zero-sum (ZS) differential game with safety constraints. Unlike [30], where disturbances are treated as passive uncertainties, our formulation models them as intelligent adversarial agents, enabling the system to learn robust Nash equilibrium strategies. Within this game-theoretic framework, a critic-only neural network (NN) is developed to learn both the value functions and optimal policies. Moreover, to meet the safety requirement of the concerned formation flying, a damping factor is added to the CBF to balance the penalty for unsafe behaviors, which enables the UAVs to yield a desired performance without compromising safety.

3) To circumvent the requirement of persistent excitation (PE) condition, the critic weights are updated via an adaptive learning law enhanced by non-Lipschitz experience replay, which extends the static critic structure in [30] and improves learning convergence in multiagent scenarios. Furthermore, the stability of the closedloop system under the proposed safe optimal formation strategy is rigorously analyzed, ensuring both uniform ultimate boundedness (UUB) and forward invariance of the safe set.

The organization of the article is as follows. Section II describes the safety constraints and control objectives. The novel CBF and safe optimal formation Nash policy are designed in Section III. Section IV introduces the online learning framework by utilizing a critic-only structure, while the stability analysis is arranged in Section V. The validity of the presented control scheme through simulation is shown in Section VI. The conclusion are summarized in Section VII.

Notations: For simplicity, define $z ( t )$ by $z \ \mathrm { o r } \ z ( \cdot )$ without causing confusion.

## II. PROBLEM FORMULATION AND PRELIMINARIES

This section is divided into two parts. First, we introduce the dynamic model and safety requirements of multiple UAVs, followed by the safe optimal formation tracking objective.

## A. Dynamic Model and Safety Constraints of UAVs

For a UAV formation control system built on graph G, the motion model of its ith $( i = 1 , \ldots , N )$ UAV is given by

$$
\begin{array} { l } { { \dot { x } _ { i } = { \nu _ { i } } \cos \theta _ { i } \cos \psi _ { i } + d _ { x i } } } \\ { { \dot { y } _ { i } = { \nu _ { i } } \cos \theta _ { i } \sin \psi _ { i } + d _ { y i } } } \\ { { \dot { z } _ { i } = { \nu _ { i } } \sin \theta _ { i } + d _ { z i } } } \end{array}\tag{1}
$$

where $( x _ { i } , y _ { i } , z _ { i } )$ denotes the position information of the ith , ,UAV in an inertia coordinate frame, $\nu _ { i }$ is the velocity, $\psi _ { i }$ denotes the course angle while $\theta _ { i }$ is the pitch angle, ψand $[ d _ { x i } , d _ { y i } , d _ { z i } ]$ θ represent unknown but bounded disturbance sequences caused by strong winds in different directions. Additionally, each UAV is equipped with an autopilot, and the dynamics of $( \nu _ { i } , \psi _ { i } , \theta _ { i } )$ can be modeled as the following , ψ , θdifferential equation form:

$$
\begin{array} { r l } & { \dot { \nu } _ { i } = \left( \rho _ { \nu } + \Delta \rho _ { \nu } \right) \left( \nu _ { i } ^ { c } - \nu _ { i } \right) } \\ & { \dot { \psi } _ { i } = \left( \rho _ { \psi } + \Delta \rho _ { \psi } \right) \left( \psi _ { i } ^ { c } - \psi _ { i } \right) } \\ & { \dot { \theta } _ { i } = \left( \rho _ { \theta } + \Delta \rho _ { \theta } \right) \left( \theta _ { i } ^ { c } - \theta _ { i } \right) } \end{array}\tag{2}
$$

where $\nu _ { i } ^ { c } , \psi _ { i } ^ { c } ;$ , and $\theta _ { i } ^ { c }$ denote the ith $\mathrm { U A V } ^ { \ , } \mathbf { s }$ input variables: $\Omega _ { i } ^ { u } \ = \ \{ \dot { ( } \dot { \nu } _ { i } ^ { c } , \dot { \psi } _ { i } ^ { c } , \theta _ { i } ^ { c } ) | { b } _ { i } ^ { v } \ \stackrel {  } { \ } \ \le \ \nu _ { i } ^ { c } \ \le \ h _ { i } ^ { \nu } , \ { b } _ { i } ^ { \psi } \ \le \ \psi _ { i } ^ { c } \ \le \ h _ { i } ^ { \psi } , \ { b } _ { i } ^ { \theta } \ \le \ \theta _ { i } ^ { c } \ \le$ $h _ { i } ^ { \theta } , | h _ { i } ^ { * } | \neq | b _ { i } ^ { * } | \}$ θwith $h _ { i } ^ { * }$ and $b _ { i } ^ { * }$ ψ θ being the maximum and mini-,mum bound of the input variables corresponding $\mathbf { \omega } _ { \mathbf { t o } } * = \nu , \psi , \theta .$ The positive constants $\rho _ { \nu } , \rho _ { \psi }$ , and $\rho _ { \theta }$ , ψ, θare the reciprocal of the ρ ρψ ρθtime constants that characterize the velocity-hold loop, course angle-hold loop, and pitch angle-hold loop, respectively. $\Delta \rho _ { \nu }$ $\Delta \rho _ { \psi }$ , and $\Delta \rho _ { \theta }$ ρdenote uncertain but bounded terms, commonly used to address scenarios in practical modeling where $\rho _ { \nu } , \rho _ { \psi }$ and $\rho _ { \theta }$ ρ ρψcannot be accurately measured by standard autopilots. ρθNext, for the concerned formation tracking problem, assume that there exists a virtual leader with the dynamic equation being given as follows:

$$
\begin{array} { r l } & { \dot { x } _ { r } = \nu _ { r } \cos \psi _ { r } \cos \theta _ { r } } \\ & { \dot { y } _ { r } = \nu _ { r } \sin \psi _ { r } \cos \theta _ { r } } \\ & { \dot { z } _ { r } = \nu _ { r } \sin \theta _ { r } } \end{array}\tag{3}
$$

where $( x _ { r } , y _ { r } , z _ { r } )$ denotes the position, and $\nu _ { r } , ~ \psi _ { r } ,$ , and $\theta _ { r } ,$ , , ψ θrespectively stand for the velocity, course angle, and pitch angle that can be preobtained by using the trajectory generator.

For the convenience of symbol definition, let $\begin{array} { r } { \zeta _ { i } ^ { \nu } = \nu _ { i } , } \end{array}$ $\zeta _ { i } ^ { \psi } = \psi _ { i } , \zeta _ { i } ^ { \theta } = \theta _ { i } , \zeta _ { r } ^ { \psi } = \psi _ { r } , \zeta _ { r } ^ { \psi } = \psi _ { r }$ , and $\zeta _ { r } ^ { \theta } \ = \ \theta _ { r }$ . And ζ ψthen, let $\eta _ { i r } = [ \eta _ { i r } ^ { x } , \eta _ { i r } ^ { y } , \eta _ { i r } ^ { z } ] ^ { \mathrm { T } }$ be the expected formation position η η , η , ηof the ith UAV regarding the leader. Define the augmented position variables and the flight states of ith UAV and virtual leader as $\boldsymbol { \eta } _ { i } = [ x _ { i } , y _ { i } , z _ { i } ] ^ { \mathrm { T } } , \boldsymbol { \eta } _ { r } = [ x _ { r } , y _ { r } , z _ { r } ] ^ { \mathrm { T } } , \boldsymbol { \zeta } _ { i } = [ \nu _ { i } , \psi _ { i } , \theta _ { i } ] ^ { \mathrm { T } }$ and $\boldsymbol { \zeta _ { r } } ~ = ~ [ \nu _ { r } , \psi _ { r } , \theta _ { r } ] ^ { \mathrm { T } }$ , , η , , ζ , ψ , θ, respectively. Then, the formation tracking error and the relative flight state of ith UAV can be defined as follows:

$$
e _ { i } = \eta _ { i } - \eta _ { i r } - \eta _ { r } , \quad \Delta \zeta _ { i } = \zeta _ { i } - \zeta _ { r }\tag{4}
$$

where $\boldsymbol { e } _ { i } \triangleq [ e _ { i } ^ { x } , e _ { i } ^ { y } , e _ { i } ^ { z } ] ^ { \mathrm { T } } , \Delta \zeta _ { i } \triangleq [ \Delta \zeta _ { i } ^ { \nu } , \Delta \zeta _ { i } ^ { \psi } , \Delta \zeta _ { i } ^ { \theta } ] ^ { \mathrm { T } } .$

, , ζ ζ , ζ , ζGenerally speaking, the objectives of the formation tracking mainly include two aspects: geometric performance and dynamic performance. Specifically, to maintain the expected geometric pattern and desired cooperative maneuver, it is necessary for each $\mathrm { U A V } ^ { \prime } \mathbf { s }$ tracking error $e _ { i }$ and relative flight state $\Delta \zeta _ { i }$ to converge to the origin. However, during the actual ζformation process, designing a controller that can drive the formation error of the UAV system to converge to the origin may result in excessive control investment and cost. In fact, for practical UAV application systems, it is feasible to make the UAVs approaching the expected formation tracking, such that both the position error $e _ { i } ^ { o }$ and relative flight state $\Delta \zeta _ { i }$ enter ζa predetermined range (but not converge to the origin).

Additionally, note that in the formation tracking process of practical applications, to avoid collisions between UAVs, the formation position errors $e _ { i } ^ { o }$ with $o \ : = \ : x , y , z$ are usually constrained in a safe set. For this reason, the concerned position errors $e _ { i } ^ { o }$ are imposed the following safety constraints:

$$
- k _ { i e } ^ { o } \left( t \right) < e _ { i } ^ { o } < k _ { i e } ^ { o } \left( t \right) , \quad o = x , y , z\tag{5}
$$

where $k _ { i e } ^ { o } ( t )$ is a time-varying function bound determined by neighboring UAVs’ positions information, ensuring inter-agent coordination and collision avoidance.

Remark 1: It is important to emphasize that achieving collision avoidance between UAVs by constraining the position tracking error $e _ { i } ^ { o }$ is a feasible approach. Specifically, in multi-UAVs formation control, the most intuitive strategy to preventing collisions is to ensure that the distances between UAVs should be larger than a predetermined minimum safe distance. Mathematically, this is done by limiting the relative position error between each pair of UAVs. Note that since the relative displacement between the ith UAV and the jth UAV can be represented as $\Delta \eta _ { i j } = \eta _ { i } - \eta _ { j } = ( \eta _ { i } - \eta _ { r } ) - ( \eta _ { j } - \eta _ { r } ) =$ $( e _ { i } + \eta _ { i r } ) - ( e _ { j } + \eta _ { j r } )$ , which means that the relative position error η ηbetween the two UAVs can be transmitted to their respective position tracking errors. By the reverse triangle inequality, the following inequality can be derived:

$$
\begin{array} { r l r } { \| \Delta \eta _ { i j } \| \ge } & { { } } & { \| \eta _ { i r } - \eta _ { j r } \| - \| e _ { i } - e _ { j } \| \ge \| \eta _ { i r } - \eta _ { j r } \| - \| e _ { i } \| - \| e _ { j } \| } \end{array}
$$

where $\| e _ { i } - e _ { j } \| \leq \| e _ { i } \| + \| e _ { j } \|$ is used in the last line. Therefore, the collision avoidance requirement $\Vert \eta _ { i } - \eta _ { j } \Vert \geq d _ { \operatorname* { m i n } }$ with $d _ { \operatorname* { m i n } } \ > \ 0$ being the minimum safe distance, is satisfied if $\| { \eta } _ { i r } - { \eta } _ { j r } \| - \| e _ { i } \| - \| e _ { j } \| > d _ { \operatorname* { m i n } }$ . This further means that

$$
\| e _ { i } \| + \| e _ { j } \| \leq \| \eta _ { i r } - \eta _ { j r } \| - d _ { \operatorname* { m i n } } .\tag{6}
$$

Note that $| e _ { i } ^ { o } | < k _ { i e } ^ { o }$ with $o = x , y , z$ is defined in (5). Then, < , ,the norm bound can be obtained as follows:

$$
\begin{array} { r } { \| e _ { i } \| \le \sqrt { \left( k _ { i e } ^ { x } \right) ^ { 2 } + \left( k _ { i e } ^ { y } \right) ^ { 2 } + \left( k _ { i e } ^ { z } \right) ^ { 2 } } \triangleq k _ { i } . } \end{array}\tag{7}
$$

Substituting inequality (7) into (6) yields

$$
k _ { i } + k _ { j } \le \vert \vert \eta _ { i r } - \eta _ { j r } \vert \vert - d _ { \mathrm { { m i n } } } .\tag{8}
$$

As a special case, one can choose $k _ { i } ~ = ~ k _ { j }$ j, leading to the symmetric allocation

$$
k _ { i } \leq \frac { 1 } { 2 } \left( \| \eta _ { i r } - \eta _ { j r } \| - d _ { \operatorname* { m i n } } \right) .\tag{9}
$$

We further note that, while the theoretical bound (9) involves the desired relative positions $\| \eta _ { i r } - \eta _ { j r } \| .$ , in practice it is η ηoften more convenient to evaluate the condition using the instantaneous relative distance $\| \eta _ { i } - \eta _ { j } \|$ . That is, in practice, we can choose the norm bound $k _ { i }$ ηas follows:

$$
\begin{array} { l } { \displaystyle { k _ { i } \le \frac { 1 } { 2 } \left( \| \eta _ { i } - \eta _ { j } \| - d _ { \operatorname* { m i n } } \right) } } \\ { \displaystyle { ~ = \frac { 1 } { 2 } \left( \sqrt { \left( x _ { i } - x _ { j } \right) ^ { 2 } + \left( y _ { i } - y _ { j } \right) ^ { 2 } + \left( z _ { i } - z _ { j } \right) ^ { 2 } } - d _ { \operatorname* { m i n } } \right) } } \\ { \displaystyle { < \frac { 1 } { 2 } \left( \sqrt { \left( x _ { i } - x _ { j } \right) ^ { 2 } + \left( y _ { i } - y _ { j } \right) ^ { 2 } + \left( z _ { i } - z _ { j } \right) ^ { 2 } } \right) . } } \end{array}\tag{10}
$$

This replacement provides a conservative and implementable rule that dynamically adjusts according to the real-time spacing of neighboring UAVs. Hence, the constraint (5) shows that bounding individual tracking errors indirectly ensures inter-UAV distance safety.

## B. Problem Formulation

Following the above preparation, let the augmented vector related to the error be $\bar { X _ { i } } = \bar { [ e _ { i } ^ { \mathrm { T } } , \Delta \zeta _ { i } ^ { \mathrm { T } } ] ^ { \mathrm { T } } }$ and the augmented input vector be $u _ { i } = [ \nu _ { i } ^ { c } , \psi _ { i } ^ { c } , \theta _ { i } ^ { c } ] ^ { \mathrm { T } }$ , ζ. Afterward, combining the dynamic , ψ , θ(1)–(3) and the definition in equation (4), we can derive the time derivative of augmented error variable $X _ { i }$ satisfies

$$
\dot { X } _ { i } = F _ { i } + G _ { i } u _ { i } + d _ { i } , \ X _ { i } \left( 0 \right) = X _ { i } ^ { 0 }\tag{11}
$$

where the nonlinear function term $F _ { i } ,$ the input matrix $G _ { i } ,$ and the augmented disturbance vector $d _ { i }$ are defined as follows:

$$
\begin{array} { r l } & { \boldsymbol { F } _ { i } = \left[ \begin{array} { c c } { \nu _ { i } \cos { \psi _ { i } } \cos { \theta _ { i } } - \nu _ { r } \cos { \psi _ { r } } \cos { \theta _ { r } } - \vec { \eta } _ { i r } ^ { x } } \\ { \nu _ { i } \sin { \psi _ { i } } \cos { \theta _ { i } } - \nu _ { r } \sin { \psi _ { r } } \cos { \theta _ { r } } - \vec { \eta } _ { i r } ^ { x } } \\ { \nu _ { i } \sin { \theta _ { i } } - \nu _ { r } \sin { \theta _ { r } } - \vec { \eta } _ { i r } ^ { z } } \\ { - \rho _ { \nu } \nu _ { i } - \nu _ { r } } \\ { - \rho _ { \mu \nu } \psi _ { i } - \vec { \psi } _ { r } } \\ { - \rho _ { \theta } \theta _ { i } - \vec { \theta } _ { r } } \end{array} \right] } \\ & { \boldsymbol { G } _ { i } = \left[ \begin{array} { c c } { 0 _ { 3 \times 3 } } \\ { \boldsymbol { \theta } _ { 1 } \sin { \psi _ { i } } \cos { \theta _ { i } } - \nu _ { r } \sin { \psi _ { r } } \cos { \theta _ { r } } - \vec { \eta } _ { i r } ^ { x } } \\ { \boldsymbol { \theta } _ { 2 } \sin { \psi _ { r } } \cos { \theta _ { r } } } \\ { \boldsymbol { \theta } _ { 3 } \cos { \psi _ { r } } } \end{array} \right] } \end{array}
$$

and $d _ { i } = [ d _ { x i } , d _ { y i } , d _ { z i } , d _ { \nu i } , d _ { \psi i } , d _ { \theta i } ]$ with $d _ { \nu i } = \Delta \rho _ { \nu } ( \nu _ { i } ^ { c } - \nu _ { i } ) , d _ { \psi i } =$ $\Delta \rho _ { \psi } ( \psi _ { i } ^ { c } - \psi _ { i } )$ , ,, and $d _ { \theta i } = \Delta \rho _ { \theta } ( \theta _ { i } ^ { \theta } - \theta _ { i } )$ ρ ψ. Due to the limitations ρψ ψ ψ θ ρθ θ θof actuators and the need for nonlinear analysis, system (11) needs to meet the following standard assumption.

Assumption 1: 1) The nonlinear term $F _ { i }$ is Lipschitz continuous with $F _ { i } ( 0 ) = 0$ on a compact set $\mathcal { X } _ { i }$ including the origin, that is, for $X _ { i } \in \mathcal { X } _ { i } , \| F _ { i } \| \leq b _ { F } \| X _ { i } \|$ holds, where $b _ { F } > 0$ is a constant and 2) the input matrix $G _ { i }$ is also bounded on $\mathcal { X } _ { i }$ by a constant $b _ { G } > 0$ , that is, $\| G _ { i } \| \leq b _ { G }$ 

>Note that inequality (5) can be expressed as the set

$$
\begin{array} { r } { \theta _ { i } = \left\{ X _ { i } \in \mathcal { X } _ { i } \vert - k _ { i e } ^ { o } < e _ { i } ^ { o } < k _ { i e } ^ { o } , ~ o = x , y , z \right\} . } \end{array}\tag{12}
$$

Following this, for ith UAV, introduce a superlevel set of a continuously differentiable function $h _ { i } ( X _ { i } )$ to further characterize the safety error requirements:

$$
S _ { i } \triangleq \{ X _ { i } \in R ^ { n } : h _ { i } \left( X _ { i } \right) \geq 0 \}
$$

$$
\begin{array} { r l } & { \partial S _ { i } \triangleq \{ X _ { i } \in R ^ { n } : h _ { i } \left( X _ { i } \right) = 0 \} } \\ & { \operatorname { I n t } \left( S _ { i } \right) \triangleq \{ X _ { i } \in R ^ { n } : h _ { i } \left( X _ { i } \right) > 0 \} } \end{array}
$$

where $\partial { \cal { S } } _ { i }$ denote the boundary of $S _ { i } , \operatorname { I n t } ( S _ { i } )$ is the interior of ${ \mathbf { } } S _ { i } ,$ and $\boldsymbol { S _ { i } }$ is considered as the safe set of ith UAV system.

Problem 1: For multi-UAV systems (1)–(2), with control constraints (2) and safety guarantees (5), the control objective is to construct a proper positive definite performance function $L _ { i 1 } ( X _ { i } , u _ { i } )$ and find a safe optimal policy $u _ { i } ^ { * }$ for each augmented ,system (11) such that.

1) The position and attitude of each UAV can converge to the expected formation References.

2) The position tracking error $e _ { i }$ remains within the safety prescribed boundaries during the entire formation operation.

3) The augmented system (11) has a L2-gain less than or equal to  for any $X _ { i } \in S _ { i }$ , that is,

$$
\int _ { t } ^ { \infty } L _ { i 1 } \left( X _ { i } \left( \tau \right) , u _ { i } \left( \tau \right) \right) d \tau \leq \delta ^ { 2 } \int _ { t } ^ { \infty } \| d _ { i } \left( \tau \right) \| ^ { 2 } d \tau .\tag{13}
$$

As can be seen, the objective of Problem 1 consists of three main elements: disturbance attenuation, input constraints, and safety constraints on formation tracking position error. To address these issues, a ZS differential game framework with both state and input constraints will be introduced. For each augmented dynamics (11), considering the safe controller $u _ { i }$ as the minimizing player and disturbance policy $d _ { i }$ as the maximizing one, the safe optimal formation tracking issue with control and safety constraints can be mathematically described by the following two-player ZS game:

$$
\begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { u _ { i } ( \cdot ) } \operatorname* { m a x } _ { d _ { i } ( \cdot ) } \ \int _ { 0 } ^ { \infty } L _ { i } \left( X _ { i } , u _ { i } , d _ { i } \right) d t } \\ { \mathrm { s u b j e c t ~ t o ~ t h e ~ s y s t e m ~ } ( 1 1 ) , \ X _ { i } ^ { 0 } \in \mathscr { S } _ { i } } \\ { \displaystyle u _ { i } ( \cdot ) \in \Omega _ { i } ^ { u } , \ X _ { i } \left( t \right) \in \mathscr { S } _ { i } , \ t \geq 0 , \ i = 1 , 2 , . . . , N } \end{array}\tag{14}
$$

with the reward function $L _ { i } ( X _ { i } , u _ { i } , d _ { i } )$ being constructed as follows:

$$
\begin{array} { r l } & { L _ { i } \left( X _ { i } , u _ { i } , d _ { i } \right) = L _ { i 1 } \left( X _ { i } , u _ { i } \right) - \delta ^ { 2 } d _ { i } ^ { T } d _ { i } } \\ & { L _ { i 1 } \left( X _ { i } , u _ { i } \right) = X _ { i } ^ { T } R _ { i } X _ { i } + \mathcal { U } _ { i } \left( u _ { i } \right) } \end{array}\tag{15}
$$

where the matrix $R _ { i } ~ \in ~ R ^ { 3 \times 3 }$ meeting $R _ { i } \ = \ R _ { i } ^ { \mathrm { T } } \ > \ 0 .$ , and $\mathcal { U } _ { i } ( u _ { i } ) \in R$ >is designed as the following nonquadratic positive definite integrand penalty function to handle the asymmetric input constraints:

$$
\begin{array} { r } { \mathcal { U } _ { i } \left( u _ { i } \right) = \displaystyle \sum _ { s = \nu , \psi , \theta } 2 \alpha _ { i } ^ { s } \int _ { \beta _ { i } ^ { s } } ^ { u _ { i } ^ { s } } \mathrm { t a n h } ^ { - 1 } \left( \frac { \tau _ { i } ^ { s } - \beta _ { i } ^ { s } } { \alpha _ { i } ^ { s } } \right) d \tau _ { i } ^ { s } } \\ { = \displaystyle \sum _ { s = \nu , \psi , \theta } 2 \alpha _ { i } ^ { s } \left( u _ { i } ^ { s } - \beta _ { i } ^ { s } \right) \mathrm { t a n h } ^ { - 1 } \left( \frac { u _ { i } ^ { s } - \beta _ { i } ^ { s } } { \alpha _ { i } ^ { s } } \right) } \\ { + \displaystyle \sum _ { s = \nu , \psi , \theta } \left( \alpha _ { i } ^ { s } \right) ^ { 2 } \mathrm { l n } \left( 1 - \left( \frac { u _ { i } ^ { s } - \beta _ { i } ^ { s } } { \alpha _ { i } ^ { s } } \right) ^ { 2 } \right) } \end{array}\tag{16}
$$

with $u _ { i } ^ { \nu } = \nu _ { i } ^ { c } , u _ { i } ^ { \psi } = \psi _ { i } ^ { c } , u _ { i } ^ { \theta } = \theta _ { i } ^ { c }$ , and

$$
\alpha _ { i } ^ { s } = \frac { h _ { i } ^ { s } - b _ { i } ^ { s } } { 2 } , \beta _ { i } ^ { s } = \frac { h _ { i } ^ { s } + b _ { i } ^ { s } } { 2 } .
$$

Remark 2: The two-player ZS game models (14) the controller as the minimizing player ensuring safety, while the disturbance acts as the maximizing player driving the system toward unsafe states. Such a formulation captures the adversarial nature of disturbances and ensures that the learned strategies are robust under worst-case conditions. Compared with conventional methods that treat disturbances as passive uncertainties, the ZS game framework explicitly models the dynamic interaction between control and disturbance, and the resulting Nash equilibrium guarantees both safety and performance in multi-UAV applications.

Remark 3: In our setting, collision avoidance is indirectly enforced through position tracking error bounds relative to the desired formation, which guarantees that inter-UAV distances remain above a safety threshold. These constraints are nonconvex and state-dependent, potentially leading to multiple local optima and nonsmooth gradients that increase optimization difficulty. Since these constraints couple each UAV with others, the number of constraints grows quadratically with the agent count, expanding the state space. In addition, the control constraints are also addressed to enhance the safety of UAVs, which different from the existing SRL results [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25] without considering input constraints. Under this case, how to develop a safe learning optimal formation framework to solve the ZS game strategy is one of the difficulties of this article. Furthermore, as mentioned in [26], to ensure the boundedness of the integral $\begin{array} { r } { \int _ { 0 } ^ { \infty } L _ { i } ( X _ { i } , u _ { i } , d _ { i } ) d t } \end{array}$ , the control $u _ { i }$ in this article is admissible , ,with respect to (14) on a set $\Omega _ { i } ^ { u }$

## III. SAFE OPTIMAL FORMATION TRACKING POLICY

## A. Barrier-Function-Based ZS Game

Due to the difficulty of finding a feedback solution for differential game (14) with safety constraints, this section will transform (14) into an unconstrained optimization problem using CBF method. The main idea is to formalize the control system’s safety by maintaining the safe set’s forward invariance [31]. Specially, the safe set $S _ { i }$ is forward invariant regarding the dynamics (11) if, for every $X _ { i } ( 0 ) \ \in \ S _ { i } ,$ , the solution $X _ { i } ( t ) \in { \mathcal { S } } _ { i }$ for all $t \geq 0$ . In this case, the dynamics (11) is called safe regarding $S _ { i } .$ . From the safe set (12), it can be seen that the safety domain is described by an asymmetric time-varying function. Hence, the following CBF $B _ { i } ^ { c } ( X _ { i } ) : S _ { i }  R$ is constructed for the ith UAV:

$$
B _ { i } ^ { c } \left( X _ { i } \right) = \sum _ { o \in \{ x , y , z \} } \frac { \left( k _ { i e } ^ { o } e _ { i } ^ { o } \right) ^ { 2 } } { \left( k _ { i e } ^ { o } + e _ { i } ^ { o } \right) \left( k _ { i e } ^ { o } - e _ { i } ^ { o } \right) } .\tag{17}
$$

Note that $B _ { i } ^ { c } ( X _ { i } )$ is a continuous differentiable function on the set Int(Si), which possesses the following properties:

$$
B _ { i } ^ { c } ( 0 ) = 0 , \operatorname* { i n f } _ { X _ { i } \in \mathrm { I n t } ( S _ { i } ) } B _ { i } ^ { c } ( X _ { i } ) > 0 , \operatorname* { l i m } _ { X _ { i }  \partial S _ { i } } B _ { i } ^ { c } ( X _ { i } ) = \infty .
$$

Subsequently, integrating the designed barrier function $B _ { i } ^ { c } ( X _ { i } )$ into the reward $L _ { i } ( X _ { i } , u _ { i } , d _ { i } )$ defined in (15) yields the augmented form

$$
L _ { i } ^ { c } \left( X _ { i } , u _ { i } , d _ { i } \right) = L _ { i 1 } \left( X _ { i } , u _ { i } \right) + \kappa _ { i } B _ { i } ^ { c } \left( X _ { i } \right) - \delta ^ { 2 } d _ { i } ^ { T } d _ { i }\tag{18}
$$

where $\kappa _ { i } > 0$ is a damping constant that enables a compromise κ >between optimality and safety by deciding the degree of influence the CBF $B _ { i } ^ { c } ( \cdot )$ (safety) to the reward function $L _ { i } ^ { c } ( \cdot )$ (optimality). Then, define the augmented cost function with the safe awareness as follows:

$$
J _ { i } ^ { s } \left( X _ { i } ^ { 0 } , u _ { i } , d _ { i } \right) = \int _ { 0 } ^ { \infty } L _ { i } ^ { c } \left( X _ { i } , u _ { i } , d _ { i } \right) d t .\tag{19}
$$

Then, we can transform the safety-constrained ZS game (14) into the following unconstrained safety-aware ZS game:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { u _ { i } } \operatorname* { m a x } _ { d _ { i } } \ \int _ { 0 } ^ { \infty } L _ { i } ^ { c } \left( X _ { i } , u _ { i } , d _ { i } \right) d t } \\ { \displaystyle \mathrm { s u b j e c t ~ t o ~ t h e ~ s y s t e m ~ } ( 1 1 ) , \quad X _ { i } ^ { 0 } \in \mathscr { S } _ { i } . } \end{array}\tag{20}
$$

Remark 4: In this article, we construct a new CBF $B _ { \mathrm { i o } } ^ { c } ( e _ { i } ^ { o } )$ in the form of (17) to characterize system safety, which may not be unique. Normally, for a CBF $B ( x )$ that depict the safe set S of variable x, an alternative construction form is $B ( x ) =$ (g(x) h(x)), where $h ( x ) > 0$ denotes the safe set function, and /g(x) satisfies: 1) $g ( 0 ) = 0 ; 2 ) \ g ( x ) > 0 , \ x \in \mathcal { S } \ \backslash$ 0; and 3) $\operatorname { s u p } _ { x \in S } g ( x )$ exists.

Remark 5: Note that the damping gain $\kappa _ { i }$ is introduced κin (18) that penalizes the violation of the safety requirement encoded by a barrier function $B _ { i } ^ { c } ( \cdot )$ . A larger $\kappa _ { i }$ increases the κpenalty as the UAV approaches the safety boundary, leading the controller to prioritize safety through more conservative maneuvers. Conversely, a smaller $\kappa _ { i }$ reduces the relative weight κof the barrier term, thereby shifting the control strategy toward task-related optimality. Additionally, in a few results [32] and [33], such gain has also been interpreted as analogous to Lagrange multipliers. While both of them aim to incorporate safety into the optimization objective, their mechanisms differ. The Lagrange multiplier is typically co-optimized with the control input and has a well-defined analytical interpretation as the sensitivity of cost to constraint violation. In contrast, the gain $\kappa _ { i }$ used to weight barrier function is empirically κselected, serving as a tunable parameter that reflects the tradeoff between performance and safety. This heuristic gain does not have a closed-form solution and is typically adjusted based on safety criticality and task requirements.

## B. Derivation of Safe Optimal Nash Equilibrium

According to the above preparation, define the optimal safe Nash value function for the ith UAV as follows:

$$
V _ { i c } ^ { * } \left( X _ { i } \right) \triangleq \operatorname* { m i n } _ { u _ { i } } \operatorname* { m a x } _ { d _ { i } } \int _ { 0 } ^ { \infty } L _ { i } ^ { c } \left( X _ { i } , u _ { i } , d _ { i } \right) d t\tag{21}
$$

and then, it meets the Hamilton–Jacobi–Isaacs (HJI) equation

$$
0 = \operatorname* { m i n } _ { u _ { i } } \operatorname* { m a x } _ { d _ { i } } H _ { i } \left( X _ { i } , \nabla V _ { i c } ^ { * } , u _ { i } , d _ { i } \right) , X _ { i } \in \mathcal { S } _ { i }\tag{22}
$$

where the function $H _ { i } ( X _ { i } , \triangledown V _ { i c } ^ { * } , u _ { i } , d _ { i } )$ denotes the correspond-, , ,ing Hamiltonian function derived via Bellman’s optimality principle with the following definition:

$$
\begin{array} { r l } & { H _ { i } \left( X _ { i } , \nabla { V } _ { i c } ^ { * } , u _ { i } , d _ { i } \right) = L _ { i } ^ { c } ( \cdot ) + \nabla { V } _ { i c } ^ { * } \left( F _ { i } + G _ { i } u _ { i } + d _ { i } \right) } \\ & { \quad \quad \quad = X _ { i } ^ { T } R _ { i } X _ { i } + \mathcal { U } _ { i } \left( u _ { i } \right) - \delta ^ { 2 } d _ { i } ^ { T } d _ { i } } \\ & { \quad \quad \quad \quad + \kappa _ { i } B _ { i } ^ { c } + \nabla { V } _ { i c } ^ { * } \left( F _ { i } + G _ { i } u _ { i } + d _ { i } \right) } \end{array}\tag{23}
$$

with $\nabla V _ { i c } = \partial V _ { i c } / \partial X _ { i } \in R ^ { 1 \times 6 }$ being the gradient row vector, ∂ /∂and the remaining parts of this article having similar definitions.

Note that if the Nash condition

$$
\begin{array} { r l } {  { \operatorname* { m i n } _ { u _ { i } } \operatorname* { m a x } _ { d _ { i } } \int _ { 0 } ^ { \infty } L _ { i } ^ { c } ( X _ { i } , u _ { i } , d _ { i } ) d t } } \\ & { \qquad = \operatorname* { m a x } _ { d _ { i } } \operatorname* { m i n } _ { u _ { i } } \int _ { 0 } ^ { \infty } L _ { i } ^ { c } ( X _ { i } , u _ { i } , d _ { i } ) d t } \end{array}\tag{24}
$$

holds, this ZS game (20) has a unique solution. Using the stationary condition

$$
\frac { \partial H _ { i } \left( X _ { i } , \nabla V _ { i c } ^ { * } , u _ { i } , d _ { i } \right) } { \partial u _ { i } } = 0 , \quad \frac { \partial H _ { i } \left( X _ { i } , \nabla V _ { i c } ^ { * } , u _ { i } , d _ { i } \right) } { \partial d _ { i } } = 0
$$

proposed in [34] for optimality, the optimal feedback Nash strategies $( u _ { i } ^ { * } , d _ { i } ^ { * } )$ for the above ZS game are

$$
u _ { i } ^ { * } \left( X _ { i } \right) = - \alpha _ { i } \operatorname { t a n h } { \left( 0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \mathrm { T } } \nabla V _ { i c } ^ { * \mathrm { T } } \right) } + \bar { \beta } _ { i }\tag{25}
$$

$$
d _ { i } ^ { * } \left( X _ { i } \right) = \frac { 1 } { 2 \delta ^ { 2 } } \nabla V _ { i c } ^ { * \mathrm { T } }\tag{26}
$$

where $\alpha _ { i } = \mathrm { d i a g } \{ \alpha _ { i } ^ { \nu } , \alpha _ { i } ^ { \psi } , \alpha _ { i } ^ { \theta } \}$ and $\bar { \boldsymbol { \beta } } _ { i } = [ \beta _ { i } ^ { \nu } , \beta _ { i } ^ { \psi } , \beta _ { i } ^ { \theta } ] ^ { \mathrm { T } }$

α α , α , α β β , β , βSubsequently, by substituting (25) and (26) into (23), the HJI equation becomes

$$
\begin{array} { r l } & { 0 = X _ { i } ^ { \mathrm { T } } R _ { i } X _ { i } + \mathcal { U } _ { i } \left( - \alpha _ { i } \operatorname { t a n h } { \left( 0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \mathrm { T } } \nabla V _ { i c } ^ { * \mathrm { T } } \right) } + \bar { \beta } _ { i } \right) } \\ & { \phantom { 0 = } + \kappa _ { i } B _ { i } ^ { c } + \frac { 1 } { 4 \delta _ { i } ^ { 2 } } \nabla V _ { i c } ^ { * } \nabla V _ { i c } ^ { * \mathrm { T } } + \nabla V _ { i c } ^ { * } F _ { i } } \\ & { \phantom { 0 = } + \nabla V _ { i c } ^ { * } G _ { i } \bar { \beta } _ { i } - \nabla V _ { i c } ^ { * } G _ { i } \alpha _ { i } \operatorname { t a n h } { \left( 0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \mathrm { T } } \nabla V _ { i c } ^ { * \mathrm { T } } \right) } . } \end{array}\tag{27}
$$

Next, we provide a theorem to show that the safety and stability of dynamics (11) can be assured by the optimal strategy $( u _ { i } ^ { * } , d _ { i } ^ { * } )$ associated with the performance function (19). ,Its proof is included in the Appendix.

Theorem 1: Consider the augmented dynamics (11) with the performance index (19). When the condition

$$
\operatorname* { s u p } _ { X _ { i } \in S _ { i } } \left| L _ { i } \left( X _ { i } , u _ { i } ^ { * } ( \cdot ) , d _ { i } ^ { * } ( \cdot ) \right) \right| < \infty\tag{28}
$$

is satisfied, the optimal Nash policy given in (25)–(26) guarantees the closed-loop system’s stability in terms of UUB and the safe set $\boldsymbol { S _ { i } } ^ { \prime } \boldsymbol { \mathrm { s } }$ forward invariance. 

## IV. SAFE ONLINE CRITIC-BARRIER LEARNING FOROPTIMAL NASH STRATEGY

Note that solving the HJI (27), can yield the optimal value $\nabla V _ { i c } ^ { * } ,$ , enabling derivation of optimal feedback strategies (25) and (26). However, due to its nonlinear nature, deriving an analytical solution for the HJI equation is often difficult. Therefore, this section will propose an online RL strategy based solely on the critic to approximate the optimal feedback solution, which is summarized in Algorithm 1.

## A. Value Function Approximation

Utilizing the Weier strass approximation theorem, introduce a NN containing a sufficient set of basis functions to approximate the value function $V _ { i c } ^ { * } ( X _ { i } )$ on a compact set $X _ { i } \subset S _ { i }$

$$
V _ { i c } ^ { * } \left( X _ { i } \right) = w _ { i } ^ { * \mathrm { { T } } } \phi _ { i } \left( X _ { i } \right) + \bar { B } _ { i } ^ { c } \left( X _ { i } \right) + \epsilon _ { i } \left( X _ { i } \right)\tag{29}
$$

```perl
Algorithm 1 Safe Online RL Algorithm
Input: Initial states (x ${ \mathrm { \Omega } } _ { i } ( 0 ) , { \mathrm { \Omega } } y _ { i } ( 0 ) , z _ { i } ( 0 ) )$ and
$( v _ { i } ( 0 ) , \psi _ { i } ( 0 ) , \theta _ { i } ( 0 ) ) ;$ critic weights $\hat { w } _ { i } ( 0 ) ;$ ; safety
constraint boundary $k _ { i e } ^ { o }$ with $o = x , y , z ;$ input
constraint $b _ { i } ^ { * }$ and $h _ { i } ^ { * }$ with $* = v , \psi , \theta ;$
parameters $\kappa _ { i } ;$ ：learningrate $\sigma _ { i w }$
Output: Approximate safe Nash strategies $\hat { u } _ { i }$ in (36),
$\hat { d } _ { i } )$ in (37)
Initialize critic network weights $\hat { w } _ { i }$ ,and set initial
time $t = 0 ;$
while the UAVs have not reached the desired
formation do
Observe the current system state $X _ { i } ( t ) ;$
Update the critic $\hat { W }$ (value function approximation)
using the learning law (42);
Compute the approximate saft Nash policies $( \hat { u } _ { i } .$
${ \hat { d } } _ { i } ) ;$
Apply the computed Nash policies $( \hat { u } _ { i } , \hat { d } _ { i } )$ to the
agent system;
Interact with the environment and observe the next
state;
Record the elapsed time $t ;$
Store the experience data into the replay buffer;
Return: Approximate safe Nash policies $( \hat { u } _ { i } , \hat { d } _ { i } ) ;$
```

where $w _ { i } ^ { * } \in R ^ { m }$ denotes an ideal but unknown constant weight vector, and $\phi _ { i } = [ \phi _ { i 1 } , \ldots , \phi _ { \mathrm { i m } } ] \in R ^ { m }$ denotes the vector of basis φ φ , . . . , φfunction. Moreover, to better reflect safety constraints during the learning process, a numerically regularized form

$$
\bar { B } _ { i } ^ { c } \left( X _ { i } \right) = \kappa _ { i } \sum _ { o \in \{ x , y , z \} } \frac { \left( k _ { i e } ^ { o } e _ { i } ^ { o } \right) ^ { 2 } } { \left( k _ { i e } ^ { o } + e _ { i } ^ { o } \right) \left( k _ { i e } ^ { o } - e _ { i } ^ { o } \right) + \sigma _ { i } }\tag{30}
$$

is used in implementation to evaluate the CBF defined in (17) and is incorporated into the approximation function (29), while $\epsilon _ { i } ( X _ { i } )$ represents the reconstruction approximation error, encompassing the term $- \bar { B } _ { i } ^ { c }$ , where $\sigma _ { i } > 0$ is a small constant to avoid singularity. Based on the approximate (29), we can get that

$$
\begin{array} { r } { \nabla V _ { i c } ^ { * } \left( X _ { i } \right) = w _ { i } ^ { * \mathrm { T } } \nabla \phi \left( X _ { i } \right) + \nabla \bar { B } _ { i } ^ { c } \left( X _ { i } \right) + \nabla \epsilon _ { i } \left( X _ { i } \right) . } \end{array}\tag{31}
$$

Due to the continuity of $\phi _ { i }$ and $\nabla \phi _ { i }$ and their gradients with respect to $X _ { i }$ φ φdefined in the compact set $\mathcal { X } _ { i } ,$ , some common conditions in NN-based approximation are provided in the following assumption.

Assumption 2: 1) There are positive constants $w _ { i d } , \phi _ { i d } ,$ and $\phi _ { i d } ^ { d }$ that make $\lVert w _ { i } ^ { * } \rVert \leq w _ { i d } , \lVert \phi _ { i } \rVert \leq \phi _ { i d } .$ , and $\lVert \nabla \phi _ { i } \rVert \leq \phi _ { i d } ^ { d }$ and 2) the augmented approximation error $\epsilon _ { i } ( X _ { i } )$ and its gradient are bounded locally by two constants $\epsilon _ { i d } > 0$ and $\epsilon _ { i d } ^ { d } > 0$ , that is, $\| \epsilon _ { i } ( X _ { i } ) \| \le \epsilon _ { i d }$ and $\lVert \nabla \epsilon _ { i } \rVert \leq \epsilon _ { i d } ^ { d }$ 

  It follows from (31) that the safe optimal feedback policies $u _ { i } ^ { * }$ and $d _ { i } ^ { * }$ can be parameterized as:

$$
u _ { i } ^ { * } = - \alpha _ { i } \operatorname { t a n h } { ( \Lambda _ { i } ) } + \epsilon _ { i u ^ { * } } \left( X _ { i } \right) + \bar { \beta } _ { i }\tag{32}
$$

$$
d _ { i } ^ { * } = \frac { 1 } { 2 { \delta ^ { 2 } } } \left[ { w _ { i } ^ { * } \mathrm { { T } } \nabla \phi _ { i } + \nabla \bar { B } _ { i } ^ { c } \left( X _ { i } \right) + \nabla \epsilon _ { i } } \right] ^ { \mathrm { T } }\tag{33}
$$

where

$$
\begin{array} { r l } & { \boldsymbol { \Lambda } _ { i } = 0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \Gamma } \left( \boldsymbol { w } _ { i } ^ { * \Gamma } \nabla \phi _ { i } + \nabla \bar { B } _ { i } ^ { c } \left( \boldsymbol { X } _ { i } \right) \right) ^ { \mathrm { T } } } \\ & { \epsilon _ { i u ^ { * } } \left( \boldsymbol { X } _ { i } \right) = - \cfrac { 1 } { 2 } \left( I _ { 3 } - \Phi _ { i } \left( \boldsymbol { \varsigma } _ { i } \left( \boldsymbol { X } _ { i } \right) \right) \right) G _ { i } ^ { \mathrm { T } } \nabla \epsilon _ { i } \left( \boldsymbol { X } _ { i } \right) } \\ & { \Phi _ { i } \left( \boldsymbol { \varsigma } _ { i } \left( \boldsymbol { X } _ { i } \right) \right) = \operatorname { d i a g } \left\{ \operatorname { t a n h } ^ { 2 } \left( \boldsymbol { \varsigma } _ { i j } \left( \boldsymbol { X } _ { i } \right) \right) \right\} , \ j = 1 , 2 , 3 } \\ & { \boldsymbol { \varsigma } _ { i } \left( \boldsymbol { X } _ { i } \right) = \left[ \boldsymbol { \varsigma } _ { i 1 } \left( \boldsymbol { X } _ { i } \right) , \boldsymbol { \varsigma } _ { i 2 } \left( \boldsymbol { X } _ { i } \right) , \boldsymbol { \varsigma } _ { i 3 } \left( \boldsymbol { X } _ { i } \right) \right] ^ { \mathrm { T } } } \end{array}
$$

with $\varsigma _ { i } ( X _ { i } )$ being selected between $0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \mathrm { T } } \nabla V _ { i c } ^ { * \mathrm { T } }$ and $\Lambda _ { i }$ . The ς . αdetailed derivation of (32) can be found in [28].

Substituting the approximation (29) and (31) into the HJI (27) yields

$$
\begin{array} { r l } & { H _ { i } \left( { X } _ { i } , { \boldsymbol w } _ { i } ^ { * \mathrm { T } } \nabla \phi _ { i } \left( { X } _ { i } \right) , \nabla \bar { B } _ { i } ^ { c } \left( { X } _ { i } \right) , { \boldsymbol u } _ { i } ^ { * } , { \boldsymbol d } _ { i } ^ { * } \right) } \\ & { = { L } _ { i } ^ { c } \left( { X } _ { i } , { \boldsymbol u } _ { i } ^ { * } , { \boldsymbol d } _ { i } ^ { * } \right) + \left( { \boldsymbol w } _ { i } ^ { * \mathrm { T } } \nabla \phi _ { i } + \nabla \bar { B } _ { i } ^ { c } \right) \left( F _ { i } + G _ { i } { \boldsymbol u } _ { i } ^ { * } + { \boldsymbol d } _ { i } ^ { * } \right) } \\ & { = \varepsilon _ { \mathrm { H i l } } } \end{array}\tag{34}
$$

where $\varepsilon _ { \mathrm { H i } 1 } = - \nabla \epsilon _ { i } ( X _ { i } ) ( F _ { i } + G _ { i } u _ { i } ^ { * } + d _ { i } ^ { * } )$ is the residual error.

ε Remark 6: Different from the existing safe online learning algorithms [23], [25], [30] that only use the form of (29) to approximate the value function, this article approximates the value function by constructing a bounded function $\bar { B } _ { i } ^ { c } ( z )$ related to the barrier function $B _ { i } ^ { c } ( z )$ . This design can directly apply safety related items $\bar { B } _ { i } ^ { c } ( z )$ to subsequent optimal strategy learning, which can reflect the attributes of safe constraints during the learning process. In addition, the works [23], [25], [30] only consider single nonlinear systems.

## B. Tuning and Convergence of the Critic NN

Since $w _ { i } ^ { * }$ is an unknown weight vector, the optimal safe strategy $u _ { i } ^ { * }$ in (32) and the disturbance policy $d _ { i } ^ { * }$ in (33) cannot be implemented during the formation process. For this reason, an online learning strategy will be presented by introducing a critic network. Specially, let $\hat { w } _ { i } ( t )$ denote the estimation of ideal weight $w _ { i } ^ { * }$ . Then, the value function $V _ { i c } ^ { * }$ is estimated as follows:

$$
\hat { V } _ { i c } \left( X _ { i } \right) = \hat { w } _ { i } ^ { \mathrm { T } } \phi _ { i } \left( X _ { i } \right) + \bar { B } _ { i } ^ { c } \left( X _ { i } \right) , \quad X _ { i } \in \mathcal { X } _ { i } .\tag{35}
$$

Following this, the approximate safe optimal feedback policies $u _ { i } ^ { * }$ and $d _ { i } ^ { * }$ for $X _ { i } \in \mathcal { X } _ { i }$ can be expressed by

$$
\hat { u } _ { i } = - \alpha _ { i } \operatorname { t a n h } { \left( \hat { \Lambda } _ { i } \right) } + \bar { \beta } _ { i }\tag{36}
$$

$$
\hat { d } _ { i } = \frac { 1 } { 2 \delta ^ { 2 } } \left[ \hat { w } _ { i } ^ { \mathrm { T } } \nabla \phi _ { i } + \nabla \bar { B } _ { i } ^ { c } ( \cdot ) \right] ^ { \mathrm { T } }\tag{37}
$$

where $\hat { \Lambda } _ { i } = 0 . 5 \alpha _ { i } ^ { - 1 } G _ { i } ^ { \mathrm { T } } ( \hat { w } _ { i } ^ { \mathrm { T } } \nabla \phi _ { i } { + } \nabla \bar { B } _ { i } ^ { c } ( \cdot ) ) ^ { \mathrm { T } }$ . From (35) to (37), we . α φcan derive that the Hamiltonian function (23) has the following approximate form:

$$
\begin{array} { r l } & { \hat { H } _ { i } \left( X _ { i } \left( t \right) , \nabla V _ { i c } ^ { \mathrm { T } } \left( X _ { i } \left( t \right) \right) , \hat { u } _ { i } \left( t \right) , \hat { d } _ { i } \left( t \right) \right) } \\ & { = L _ { i } ^ { c } \left( X _ { i } \left( t \right) , \hat { u } _ { i } \left( t \right) , \hat { d } _ { i } \left( t \right) \right) + \left( \hat { w } _ { i } ^ { \mathrm { T } } \left( t \right) \nabla \phi _ { i } \left( X _ { i } \left( t \right) \right) \right. } \\ & { ~ \left. + \nabla \bar { B } _ { i } ^ { c } \left( X _ { i } \left( t \right) \right) \right) \left( F _ { i } \left( X _ { i } \left( t \right) \right) + G _ { i } \left( X _ { i } \left( t \right) \right) \hat { u } _ { i } \left( t \right) + \hat { d } _ { i } \left( t \right) \right) } \\ & { = \hat { w } _ { i } ^ { \mathrm { T } } \left( t \right) \varphi _ { i } \left( t \right) + L _ { i } ^ { c } \left( X _ { i } \left( t \right) , \hat { u } _ { i } \left( t \right) , \hat { d } _ { i } \left( t \right) \right) + \bar { B } _ { i } ^ { c ^ { \prime } } \left( t \right) } \end{array}\tag{38}
$$

where

$$
\begin{array} { r l } & { { { \varphi } _ { i } } \left( t \right) = \nabla \phi \left( { { X } _ { i } } \left( t \right) \right) \left( { { F } _ { i } } \left( { { X } _ { i } } \left( t \right) \right) + { { G } _ { i } } \left( { { X } _ { i } } \left( t \right) \right) { { { \hat { u } } } _ { i } } \left( t \right) + { { { \hat { d } } } _ { i } } \left( t \right) \right) } \\ & { \hat { { B } } _ { i } ^ { c ^ { \prime } } \left( t \right) = \nabla { { \bar { B } } _ { i } ^ { c } } \left( { { X } _ { i } } \left( t \right) \right) \left( { { F } _ { i } } \left( { { X } _ { i } } \left( t \right) \right) + { { G } _ { i } } \left( { { X } _ { i } } \left( t \right) \right) { { { \hat { u } } } _ { i } } \left( t \right) + { { { \hat { d } } } _ { i } } \left( t \right) \right) . } \end{array}
$$

Since $H _ { i } ( X _ { i } , \triangledown V _ { i c } ^ { * } , u _ { i } ^ { * } , d _ { i } ^ { * } ) = 0$ , the Hamiltonian estimation error ,simplifies to

$$
\begin{array} { r l } & { { { e } _ { i } } \left( t \right) = { { \hat { H } } _ { i } } \left( { { X } _ { i } } \left( t \right) , \nabla { { V } _ { i c } ^ { \mathrm { T } } } \left( { { X } _ { i } } \left( t \right) \right) , { { \hat { u } } _ { i } } \left( t \right) , { { \hat { d } } _ { i } } \left( t \right) \right) } \\ & { \qquad = { { \hat { w } } _ { i } ^ { \mathrm { T } } } \left( t \right) { { \varphi } _ { i } } \left( t \right) + { { L } _ { i } ^ { s } } \left( { { X } _ { i } } \left( t \right) , { { \hat { u } } _ { i } } \left( t \right) , { { \hat { d } } _ { i } } \left( t \right) \right) + { { { \bar { B } } _ { i } ^ { c } } } ^ { \prime } \left( t \right) } \end{array}\tag{39}
$$

which corresponds to the data collected at the current time $t \geq 0$ . Meanwhile, to avoid assuming the PE condition, we introduce the error $e _ { i } ( t _ { p } , t )$ related to the recorded data at the time instants $0 \leq t _ { 1 } < t _ { 2 } , . . . , t _ { l } < t$ as follows:

$$
\begin{array} { r l } & { e _ { i } \left( t _ { p } , t \right) = \hat { w } _ { i } ^ { \mathrm { T } } \left( t \right) \varphi _ { i } \left( t _ { p } \right) + L _ { i } ^ { c } \left( X _ { i } \left( t _ { p } \right) , \hat { u } _ { i } \left( t _ { p } \right) , \hat { d } _ { i } \left( t _ { p } \right) \right) } \\ & { \phantom { e                                                 \ } + \bar { B } _ { i } ^ { c ^ { \prime } } \left( t _ { p } \right) , \ p = 1 , 2 , \ldots , l . } \end{array}\tag{40}
$$

Subsequently, a loss function for $t \geq 0$ is constructed as follows:

$$
\begin{array} { l } { \displaystyle E _ { i } \left( \hat { w } _ { i } \left( t \right) \right) = \frac { 1 } { 2 } \left[ \frac { \left( e _ { i } \left( t \right) \right) ^ { 2 } } { \left( \varphi _ { i } ^ { \mathrm { T } } \left( t \right) \varphi _ { i } \left( t \right) + 1 \right) ^ { 2 } } \right. } \\ { \displaystyle \left. + \sum _ { p = 1 } ^ { l } \frac { \left( e _ { i } \left( t _ { p } , t \right) \right) ^ { 2 } } { \left( \varphi _ { i } ^ { \mathrm { T } } \left( t _ { p } \right) \varphi _ { i } \left( t _ { p } \right) + 1 \right) ^ { 2 } } \right] . } \end{array}\tag{41}
$$

Note that the basis for selecting weight $\hat { w } _ { i }$ is to minimize $E _ { i } ( \hat { w } _ { i } )$ and ensure the convergence of $\hat { w } _ { i } .$ . To this end, by employing the gradient descent method, the following form of weight update law is designed:

$$
\begin{array} { l } { { \displaystyle { \dot { \hat { w } } } _ { i } \left( t \right) = - \sigma _ { i w } \frac { \partial E \left( { \hat { w } } _ { i } \left( t \right) \right) } { \partial { \hat { w } } _ { i } } } \ ~ } \\ { { \displaystyle ~ = - \sigma _ { i w } \frac { \varphi _ { i } \left( t \right) e _ { i } \left( t \right) } { \left( \varphi _ { i } ^ { \operatorname { T } } \left( t \right) \varphi _ { i } \left( t \right) + 1 \right) ^ { 2 } } } \ ~ } \\ { { \displaystyle ~ - \sigma _ { i w } \sum _ { p = 1 } ^ { l } \frac { \varphi _ { i } \left( t _ { p } \right) e _ { i } \left( t _ { p } , t \right) } { \left( \varphi _ { i } ^ { \operatorname { T } } \left( t _ { p } \right) \varphi _ { i } \left( t _ { p } \right) + 1 \right) ^ { 2 } } } \ ~ } \end{array}\tag{42}
$$

with the initial value being $\hat { w } _ { i } ( 0 ) = \hat { w } _ { i 0 }$ , and $\sigma _ { i w } > 0$ denoting a learning rate. Define $\tilde { w } _ { i } ( t ) = \hat { w } _ { i } ( t ) - w _ { i } ^ { * }$ as the weight estimation error. Then, the Hamiltonian estimation error $e _ { i } ( t )$ is further rewritten as follows:

$$
e _ { i } \left( t \right) = \tilde { w } _ { i } ^ { \mathrm { T } } \left( t \right) { \varphi _ { i } \left( t \right) } + \varepsilon _ { i H } \left( t \right)\tag{43}
$$

where $\begin{array} { r l r } { \varepsilon _ { i H } } & { { } = } & { w _ { i } ^ { * \mathrm { T } } \varphi _ { i } ~ + ~ L _ { i } ^ { s } ( X _ { i } , \hat { u } _ { i } , \hat { d } _ { i } ) ~ + ~ \bar { B } _ { i } ^ { c ^ { \prime } } ( X _ { i } , \hat { u } _ { i } , \hat { d } _ { i } ) \quad \triangleq } \end{array}$ $H _ { i } ( X _ { i } , \nabla \phi _ { i } ^ { \mathrm { T } } ( \cdot ) \boldsymbol { w } _ { i } ^ { * } , \nabla \bar { B } _ { i } ^ { c } ( X _ { i } ) , \hat { \boldsymbol { u } } _ { i } , \hat { d } _ { i } )$ , , , is the residual error.

,Set $\bar { \varphi } _ { i } = ( \varphi _ { i } / \varphi _ { i } ^ { \mathrm { T } } \varphi _ { i } + 1 ) .$ ,, and $\bar { \varepsilon } _ { i H } = ( \varepsilon _ { i H } / \varphi _ { i } ^ { \mathrm { T } } \varphi _ { i } + 1 )$ . Then, let $\Pi _ { i } = [ \bar { \varphi } _ { i } ( t _ { 1 } ) , \bar { \varphi } _ { i } ( t _ { 2 } ) , \ldots , \bar { \varphi } _ { i } ( t _ { l } ) ]$ ε ε /ϕ ϕ be the set of recorded data. ϕ , ϕ , . . . , ϕBefore stating the convergence of $\tilde { w } _ { i } ( t )$ , we give two indispensable assumptions, which are utilized in [26] and [28]. Both assumptions are standard and practically reasonable. Specifically, Assumption 3 requires the Hamiltonian estimation error to be bounded on a compact set, which is consistent with the boundedness of disturbances and approximation errors in UAV systems. Assumption 4 ensures that the recorded dataset is sufficiently rich (i.e., has full rank), which can be achieved in practice by using experience replay.

Assumption 3: For all $X _ { i } ~ \in ~ { \mathcal { X } } _ { i }$ , there exists a positive constant $d _ { \varepsilon _ { i } }$ such that $\| \varepsilon _ { i H } \| < d _ { \varepsilon _ { i } }$

Assumption 4: The recorded data contain the same number of linearly independent elements, that is, rank ${ \mathrm { ( } } \Pi _ { i } { \mathrm { ) } } = m$ with $m < l .$

Then, for the weight update law (42), the following properties can be obtained, which proof is included in Appendix.

Theorem 2: For the adaptive critic learning law $\hat { w } _ { i }$ designed in (42), under Assumptions 2 and 3, the weight error $\tilde { w } _ { i } ( t )$ for any $\tilde { w } _ { i } ( 0 ) \in R ^ { m }$ is UUB. 

Remark 7: Note that Assumption 4 means that the number of stored samples l is greater than m. Theorem 2 indicates that if the recorded dataset $\{ \bar { \varphi } _ { i } ( t _ { j } ) \} _ { j = 1 } ^ { l }$ is m-sufficiently rich, the weight error $\tilde { w } _ { i } ( t )$ ϕ can be ensured to converge to the origin if $\varepsilon _ { i H }$ lies in a sufficiently small neighborhood of the origin. εThen, from the definition of $\tilde { w } _ { i } ( t )$ , it follows that the update law wˆ i(t) in (42) also has UUB performance. Unlike the existing SRL strategy [30] that requires PE conditions, Assumption 4 can be easily checked online, thus reducing the complexity of the algorithm.

Remark 8: The main online computational cost scheme stems from the evaluation and update of the critic network. Each evaluation involves one forward pass and one gradient computation, both of which have a computational complexity of O(m), where m denotes the number of hidden neurons. The adaptive weight update using non-Lipschitz experience replay operates over a finite set of recorded data points, leading to a total computational complexity of $\mathcal { O } ( m l )$ , where l is the number of recorded samples. Given that both m and l are relatively small, the computational burden is moderate and suitable for real-time applications on typical UAV processors.

## V. STABILITY ANALYSIS

Based on the designed approximate optimal safe controller $\hat { u } _ { i }$ in (36), the optimal disturbance policy $\hat { d } _ { i }$ in (37) and the critic learning law $\hat { w } _ { i }$ in (42) with the weight error dynamics (64) for $i = 1 , 2 , \ldots , N ,$ , we will discuss the stability of the , , . .closed-loop system $( X _ { i } , \tilde { w } _ { i } )$ consisting of N UAVs with the ,following dynamic equations:

$$
\begin{array} { r } { \left\{ \dot { X } _ { i } \left( t \right) = F \left( X _ { i } \left( t \right) \right) + G _ { i } \left( X _ { i } \left( t \right) \right) \hat { u } _ { i } \left( t \right) + \hat { d } _ { i } \left( t \right) \right. } \\ { \left. \dot { \tilde { w } } _ { i } \left( t \right) = \left( \bar { \varphi } _ { i } \left( t \right) \bar { \varphi } _ { i } ^ { \mathrm { T } } \left( t \right) + \sum _ { p = 1 } ^ { l } \bar { \varphi } _ { i } \left( t _ { p } \right) \bar { \varphi } _ { i } ^ { \mathrm { T } } \left( t _ { p } \right) \right) \tilde { w } _ { i } \left( t \right) \right. } \\ { \left. - \left( \bar { \varphi } _ { i } \left( t \right) \bar { \varepsilon } _ { i H } \left( t \right) + \sum _ { p = 1 } ^ { l } \bar { \varphi } _ { i } \left( t _ { p } \right) \bar { \varepsilon } _ { i H } \left( t _ { p } \right) \right) \right. } \end{array}\tag{44}
$$

where $\bar { \varphi } _ { i } ( \cdot )$ and $\bar { \varepsilon } _ { i H } ( \cdot )$ are the same as defined in (64).

ϕ εTheorem 3: For the system (44), let the augmented state be $Z = [ Z _ { 1 } ^ { \mathrm { T } } , Z _ { 2 } ^ { \mathrm { T } } , \ldots , Z _ { N } ^ { \mathrm { T } } ] ^ { \mathrm { T } }$ with $Z _ { i } = [ X _ { i } ^ { \mathrm { T } } , \tilde { w } _ { i } ^ { \mathrm { T } } ] ^ { \mathrm { T } }$ for $i = 1 , \ldots , N .$ , , . . . , , , . . . ,Then, under Assumptions 1–4, the closed-loop system is stable in the sense of UUB, and the solution $Z _ { i } ( t )$ converges to the compact set $\mathcal { C } _ { Z _ { i } } ~ = ~ \bar { \mathcal { X } } _ { i } \times \bar { \Omega } _ { i \tilde { w } }$ for each initial $Z _ { i } ( 0 ) \ =$ $( X _ { i } ( 0 ) , \tilde { w } _ { i } ( 0 ) ) \in \mathcal { X } _ { i } \times R ^ { m }$ , where $\bar { \mathcal X } _ { i }$ and $\Omega _ { i w }$ are defined in ,(53). Furthermore, the optimal safe formation strategy $( \hat { u } _ { i } , \hat { d } _ { i } )$ given in (36) and (37) can converge to the Nash equilibrium $( u _ { i } ^ { * } , d _ { i } ^ { * } )$ in (32) and (33) within an adjustable bound.

,Proof: Design a total Lyapunov function as $\begin{array} { r l } { \bar { V } ( t ) } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N } ( V _ { i c } ^ { * } ( X _ { i } ) + V _ { i w } ( \tilde { w } _ { i } ) ) } \end{array}$ with $V _ { i w }$ defined in the proof of Theorem 2. Then, the time derivative $\dot { \bar { V } } ( t )$ satisfies

$$
\dot { \bar { V } } \left( t \right) = \sum _ { i = 1 } ^ { N } \left( \dot { V } _ { i c } ^ { * } \left( X _ { i } \right) + \dot { V } _ { i w } \left( \tilde { w } _ { i } \right) . \right.\tag{45}
$$

Using inequality (61) in Appendix, we can calculate the time derivative $\dot { V } _ { i c } ^ { * }$ along the dynamics (44) as follows:

$$
\begin{array} { r } { \dot { V } _ { i c } ^ { * } \left( X _ { i } \right) \leq - \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) X _ { i } ^ { \operatorname { T } } X _ { i } - \kappa _ { i } B _ { i } ^ { c } } \end{array}
$$

$$
\begin{array} { r l } & { - \displaystyle \sum _ { s = \nu , \psi , \theta } ( \alpha _ { i } ^ { s } \operatorname { t a n h } ^ { - 1 } ( \tau _ { i } ^ { s * } ) ) ^ { 2 } + \varrho _ { i 1 } } \\ & { +  \nabla V _ { i c } ^ { * } G _ { i } ( \hat { u } _ { i } - u _ { i } ^ { * } ) + \nabla V _ { i c } ^ { * } ( \hat { d } _ { i } - d _ { i } ^ { * } ) . } \end{array}\tag{46}
$$

According to the definition of $u _ { i } ^ { * }$ in (25), we have

$$
\begin{array} { r } { \nabla V _ { i c } ^ { * } G _ { i } = - \left( 2 \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \right) ^ { \mathrm { T } } } \end{array}\tag{47}
$$

which together with the optimal controller (32) and the corresponding estimated (36) yields

$$
\begin{array} { r } { \nabla V _ { i c } ^ { * } G _ { i } \left( \widehat { u } _ { i } - u _ { i } ^ { * } \right) = - \big ( 2 \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \big ) ^ { \mathrm { T } } } \\ { \times \left( \alpha _ { i } \left( \operatorname { t a n h } { \left( \Lambda _ { i } \right) } - \operatorname { t a n h } { \left( \widehat { \Lambda } _ { i } \right) } \right) - \epsilon _ { i u _ { i } ^ { * } } \right) } \end{array}\tag{48}
$$

It can be further derived as follows:

$$
\begin{array} { r l } & { \nabla V _ { i c } ^ { * } G _ { i } \left( \widehat { u } _ { i } - u _ { i } ^ { * } \right) \leq \left. \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \right. ^ { 2 } } \\ & { \qquad + \ 2 4 \Vert \alpha _ { i } \Vert ^ { 2 } + 2 b _ { G } ^ { 2 } \left( \epsilon _ { i d } ^ { d } \right) ^ { 2 } } \end{array}\tag{49}
$$

which proof is included in Appendix.

Furthermore, based on the fact that kO $V _ { i c } ^ { * } \| \leq b _ { i \nu }$ in Theorem 1 and Assumption 4, it follows from equations (33) and (37) that:

$$
\nabla V _ { i c } ^ { * } \left( \widehat { d } _ { i } - d _ { i } ^ { * } \right) \leq \frac { 1 } { 2 \delta ^ { 2 } } b _ { i \nu } \left( \phi _ { i d } ^ { d } \| \tilde { w } _ { i } \| + \epsilon _ { i d } ^ { d } \right) .\tag{50}
$$

Note that from the definitions of $\tau _ { i } ^ { s * }$ in $( 5 9 ) , \alpha _ { i } ^ { s }$ in (16), and $\alpha _ { i }$ in (25), we have

$$
\sum _ { s = \nu , \psi , \theta } \left( \alpha _ { i } ^ { s } \operatorname { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) \right) ^ { 2 } = \left. \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \right. ^ { 2 } .
$$

This together with inequalities (49) and (50), inequality (46) can be further derived as follows:

$$
\begin{array} { l } { { \displaystyle { \dot { V } } _ { i c } ^ { * } \le - \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) X _ { i } ^ { \mathrm { T } } X _ { i } - \kappa _ { i } B _ { i } ^ { c } + 2 4 \| \alpha _ { i } \| ^ { 2 } + 2 b _ { G } ^ { 2 } \left( \epsilon _ { i d } ^ { d } \right) ^ { 2 } } } \\ { { \displaystyle ~ + \frac { 1 } { 2 \delta ^ { 2 } } b _ { i \nu } \left( \phi _ { i d } ^ { d } \| \tilde { w } _ { i } \| + \epsilon _ { i d } ^ { d } \right) - \sum _ { s = \nu , \psi , \theta } \left( \alpha _ { i } ^ { s } \operatorname { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) \right) ^ { 2 } } } \\ { { \displaystyle ~ + \| \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \| ^ { 2 } + \varrho _ { i 1 } } } \\ { { \displaystyle \triangleq - \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) X _ { i } ^ { \mathrm { T } } X _ { i } - \kappa _ { i } B _ { i } ^ { c } + \varrho _ { i w } \| \tilde { w } _ { i } \| + \varrho _ { i 2 } } } \end{array}\tag{51}
$$

where $\varrho _ { i w } = ( 1 / 2 \delta ^ { 2 } ) b _ { i \nu } \phi _ { i d } ^ { d }$ and $\varrho _ { i 2 } = \varrho _ { i 1 } + 2 4 \| \alpha _ { i } \| ^ { 2 } + 2 b _ { G } ^ { 2 } ( \epsilon _ { i d } ^ { d } ) ^ { 2 } +$ $( 1 / 2 \delta ^ { 2 } ) b _ { i \nu } \epsilon _ { i d } ^ { d }$ / δ φ % %are two positive constants.

Afterward, combining inequalities (67) and $( 5 1 ) , \dot { \bar { V } } ( t )$ in (45) can be further written as follows:

$$
\begin{array} { r l } & { \dot { \bar { V } } \leq \displaystyle \sum _ { i = 1 } ^ { N } \left( - \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) X _ { i } ^ { \mathrm { T } } X _ { i } - \kappa _ { i } B _ { i } ^ { c } + \varrho _ { i 2 } \right. } \\ & { \quad \quad \left. + \left( \varrho _ { i w } + c _ { i w } \right) \| \tilde { w } _ { i } \left( t \right) \| - s _ { \operatorname* { m i n } } ^ { 2 } \left( \Pi _ { i } \right) \| \tilde { w } _ { i } \left( t \right) \| ^ { 2 } \right) . } \end{array}\tag{52}
$$

Therefore, inequality (52) indicates $\dot { \bar { V } } < 0$ as long as $( X _ { i } , \tilde { w } _ { i } )$ are out of the set $\mathcal { C } _ { Z _ { i } }$ , where $\mathcal { C } _ { Z _ { i } } = \bar { \mathcal { X } } _ { i } \times \bar { \Omega } _ { \tilde { w } _ { i } }$ and

$$
\begin{array} { r l r } & { } & { \bar { \mathcal { X } } _ { i } = \left\{ X _ { i } \vert \vert X _ { i } \vert \vert \le \sqrt { \frac { \varrho _ { i 2 } } { \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) } } = r _ { i 1 } \right\} } \\ & { } & { \bar { \Omega } _ { \tilde { w } _ { i } } = \left\{ \tilde { w } _ { i } \vert \vert \vert \tilde { w } _ { i } \vert \vert \le \frac { \varrho _ { i w } + c _ { i w } } { s _ { \operatorname* { m i n } } ^ { 2 } \left( \Pi _ { i } \right) } = r _ { i 2 } \right\} . } \end{array}\tag{53}
$$

This property in inequality (52) also means that the closedloop dynamics (44) is stable in the sense of UUB. That is, the trajectories $( X _ { i } ( t ) , \ \tilde { w } _ { i } ( t ) ) , \ t \geq 0$ , will converge to the set $\mathcal { C } _ { Z _ { i } }$ with $\bar { \mathcal { X } } _ { i } \subset \mathcal { X } _ { i } \subset \mathcal { S } _ { i }$ . Furthermore, according on the fact that the Cartesian product of two compact sets is itself a compact set, one can infer that $\mathcal { C } _ { Z _ { i } }$ is compact. This further indicates that the safety constraints for $X _ { i } ~ ( i = 1 , 2 , \ldots , N )$ are satisfied during the entire learning process.

Finally, by applying the mean value theorem, according to (32) and (36), we have

$$
\begin{array} { l } { { \hat { u } } _ { i } \left( X _ { i } \right) - u _ { i } ^ { * } \left( X _ { i } \right) = \alpha _ { i } \left( \operatorname { t a n h } \left( \Lambda _ { i } \right) - \operatorname { t a n h } \left( \hat { \Lambda } _ { i } \right) \right) - \epsilon _ { i u _ { i } ^ { * } } } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \left( I _ { 3 } - \Phi _ { i } \left( \bar { \zeta } _ { i } ( \cdot ) \right) \right) G _ { i } ^ { \mathrm { T } } \nabla \phi _ { i } ^ { \mathrm { T } } \tilde { w } _ { i } - \epsilon _ { i u _ { i } ^ { * } } } \end{array}\tag{54}
$$

where $\Phi _ { i } ( \bar { \varsigma } _ { i } ( X _ { i } ) ) = \mathrm { d i a g } \{ \mathrm { t a n h } ^ { 2 } ( \bar { \varsigma } _ { i j } ( X _ { i } ) ) \}$ with $j = 1 , 2 , 3$ and $\bar { \varsigma } _ { i } ( \cdot ) = [ \bar { \varsigma } _ { i 1 } ( \cdot ) , \bar { \varsigma } _ { i 2 } ( \cdot ) , \bar { \varsigma } _ { i 3 } ( \cdot ) ] ^ { \mathrm { T } }$ being chosen between $\Lambda _ { i }$ ,and $\hat { \Lambda } _ { i }$ ςBecause $\lVert I _ { 3 } - \Phi _ { i } ( \bar { \varsigma } _ { i } ) \rVert \leq 2$ , similar to the derivation of (71), utilizing Assumptions 1 and 2 yields

$$
\| \hat { \boldsymbol { u } } _ { i } \left( \boldsymbol { X } _ { i } \right) - \boldsymbol { u } _ { i } ^ { * } \left( \boldsymbol { X } _ { i } \right) \| \le b _ { G } \phi _ { i d } ^ { d } \| \tilde { \boldsymbol { w } } _ { i } \| + b _ { G } \epsilon _ { i d } ^ { d } .\tag{55}
$$

It follows from Theorem 2 that the ultimate bound of $\tilde { w } _ { i }$ is $r _ { i 2 }$ . This together with inequality (55) yields

$$
\| \hat { { \boldsymbol u } } _ { i } \left( { \boldsymbol X } _ { i } \right) - { \boldsymbol u } _ { i } ^ { * } \left( { \boldsymbol X } _ { i } \right) \| \le b _ { G } \left( \phi _ { i d } ^ { d } r _ { i 2 } + \epsilon _ { i d } ^ { d } \right) = r _ { { \boldsymbol u } _ { i } }\tag{56}
$$

where $r _ { u _ { i } }$ is an adjustable positive constant. Next, similar to the derivation of (50), it has

$$
\lVert \hat { d } _ { i } - d _ { i } ^ { * } \rVert \leq \frac { 1 } { 2 \delta ^ { 2 } } \left( \phi _ { i d } ^ { d } r _ { i 2 } + \epsilon _ { i d } ^ { d } \right) = r _ { d _ { i } }\tag{57}
$$

with $r _ { d _ { i } }$ being an adjustable positive constant. Inequalities (56) and (57) indicate that $( \hat { u } _ { i } , \hat { d } _ { i } )$ converges to $( u _ { i } ^ { * } , d _ { i } ^ { * } )$ with an ,adjustable bound. This completes the proof.

Remark 9: Note that the ultimate uniform boundedness of the closed-loop system (44) implies that all error states $X _ { i }$ $( i ~ = ~ 1 , 2 , \ldots , N )$ of all UAVs converges to a small neigh-, , . . . ,borhood of the origin, that is, li $\mathtt { n } _ { t \to \infty } X _ { i } ( t ) \ = \ o _ { i }$ . It follows from $X _ { i } = [ e _ { i } ^ { \mathrm { T } } , \Delta \zeta _ { i } ^ { \mathrm { T } } ] ^ { \mathrm { T } }$ that both the formation tracking error $e _ { i }$ , ζand the relative flight state error $\Delta \zeta _ { i }$ also converge to a small ζneighborhood of the origin. Given the definition of $e _ { i }$ and $\Delta \zeta _ { i }$ ζthe practical formation control performance can be achieved.

Remark 10: It is noteworthy that inequalities (56) and (57) demonstrate that the errors in the approximate optimal safety controller $\hat { u } _ { i }$ and the disturbance strategy $\hat { d } _ { i } ,$ relative to their corresponding optimal values, can be determined by the bounds $r _ { u _ { i } }$ and $r _ { d _ { i } }$ , respectively. Since the boundaries $r _ { u _ { i } }$ and $r _ { d _ { i } }$ depend on constants $r _ { i 2 }$ and $\epsilon _ { i d } ^ { d } ,$ they are essentially related to $\nabla \epsilon _ { i } ( X _ { i } )$ and $\delta _ { i } .$ . As stated in [28] when the dimension of NNs $m \to \infty$ δ, it has that $\epsilon _ { i } ( X _ { i } )  0$ and $\nabla \epsilon _ { i } ( X _ { i } )  0$ . In addition, $\delta _ { i }$  is the design parameter such that the boundaries $r _ { u _ { i } }$ and $r _ { d _ { i } }$ can be adjustable and made arbitrarily small.

## VI. EXAMPLE AND SIMULATION

To substantiate the reliability and effectiveness of the presented safe formation tracking control scheme, this part gives the simulation results consisting of five flight controller.

We consider the formation tracking task as a cooperative low-altitude penetration, whose trajectory reference is provided by a virtual leader. The desired formation positions of five UAVs are set as $\eta _ { 1 r } = [ 0 , 0 , 0 ] ^ { \mathrm { T } } , \eta _ { 2 r } = [ - 1 2 5 , - 2 5 , 0 ] ^ { \mathrm { T } }$ $\eta _ { 3 r } = [ - 1 2 5 , 2 5 , 0 ] ^ { \mathrm { T } } , \eta _ { 4 r } = [ - 2 5 0 , - 2 5 , 0 ] ^ { \mathrm { T } }$ , and $\begin{array} { r l } { \eta _ { 5 r } } & { { } = } \end{array}$ $[ - 2 5 0 , 2 5 , 0 ] ^ { \mathrm { T } }$ , , η , ,. The selected constants of autopilot are $\rho _ { \nu } = 1$ $\rho _ { \psi } ~ = ~ 1 0$ and $\rho _ { \theta } ~ = ~ 1 0 ~ $ . Considering the factors of electroρψ ρθmagnetic interference and gusts, as well as the attributes of high-frequency vibration and large amplitude, the system is inevitably affected by disturbances. Hence, during the time interval [20 60]s, we test the formation tracking performance ,of multi-UAV system under the complicated disturbances $d _ { x i } = 2 \sin ( t ) , d _ { y i } = 3 \cos ( t ) , d _ { z i } = 2 \sin ( t ) , d _ { v i } = 2 \sin ( t ) + \varpi ,$ $d _ { \psi i } = 2 \cos ( t ) + \varpi .$ , and $d _ { \theta i } = 2 \sin ( t ) + \varpi$ with $i = 1 , 2 , \ldots , 5 ,$ ψwhere $\varpi$ \$ θ \$ , , . . . ,denotes a random noise with a standard normal \$distribution. In addition, the initial values of virtual leader and UVAs are given as $\eta _ { r } ( 0 ) = [ 4 0 , 0 , 6 0 0 ] ^ { \mathrm { T } } , \eta _ { 1 } ( 0 ) = [ 5 , 5 , 6 0 6 ] ^ { \mathrm { T } }$ $\eta _ { 2 } ( 0 ) = [ - 1 3 6 , - 3 7 , 6 0 3 ] ^ { \mathrm { T } } , \eta _ { 3 } ( 0 ) = [ - 1 2 5 , 3 7 , 5 9 6 ] ^ { \mathrm { T } } , \eta _ { 4 } ( 0 ) =$ η ,[−266 −32 595]T, $\eta _ { 5 } ( 0 ) = [ - 2 6 4 , 3 4 , 6 0 7 ] ^ { \mathrm { T } } , \ \nu _ { i } ( 0 ) = 1 0 \mathrm { m / s } .$ $\psi _ { i } ( 0 ) = 0 ^ { \circ }$ , and $\theta _ { i } ( 0 ) = 0 ^ { \circ }$

TABLE I  
KEY SIMULATION PARAMETERS
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Learning rate  $\sigma _ { i w }$ </td><td>0.5</td></tr><tr><td>Mini-batch size</td><td>64</td></tr><tr><td>Replay buffer size</td><td> $1 . 0 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Weight matrix  $R _ { i }$ </td><td> $I _ { 6 }$ </td></tr><tr><td> $L _ { 2 }$  gain δ</td><td>2</td></tr><tr><td>Damping coefficient  $\kappa _ { i }$ </td><td>0.1</td></tr><tr><td>Error bounds  $k _ { i e } ^ { o }$ </td><td>(30,30,20) m</td></tr><tr><td>Critic architecture</td><td>MLP [6 input,19 hidden,1 output]</td></tr></table>

<!-- image-->  
Fig. 1. Trajectories of velocity vi and $\nu _ { r } .$

<!-- image-->  
Fig. 2. Trajectories of velocity $\psi _ { i }$ and $\psi _ { r }$

Next, for $i = 1 , \ldots , 5$ , the control constraints are predefined as $0 ~ \leq ~ \nu _ { i } ^ { c } ~ \leq ~ 2 5 , ~ - 2 6 ~ \leq ~ \psi _ { i } ^ { c } ~ \leq ~ 3 5$ , and $- 2 6 ~ \le ~ \theta _ { i } ^ { c } ~ \le ~ 6 9$ Furthermore, the constraint norm bound is chosen as $k _ { i } ~ <$ $( ( x _ { i } - x _ { j } ) ^ { 2 } + ( y _ { i } - y _ { j } ) ^ { 2 } + ( z _ { i } - z _ { j } ) ^ { 2 } ) ^ { 1 / 2 } / 2$ by using Remark 1, where /jth UAV is closest neighbor of the ith UAV. For simulation, the parameters are given in Table I. The initial condition of weight matrix is selected as $\hat { w } _ { i } ( 0 ) = [ 1 , 1 , . . . , 1 ] _ { 1 9 \times 1 } ^ { \mathrm { T } }$

<!-- image-->  
Fig. 3. Trajectories of velocity i and r.

<!-- image-->  
Fig. 4. Optimal control trajectories of vc and its constrained boundary.

Fig. 5. Optimal control trajectories of ci and its constrained boundary.  
<!-- image-->  
Fig. 6. Optimal control trajectories of c and its constrained boundary.

<!-- image-->

Utilizing the above given parameters and applying the proposed safe formation tracking control scheme in Section IV, Figs. 1–10 illustrate the simulation results. The tracking performance of velocities, course angles, and pitch angles are shown in Figs. 1–3. From Figs. 1–3, it can be seen that each UAV effectively tracks the velocity, course angle, and pitch angle of the virtual leader. Fig. 7 shows the 3-D flight trajectory of five UAVs. The whole formation tracking process includes four stages: the first stage is formation configuration, where these UAVs are assembled from different initial positions to form a pattern. Formation diving is occurred in the second stage, where the UAV formation reduces the altitude while increasing the velocity. The third stage is formation level-off, where the UAV formation transitions from rapid descent to level flight. The final stage is the formation maintenance in the subsequent period. Fig. 8 indicates that the tracking error is successfully constrained in the safe bound. In addition, according to the curves within the time interval [20 60]s, it is evident that the presented control ,scheme can maintain high precision in achieving formation performance, even amidst complex disturbances. The collision avoidance capability of the UAV formation process is depicted in Fig. 9, where the curve represents the minimum distance between each UAV and its adjacent UAVs, indicating that the proposed algorithm effectively ensures collision avoidance between UAVs.

<!-- image-->

Fig. 7. Formation trajectory among UAVs.  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 8. Trajectories of tracking error norm and the norm bound.

<!-- image-->  
Fig. 9. Minimum distances between each UAV and their adjacent UAVs.

<!-- image-->

Fig. 10. Response curves of the critic weight.  
<!-- image-->  
Fig. 11. Comparison curves on relative distance trajectory M xi.

<!-- image-->  
Fig. 12. Comparison curves on relative distance trajectory M yi.

Figs. 4–6 show the optimal control curves of the controller based on SRL. The commands for velocity, course angle, and pitch angle are kept within a reasonable range, with appropriate limited excitation incorporated into these commands. The updating processes of critic weights is illustrated in Fig. 10, demonstrating that the weights can also converge after a period of learning. These figures indicate that the developed safety learning control algorithm is effective.

Next, we will further show the superiority of the developed control scheme through the simulation comparison. Specifically, while employing the same optimal Nash strategies (25) and (26) as presented in this article, the learning framework for deriving the approximate optimal Nash strategy adopts the approach proposed in [30]. In other words, the comparison scheme uses the method in [30] without incorporating a bounded function $\bar { B } _ { i } ^ { c } ( X _ { i } )$ into the approximation of the value function $V _ { i c } ^ { * }$ to reflect the safety properties during the learning process. Note that our scheme primarily avoids collisions between UAVs by constraining the tracking errors, that is, constraining the position error of each UAV relative to the leader, to ensure safety during operation. Thus, based on the aforementioned analysis and given space limitations, the comparison of simulation curves focuses mainly on relative distance trajectories. Under the same control parameters and learning parameters as described above, the comparative results are shown in Figs. 11–13. From Figs. 11–13, it can be seen that although the comparative method can ensure the safety during operation, it has a certain degree of conservatism. Specifically, the relative distance curves in the comparison scheme consistently exhibit greater amplitude than those of the proposed scheme, indicating that to avoid collisions, the comparison scheme tends to take a “wider detour” around neighboring agents. This excessively cautious collision avoidance behavior may lead to unnecessary resource consumption in practical applications. Therefore, the scheme developed in our work may have more practical significance.

<!-- image-->  
Fig. 13. Comparison curves on relative distance trajectory $\Delta \ { z } _ { i } .$

## VII. CONCLUSION

In this article, a novel robust safe optimal formation tracking control framework has been presented for multiple UAVs in the presence of disturbances and input constraints. The safety constraints for the collision avoidance have been transformed into an appropriate safety set with respect to the relative tracking errors. Following this, a new barrier function has been constructed, so that the safety can be ensured by combining the forward invariance of safe sets. Additionally, we have provided sufficient conditions for the existence of a safe Nash equilibrium under input constraints. The stability and safety analysis of closed-loop UAV systems based on the proposed optimal safe formation Nash strategy have also been provided.

Despite these promising results, the current simulations are limited to representative scenarios and UAV scales. Further validation under different scales, diverse flight environments, and stronger disturbance conditions would strengthen the generalizability of the framework. Moreover, beyond SRL, recent advances [37] in distributed foundation models and multimodal learning open promising directions for UAV swarm coordination. Multimodal learning could integrate heterogeneous information from UAV sensors, onboard cameras, and communication signals to improve decision-making robustness, while distributed aggregation techniques could enhance scalable cooperation among UAV swarms. Incorporating these directions into UAV formation tracking remains an important avenue for future work.

## APPENDIX

## A. Proof of Theorem 1

Proof: Based on HJI equation (27), we have

$$
\begin{array} { l } { { \displaystyle \dot { V } _ { i c } ^ { * } = \nabla V _ { i c } ^ { * } \left( F _ { i } + G _ { i } \boldsymbol { u } _ { i } ^ { * } + d _ { i } ^ { * } \right) } } \\ { { \displaystyle \quad = - X _ { i } ^ { \mathrm { T } } R _ { i } X _ { i } - \mathcal { U } _ { i } \left( \boldsymbol { u } _ { i } ^ { * } \right) - \kappa _ { i } B _ { i } ^ { c } + \frac { 1 } { 4 \delta _ { i } ^ { 2 } } \nabla V _ { i c } ^ { * } \nabla V _ { i c } ^ { * \mathrm { T } } } . }  \end{array}\tag{58}
$$

For the integrand penalty function $\boldsymbol { \mathcal { U } } _ { i } ( \boldsymbol { u } _ { i } ^ { * } )$ , let

$$
\xi _ { i } ^ { s } = \operatorname { t a n h } ^ { - 1 } \left( \frac { \tau _ { i } ^ { s } - \beta _ { i } ^ { s } } { \alpha _ { i } ^ { s } } \right) .
$$

Utilizing the variable substitution method, the following equation can be derived:

$$
\begin{array} { r l } & { - \mathcal { U } _ { i } \left( u _ { i } ^ { * } \right) } \\ & { \quad = - \displaystyle \sum _ { s = \nu , \psi , \theta } 2 \left( \alpha _ { i } ^ { s } \right) ^ { 2 } \int _ { 0 } ^ { \mathrm { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) } \xi _ { i } ^ { s } \left( 1 - \mathrm { t a n h } ^ { 2 } \left( \xi _ { i } ^ { s } \right) \right) d \xi _ { i } ^ { s } } \\ & { \quad = \displaystyle \sum _ { s = \nu , \psi , \theta } 2 \left( \alpha _ { i } ^ { s } \right) ^ { 2 } \mathrm { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) \theta _ { i } ^ { s } \mathrm { t a n h } ^ { 2 } \left( \theta _ { i } ^ { s } \right) } \\ & { \quad \quad - \displaystyle \sum _ { s = \nu , \psi , \theta } \left( \alpha _ { i } ^ { s } \mathrm { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) \right) ^ { 2 } } \end{array}\tag{59}
$$

where $\tau _ { i } ^ { s * } = ( u _ { i } ^ { s * } - \beta _ { i } ^ { s } ) / \alpha _ { i } ^ { s }$ . The first equation can be derived τ β /αby applying the integral mean-value theorem [34], and $\theta _ { i } ^ { s }$ is chosen between 0 and $\operatorname { t a n h } ^ { - 1 } ( \tau _ { i } ^ { s * } )$ .

τFrom the derivation of Theorem 1 in [27], it can be concluded that the first term of the above (59) is bounded by a positive constant $M _ { i } .$ , that is,

$$
\sum _ { s = \nu , \psi , \theta } 2 \left( \alpha _ { i } ^ { s } \right) ^ { 2 } \operatorname { t a n h } ^ { - 1 } \left( \tau _ { i } ^ { s * } \right) \theta _ { i } ^ { s } \operatorname { t a n h } ^ { 2 } \left( \theta _ { i } ^ { s } \right) \leq M _ { i } .\tag{60}
$$

In addition, since $u _ { i } ^ { * }$ is an admissible control, this means that $V _ { i c } ^ { * }$ is finite for any $X _ { i } \in S _ { i }$ by using the definition of admissible control [35]. Following this, it can be inferred that $\nabla V _ { i c } ^ { * }$ is also bounded, that is, kO $V _ { i c } ^ { * } \| \leq b _ { i \nu }$ with $b _ { i \nu }$ being a positive constant. Note that CBF $B _ { i } ^ { c } ( \cdot ) > 0$ for $X _ { i } \in \operatorname { I n t } ( S _ { i } )$ . Therefore, >combining (59) and (60) yields

$$
\begin{array} { r l } & { \dot { V } _ { i c } ^ { * } \leq - \lambda _ { \operatorname* { m i n } } ( R _ { i } ) X _ { i } ^ { \mathrm { T } } X _ { i } - \kappa _ { i } B _ { i } ^ { c } } \\ & { \qquad - \displaystyle \sum _ { s = \nu , \psi , \theta } ( \alpha _ { i } ^ { s } \operatorname { t a n h } ^ { - 1 } ( \tau _ { i } ^ { s * } ) ) ^ { 2 } + \varrho _ { i 1 } } \\ & { \qquad \leq - \lambda _ { \operatorname* { m i n } } ( R _ { i } ) X _ { i } ^ { \mathrm { T } } X _ { i } ) - \kappa _ { i } + \varrho _ { i 1 } } \end{array}\tag{61}
$$

where $\lambda _ { \operatorname* { m i n } } ( R _ { i } ) > 0$ is the minimum eigenvalue of the matrix $R _ { i } ,$ λ and $\varrho _ { i 1 } \ = \ M _ { i } + ( 1 / 4 \delta _ { i } ^ { 2 } ) b _ { i \nu } ^ { 2 }$ is a positive constant. The

above inequality (61) further implies that $\dot { V } _ { i c } ^ { * } < 0$ , provided the augmented error variable $X _ { i }$ <is outside of the compact set

$$
\Omega _ { X _ { i } } = \left\{ X _ { i } | | | X _ { i } | | \leq \ \sqrt { \frac { \varrho _ { i } } { \lambda _ { \operatorname* { m i n } } \left( R _ { i } \right) } } \right\} .
$$

Additionally, it follows from the definition of $V _ { i c } ^ { * } ( X _ { i } )$ in (21) that $V _ { i c } ^ { * } ~ > ~ 0$ for $X _ { i } ~ \neq ~ 0$ and $V _ { i c } ^ { * } ~ = ~ 0 ~ \Leftrightarrow ~ X _ { i } ^ { ' { } } ~ = ~ 0$ can be >obtained. So, the augmented dynamics (11) can be considered a Lyapunov function. For this reason, inequality (61) indicates that the closed-loop dynamics (11) under the optimal strategy $u _ { i } ^ { * }$ is stable in the sense of UUB.

Finally, the forward invariance of safe set is to be proven. According to $\dot { V } _ { i c } ^ { * } ( t ) ~ = ~ \nabla V _ { i c } ^ { * } \times \dot { X } _ { i } ( t )$ and the definition of $L _ { i } ^ { c } ( X _ { i } , u _ { i } , d _ { i } ) = L _ { i } ( X _ { i } , u _ { i } , d _ { i } ) + \kappa _ { i } B _ { i } ^ { c } ( X _ { i } )$ in (18), we can rewrite , , , ,the HJI (27) as follows:

$$
L _ { i } \left( X _ { i } , u _ { i } ^ { * } ( \cdot ) , d _ { i } ^ { * } ( \cdot ) \right) + \kappa _ { i } B _ { i } ^ { c } \left( X _ { i } \right) + \dot { V } _ { i c } ^ { * } = 0 .\tag{62}
$$

Let $X _ { i } ^ { 0 } \in { \mathcal { S } } _ { i }$ and consider $X _ { i } ( t ) , t \geqslant 0$ , as the solution to (11) with the Nash policy $( u _ { i } ^ { * } , d _ { i } ^ { * } )$ . Now, a contradiction will be uti-,lized to prove the forward invariance of safe set. Assume that there exists a time $T _ { i } ^ { * } > 0$ such that lim $1 _ { t  T ^ { * } }$ dist $( X _ { i } ( t ) , \partial { \cal S } _ { i } ) = 0$ >Based on this, integrating (62) over $[ 0 , T ^ { * } ]$ yields

$$
\begin{array} { c } { { V _ { i c } \left( X _ { i } \left( T _ { i } ^ { * } \right) \right) = - { \displaystyle \left( - V _ { i c } \left( X _ { i } ^ { 0 } \right) + \kappa _ { i } \int _ { 0 } ^ { T _ { i } ^ { * } } B _ { i } ^ { c } \left( X _ { i } \left( t \right) \right) d t \right. } } } \\ { { \displaystyle \left. + \int _ { 0 } ^ { T _ { i } ^ { * } } L _ { i } \left( X _ { i } \left( t \right) , u _ { i } ^ { * } \left( X _ { i } \right) , d _ { i } ^ { * } \left( X _ { i } \right) \right) d t \right) } } \end{array}\tag{63}
$$

It follows from inequality (1) that $- V _ { i c } ( X _ { i } ^ { 0 } ) +$ $\begin{array} { r } { \int _ { 0 } ^ { T _ { i } ^ { * } } L _ { i } ( X _ { i } ( t ) , u _ { i } ^ { * } ( X _ { i } ( t ) ) , d _ { i } ^ { * } ( X _ { i } ( t ) ) ) d t } \end{array}$ is bounded. Moreover, , ,using the property of barrier function $B _ { i } ^ { c } ( X _ { i } ) .$ , it follows from $\mathrm { l i m } _ { t  T _ { i } ^ { * } }$ dist(Xi(t) Si) = 0 that limt $\ l _ { \to T _ { i } ^ { * } } B _ { i } ^ { c } ( X _ { i } ( t ) ) \ = \ \infty$ . This further implies that $\begin{array} { r } { \int _ { 0 } ^ { T _ { i } ^ { * } } B _ { i } ^ { c } ( X _ { i } ( t ) ) d t = \infty } \end{array}$ and $V _ { i c } ( X _ { i } ( T _ { i } ^ { * } ) ) = - \infty$ leading to a contradiction. As a result, the above derivation indicates that as long as $X _ { i } ^ { 0 } \ \in \ S _ { i }$ , the solution $X _ { i } ( t ) , \ t \ \geq \ 0$ of the dynamics (11) cannot escape the safe set Si. So, the forward invariance of the safe set regarding the dynamics (11) can be ensured under the optimal control strategy. This completes the proof.

## B. Proof of Theorem 2

Proof: According to equation (42), we can derive that

$$
\begin{array} { l } { { \displaystyle { { \dot { \tilde { w } } } _ { i } } \left( t \right) = - \sigma _ { i w } { { \bar { \varphi } } _ { i } } \left( t \right) \left( { { { \bar { \varphi } } } _ { i } ^ { \mathrm { T } } \left( t \right) { { \tilde { w } } _ { i } } \left( t \right) + { { { \bar { \varepsilon } } } _ { i H } } \left( t \right) } \right) } } \\ { ~ - \sigma _ { i w } \displaystyle \sum _ { p = 1 } ^ { l } { { { { \bar { \varphi } } } _ { i } } \left( { { t } _ { p } } \right) \left( { { { \bar { \varphi } } } _ { i } ^ { \mathrm { T } } \left( { { t } _ { p } } \right) { { \tilde { w } } _ { i } } \left( t \right) + { { { \bar { \varepsilon } } } _ { i H } } \left( { { t } _ { p } } \right) } \right) } . } \end{array}\tag{64}
$$

Introduce a Lyapunov function

$$
V _ { i w } \left( \tilde { w } _ { i } \left( t \right) \right) = \frac { 1 } { { 2 { \sigma _ { i w } } } } \tilde { w } _ { i } ^ { \mathrm { T } } \left( t \right) \tilde { w } _ { i } \left( t \right) .\tag{65}
$$

Then, substituting the dynamics (64) into $\dot { V } _ { i w }$ yields

$$
\begin{array} { c } { { \displaystyle \dot { V } _ { i w } = - \tilde { w } _ { i } ^ { \mathrm { T } } ( t ) ( \bar { \varphi } _ { i } ( t ) \bar { \varphi } _ { i } ^ { \mathrm { T } } ( t ) + \displaystyle \sum _ { p = 1 } ^ { l } \bar { \varphi } _ { i } ( t _ { p } ) \bar { \varphi } _ { i } ^ { \mathrm { T } } ( t _ { p } ) ) \tilde { w } _ { i } ( t ) } } \\ { { -  \tilde { w } _ { i } ^ { \mathrm { T } } ( t ) ( \bar { \varphi } _ { i } ( t ) \bar { \varepsilon } _ { i H } ( t ) + \displaystyle \sum _ { p = 1 } ^ { l } \bar { \varphi } _ { i } ( t _ { p } ) \bar { \varepsilon } _ { i H } ( t _ { p } ) ) . } } \end{array}\tag{66}
$$

Note that for any variable $z , \| ( z / z ^ { T } z + 1 ) \| \le ( 1 / 2 )$ and $\| ( 1 / z ^ { T } z +$ $1 ) \parallel \leq 1$ / / / hold. Therefore, according to Assumption 2 and the definitions of $\bar { \varphi } _ { i }$ and $\bar { \varepsilon } _ { i H }$ , we have

$$
\bar { \varphi } _ { i } \left( t \right) \bar { \varepsilon } _ { i H } \left( t \right) + \sum _ { p = 1 } ^ { l } \bar { \varphi } _ { i } \left( t _ { p } \right) \bar { \varepsilon } _ { i H } \left( t _ { p } \right) \leq \frac { 1 } { 2 } \left( l + 1 \right) d _ { \varepsilon _ { i } } .\tag{67}
$$

Since $\bar { \varphi } _ { i } ( t ) \bar { \varphi } _ { i } ^ { \mathrm { T } } ( t ) > 0$ and $\bar { \varphi } _ { i } ^ { \mathrm { T } } ( t _ { p } ) \tilde { w } _ { i } ( t )$ is a scalar value, combined with inequality (67), the derivative $\dot { V } _ { i w } ( t )$ in (66) further satisfies

$$
\begin{array} { l } { { \displaystyle { { \dot { V } } _ { i w } } \left( t \right) \leq - \sum _ { p = 1 } ^ { l } { \left( { { \bar { \varphi } } _ { i } ^ { \mathrm { T } } \left( { t _ { p } } \right) { { \tilde { w } } _ { i } } \left( t \right) } \right) ^ { 2 } } + c _ { i w } \| { { \tilde { w } } _ { i } } \left( t \right) \| } } \\ { { \displaystyle \qquad = - \left\| { \Pi _ { i } ^ { \mathrm { T } } { { \tilde { w } } _ { i } } \left( t \right) } \right\| ^ { 2 } } + c _ { i w } \| { { \tilde { w } } _ { i } } \left( t \right) \| } \end{array}\tag{68}
$$

where $c _ { i w } = ( l + 1 / 2 ) d _ { \varepsilon _ { i } }$ is a positive constant.

/ εAdditionally, it follows from Assumption 2 that rank(Πi) = rank $\begin{array} { r } { { ( \Pi _ { i } \Pi _ { i } ^ { \mathrm { T } } ) = m } } \end{array}$ This means that $\mathrm { \bar { I I } } _ { i } \Pi _ { i } ^ { \mathrm { T } }$ is the positive definite. Then, it has $\sigma _ { \operatorname* { m i n } } ^ { 2 } ( \Pi _ { i } ) \lVert \tilde { w } _ { i } \rVert ^ { 2 } \leq \tilde { w } _ { i } ^ { \mathrm { T } } \Pi _ { i } \dot { \Pi } _ { i } ^ { \mathrm { T } } \tilde { w } _ { i } = \lVert \Pi _ { i } ^ { \mathrm { T } } \tilde { w } _ { i } \rVert ^ { 2 } \leq$ $\sigma _ { \mathrm { m a x } } ^ { 2 } ( \Pi _ { i } ) \lVert \tilde { { w } } _ { i } \rVert ^ { 2 }$ , where $\sigma _ { \operatorname* { m a x } } ( \Pi _ { i } )$ denotes the maximum singular σvalue of $\Pi _ { i } .$ σ. In view of the above properties, we can further show that the inequality (67) satisfies

$$
\dot { V } _ { i w } \left( t \right) \leq - \sigma _ { \operatorname* { m i n } } ^ { 2 } \left( \Pi _ { i } \right) \lVert \tilde { w } _ { i } \left( t \right) \rVert ^ { 2 } + c _ { i w } \lVert \tilde { w } _ { i } \left( t \right) \rVert .\tag{69}
$$

Consequently, $\dot { V } _ { i w } < 0 \mathrm { i f } \| \tilde { w } _ { i } ( t ) \| > c _ { i w } / \sigma _ { \operatorname* { m i n } } ^ { 2 } ( \Pi _ { i } )$ . Using the < > /σLyapunov extension theorem [36], this further indicates that $\tilde { w } _ { i } ( t )$ is UUB and converges to a compact set

$$
\Omega _ { \tilde { w } _ { i } } = \left\{ \tilde { w } _ { i } \| | \tilde { w } _ { i } | \| \leq \frac { c _ { i w } } { \sigma _ { \operatorname* { m i n } } ^ { 2 } \left( \Pi _ { i } \right) } \right\} .
$$

This completes the proof.

## C. Proof of Inequality (48)

Note that, for any two n-dimensional vectors $z _ { 1 }$ and $z _ { 2 } ,$ , by Young’s inequality [6] and the complete square formula, the following inequalities hold:

$$
\begin{array} { c } { 2 z _ { 1 } ^ { T } z _ { 2 } \leq z _ { 1 } ^ { T } z _ { 1 } + z _ { 2 } ^ { T } z _ { 2 } = \| z _ { 1 } \| ^ { 2 } + \| z _ { 2 } \| ^ { 2 } } \\ { \| z _ { 1 } + z _ { 2 } \| ^ { 2 } \leq 2 \| z _ { 1 } \| ^ { 2 } + 2 \| z _ { 2 } \| ^ { 2 } . } \end{array}\tag{70}
$$

Using the above inequality properties, combined with the definition of $\epsilon _ { i u _ { i } ^ { * } }$ in (32) and Assumptions 1 and 2, we have

$$
\begin{array} { r l } & { \| \alpha _ { i } \left( \operatorname { t a n h } { ( \Lambda _ { i } ) } \mathrm { - t a n h } \left( \hat { \Lambda } _ { i } \right) \right) - \epsilon _ { i u _ { i } ^ { * } } \| ^ { 2 } } \\ & { \quad \leq 2 \| \alpha _ { i } \| ^ { 2 } \left\| \operatorname { t a n h } { ( \Lambda _ { i } ) } \mathrm { - t a n h } \left( \hat { \Lambda } _ { i } \right) \right\| ^ { 2 } + 2 \| \epsilon _ { i u _ { i } ^ { * } } \| ^ { 2 } } \\ & { \quad \leq 4 \| \alpha _ { i } \| ^ { 2 } \left( \| \operatorname { t a n h } { ( \Lambda _ { i } ) } \| ^ { 2 } + \left\| \operatorname { t a n h } \left( \hat { \Lambda } _ { i } \right) \right\| ^ { 2 } \right) } \\ & { \quad + 2 \| 0 . 5 \left( I _ { 3 } - \Phi _ { i } \left( \varsigma _ { i } \left( X _ { i } \right) \right) \right) G _ { i } ^ { \mathrm { T } } \nabla \epsilon _ { i } \left( X _ { i } \right) \| ^ { 2 } } \\ & { \quad \leq 2 4 \| \alpha _ { i } \| ^ { 2 } + 2 b _ { G } ^ { 2 } \left( \epsilon _ { i d } ^ { d } \right) ^ { 2 } } \end{array}\tag{71}
$$

where the last inequality is derived by using $\| \operatorname { t a n h } ( * _ { i } ) \| \leq \sqrt { 3 }$ $( * = \Lambda , \hat { \Lambda } )$ and $| | I _ { 3 } - \mathrm { d i a g } \{ \operatorname { t a n h } ^ { 2 } ( \varsigma _ { i j } ( X _ { i } ) ) \} | | \leq 2$ proposed in [28]. , ςHence, based on inequalities in (70) and (71), inequality (48) can be further derived as follows:

$$
\begin{array} { r l } & { \nabla V _ { i c } ^ { * } G _ { i } \left( \widehat { u } _ { i } - u _ { i } ^ { * } \right) \leq \left. \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \right. ^ { 2 } } \\ & { \qquad + \left. \alpha _ { i } \left( \operatorname { t a n h } { ( \Lambda _ { i } ) } - \operatorname { t a n h } { \left( \widehat { \Lambda } _ { i } \right) } \right) - \epsilon _ { i u _ { i } ^ { * } } \right. ^ { 2 } } \\ & { \qquad \leq \left. \alpha _ { i } \operatorname { t a n h } ^ { - 1 } \left( \alpha _ { i } ^ { - 1 } \left( u _ { i } ^ { * } - \bar { \beta } _ { i } \right) \right) \right. ^ { 2 } } \\ & { \qquad + 2 4 \Vert \alpha _ { i } \Vert ^ { 2 } + 2 b _ { G } ^ { 2 } \left( \epsilon _ { i d } ^ { d } \right) ^ { 2 } . } \end{array}\tag{72}
$$

## ACKNOWLEDGMENT

The authors are grateful to Prof. Maolong Lv, Dr. Boyang Zhang, and a graduate student Zifei Wang for both their contributions in simulation part and helpful comments.

## REFERENCES

[1] C. Yuan, Y. Zhang, and Z. Liu, “A survey on technologies for automatic forest fire monitoring, detection, and fighting using unmanned aerial vehicles and remote sensing techniques,” Can. J. Forest Res., vol. 45, no. 7, pp. 783–792, Jul. 2015.

[2] P. Sun, B. Zhu, Z. Zuo, and M. V. Basin, “Vision-based finite-time uncooperative target tracking for UAV subject to actuator saturation,” Automatica, vol. 130, Aug. 2021, Art. no. 109708.

[3] H. Huang and A. V. Savkin, “Aerial surveillance in cities: When UAVs take public transportation vehicles,” IEEE Trans. Autom. Sci. Eng., vol. 20, no. 2, pp. 1069–1080, Apr. 2023.

[4] T. Balch and R. C. Arkin, “Behavior-based formation control for multirobot teams,” IEEE Trans. Robot. Autom., vol. 14, no. 6, pp. 926–939, Dec. 1998.

[5] C. Liu, L. Liu, J. Cao, and M. Abdel-Aty, “Intermittent event-triggered optimal leader-following consensus for nonlinear multi-agent systems via actor-critic algorithm,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 8, pp. 3992–4006, Aug. 2023.

[6] P. Wang, C. Yu, and Y.-J. Pan, “Finite-time output feedback cooperative formation control for marine surface vessels with unknown actuator faults,” IEEE Trans. Control Netw. Syst., vol. 10, no. 2, pp. 887–899, Jun. 2023.

[7] Y. Xu, J. Sun, Y.-J. Pan, and Z.-G. Wu, “Optimal tracking control of heterogeneous MASs using event-driven adaptive observer and reinforcement learning,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 4, pp. 5577–5587, Apr. 2024.

[8] P. Wang, C. Yu, M. Lv, and J. Cao, “Adaptive fixed-time optimal formation control for uncertain nonlinear multiagent systems using reinforcement learning,” IEEE Trans. Netw. Sci. Eng., vol. 11, no. 2, pp. 1729–1743, Mar. 2024.

[9] P. Wang, C. Yu, and M. Lv, “Optimized formation control of nonlinear systems with full-state constraints using adaptive fixed-time techniques,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 3331–3344, 2024, doi: 10.1109/TASE.2024.3392936.

[10] Y. Guo, G. Chen, and T. Zhao, “Learning-based collision-free coordination for a team of uncertain quadrotor UAVs,” Aerosp. Sci. Technol., vol. 119, Dec. 2021, Art. no. 107127.

[11] Z. Xia et al., “Multi-agent reinforcement learning aided intelligent UAV swarm for target tracking,” IEEE Trans. Veh. Technol., vol. 71, no. 1, pp. 931–945, Jan. 2022.

[12] W. Wang, X. Chen, J. Jia, K. Wu, and M. Xie, “Optimal formation tracking control based on reinforcement learning for multi-UAV systems,” Control Eng. Pract., vol. 141, Dec. 2023, Art. no. 105735.

[13] Z. Yu, J. Li, Y. Xu, Y. Zhang, B. Jiang, and C.-Y. Su, “Reinforcement learning-based fractional-order adaptive fault-tolerant formation control of networked fixed-wing UAVs with prescribed performance,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 3, pp. 3365–3379, Mar. 2024.

[14] S. Bandyopadhyay and S. Bhasin, “Safe reinforcement learning control for continuous-time nonlinear systems without a backup controller,” 2022, arXiv:2209.08922.

[15] F. Blanchini, “Set invariance in control,” Automatica, vol. 35, no. 11, pp. 1747–1767, Nov. 1999.

[16] L. Brunke et al., “Safe learning in robotics: From learning-based control to safe reinforcement learning,” Annu. Rev. Control, Robot., Auto. Syst., vol. 5, no. 1, pp. 411–444, May 2022.

[17] I. M. Mitchell, A. M. Bayen, and C. J. Tomlin, “A time-dependent Hamilton–Jacobi formulation of reachable sets for continuous dynamic games,” IEEE Trans. Autom. Control, vol. 50, no. 7, pp. 947–957, Jul. 2005.

[18] J. F. Fisac, A. K. Akametalu, M. N. Zeilinger, S. Kaynama, J. Gillula, and C. J. Tomlin, “A general safety framework for learning-based control in uncertain robotic systems,” IEEE Trans. Autom. Control, vol. 64, no. 7, pp. 2737–2752, Jul. 2019.

[19] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, “Control barrier function based quadratic programs for safety critical systems,” IEEE Trans. Autom. Control, vol. 62, no. 8, pp. 3861–3876, Aug. 2017.

[20] L. Wang, E. A. Theodorou, and M. Egerstedt, “Safe learning of quadrotor dynamics using barrier certificates,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2018, pp. 2460–2465.

[21] Z. Marvi and B. Kiumarsi, “Safe reinforcement learning: A control barrier function optimization approach,” Int. J. Robust Nonlinear Control, vol. 31, no. 6, pp. 1923–1940, Apr. 2021.

[22] S. Liu, L. Liu, and Z. Yu, “Safe reinforcement learning for affine nonlinear systems with state constraints and input saturation using control barrier functions,” Neurocomputing, vol. 518, pp. 562–576, Jan. 2023.

[23] M. H. Cohen and C. Belta, “Safe exploration in model-based reinforcement learning using control barrier functions,” Automatica, vol. 147, Jan. 2023, Art. no. 110684.

[24] F. M. Golmisheh and S. Shamaghdari, “Heterogeneous optimal formation control of nonlinear multi-agent systems with unknown dynamics by safe reinforcement learning,” Appl. Math. Comput., vol. 460, Jan. 2024, Art. no. 128302.

[25] N.-M.-T. Kokolakis and K. G. Vamvoudakis, “Safety-aware pursuitevasion games in unknown environments using Gaussian processes and finite-time convergent reinforcement learning,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 3, pp. 3130–3143, Mar. 2024.

[26] H. Modares, F. L. Lewis, and M.-B. Naghibi-Sistani, “Integral reinforcement learning and experience replay for adaptive optimal control of partially-unknown constrained-input continuous-time systems,” Automatica, vol. 50, no. 1, pp. 193–202, Jan. 2014.

[27] D. Liu, X. Yang, D. Wang, and Q. Wei, “Reinforcement-learningbased robust controller design for continuous-time uncertain nonlinear systems subject to input constraints,” IEEE Trans. Cybern., vol. 45, no. 7, pp. 1372–1385, Jul. 2015.

[28] X. Yang and B. Zhao, “Optimal neuro-control strategy for nonlinear systems with asymmetric input constraints,” IEEE/CAA J. Autom. Sinica, vol. 7, no. 2, pp. 575–583, Mar. 2020.

[29] J. Qiao, M. Li, and D. Wang, “Asymmetric constrained optimal tracking control with critic learning of nonlinear multiplayer zero-sum games,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 4, pp. 5671–5683, Apr. 2024.

[30] D. Zhang, Y. Wang, K. Jiang, and L. Liang, “Safe optimal robust control of nonlinear systems with asymmetric input constraints using reinforcement learning,” Int. J. Speech Technol., vol. 54, no. 1, pp. 1–13, Jan. 2024.

[31] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in Proc. IEEE 18th Eur. Control Conf. (ECC), 2019, pp. 3420–3431.

[32] M. H. Cohen, P. Ong, G. Bahati, and A. D. Ames, “Characterizing smooth safety filters via the implicit function theorem,” IEEE Control Syst. Lett., vol. 7, pp. 3890–3895, 2023.

[33] S. Bandyopadhyay and S. Bhasin, “Lagrangian-based online safe reinforcement learning for state-constrained systems,” 2023, arXiv:2305.12967.

[34] F. L. Lewis, D. Vrabie, and V. L. Syrmos, Optimal Control. Hoboken, NJ, USA: Wiley, 2012.

[35] W. Rudin, Principles of Mathematical Analysis, 3rd ed., 3, Ed., Singapore: McGraw-Hill, 1976.

[36] R. W. Beard, G. N. Saridis, and J. T. Wen, “Galerkin approximations of the generalized Hamilton–Jacobi–Bellman equation,” Automatica, vol. 33, no. 12, pp. 2159–2177, Dec. 1997.

[37] F. L. Lewis, S. Jagannathan, and A. Yesildirek, Neural Network Control of Robot Manipulators and Nonlinear Systems. London, U.K.: Taylor and Francis, 1999.

[38] J. Du, T. Lin, C. Jiang, Q. Yang, C. F. Bader, and Z. Han, “Distributed foundation models for multi-modal learning in 6G wireless networks,” IEEE Wireless Commun., vol. 31, no. 3, pp. 20–30, Jun. 2024.

<!-- image-->

Ping Wang (Member, IEEE) received the Ph.D. degree in control science and engineering from the School of Automation, Beijing Institute of Technology, Beijing, China, in 2023.

From July 2023 to October 2025, she held a postdoctoral position with the School of Automation and Intelligent Manufacturing, Southern University of Science and Technology, Shenzhen, China. From 2021 to 2022, she was a Visiting Ph.D. student with the Department of Mechanical Engineering, Dalhousie University, Halifax, NS, Canada. She is currently with the School of Automation, Beijing Information Science and Technology University, Beijing. Her current research interests include optimal control of nonlinear system, safe reinforcement learning (RL) control, formation control of multiagent systems, and fully actuated system theory.

<!-- image-->

Chengpu Yu (Senior Member, IEEE) received the B.E. and M.E. degrees in electrical engineering from the University of Electronic Science and Technology of China, Chengdu, China, in 2006 and 2009, respectively, and the Ph.D. degree in electrical engineering from Nanyang Technological University, Singapore, in 2014.

From 2013 to 2014, he was with the Internet of Things Laboratory, Nanyang Technological University, as a Research Associate. From June 2014 to August 2017, he was a Post-Doctoral with Delft

Center for Systems and Control, Delft, The Netherlands. He is currently a Professor with the School of Automation, Beijing Institute of Technology, Beijing, China. His research interests include system identification, distributed optimization, and network system control.

<!-- image-->

Fang Deng (Fellow, IEEE) received the B.E. and Ph.D. degrees in control science and engineering from Beijing Institute of Technology, Beijing, China, in 2004 and 2009, respectively.

He is currently a Professor with the School of Automation, Beijing Institute of Technology, and Beijing Institute of Technology, Zhuhai, China. His research interests include intelligent fire control, intelligent information processing, and smart wearable devices.

<!-- image-->

Jie Chen (Fellow, IEEE) received the B.Sc., M.Sc., and Ph.D. degrees in control theory and control engineering from Beijing Institute of Technology, Beijing, China, in 1986, 1996, and 2001, respectively.

From 1989 to 1990, he was a Visiting Scholar with California State University, Long Beach, CA, USA. From 1996 to 1997, he was a Research Fellow with the School of Engineering, University of Birmingham, Birmingham, U.K. He was the President of Tongji University, Shanghai, China, from 2018 to 2023. He serves as the Director of the National Key Laboratory of Autonomous Intelligent Unmanned Systems (KAIUS), Beijing. He is currently the Party Secretary with Harbin Institute of Technology, Harbin, China. His research interests include complex systems, multiagent systems, multiobjective optimization and decision, and safety control.

Dr. Chen is a fellow of IFAC and a member of the Chinese Academy of Engineering.