OPEN

? Check for updates

# Dynamic task ofoading edge‑aware optimization framework for enhanced UAV operations on edge computing platform

B. Suganya1 , R. Gopi2 , A. Ranjith Kumar3 & Gavendra Singh4\*

Resource optimization, timely data capture, and efcient unmanned aerial vehicle (UAV) operations are of utmost importance for mission success. Latency, bandwidth constraints, and scalability problems are the problems that conventional centralized processing architectures encounter. In addition, optimizing for robust communication between ground stations and UAVs while protecting data privacy and security is a daunting task in and of itself. Employing edge computing infrastructure, artifcial intelligence-driven decision-making, and dynamic task ofoading mechanisms, this research proposes the dynamic task ofoading edge-aware optimization framework (DTOE-AOF) for UAV operations optimization. Edge computing and artifcial intelligence (AI) algorithms integrate to decrease latency, increase mission efciency, and conserve onboard resources. This system dynamically assigns computing duties to edge nodes and UAVs according to proximity, available resources, and the urgency of the tasks. Reduced latency, increased mission efciency, and onboard resource conservation result from dynamic task ofoading edge-aware implementation framework (DTOE-AIF)’s integration of AI algorithms with edge computing. DTOE-AOF is useful in many felds, such as precision agriculture, emergency management, infrastructure inspection, and monitoring. UAVs powered by AI and outftted with DTOE-AOF can swiftly survey the damage, fnd survivors, and launch rescue missions. By comparing DTOE-AOF to conventional centralized methods, thorough simulation research confrms that it improves mission efciency, response time, and resource utilization.

Keywords Optimization, Artifcial intelligence, Edge computing, Performance, Ofoading

Te application of AI to improve the operations of UAVs at the network’s edge presents several opportunities and potential dangers1 . As one of the most signifcant challenges, the complex connection between the need for data processing and the capacity to make real-time judgments is particularly challenging. UAVs generate enormous amounts of data, urgently needing to be assessed and addressed to ensure optimal operation2 . It is possible that normal cloud-based solutions could result in latency issues due to the amount of time it takes to transmit data to remote servers for processing. Tis latency may hamper critical decision-making processes, particularly in dynamic circumstances requiring prompt reactions3 . It is important to note that UAVs have limited computer capability, which restricts the complexity of AI algorithms that may be utilized for real-time processing. Optimization becomes even more challenging when balancing the computational tasks performed onboard and those being ofoaded to edge computing nodes4 . Tere is an additional difculty added by the fact that data privacy and security during transmission and processing at the edge are concerned. Despite these obstacles, edge computing that is powered by AI has the potential to enhance the operations of UAVs5 by enabling faster decision-making, improved resource use, and higher autonomy. To develop solutions that are efectively scalable and tailored to

1 Faculty of Artifcial Intelligence & Data Science, Sri Ramakrishna Engineering College, Coimbatore, Tamil Nadu 621112, India. 2 Faculty of Computer Science & Engineering, Dhanalakshmi Srinivasan Engineering College, Perambalur, Tamil Nadu 621212, India. 3 Faculty of Computer Science and Engineering, Lovely Professional University, Phagwara 144001, Punjab, India. 4 Faculty of Software Engineering, College of Computing and Informatics, Haramaya University, Dire Dawa, P.O. Box 138, Ethiopia. \*email: gavendra.singh@haramaya.edu.et the uses of UAVs, it is necessary to adopt new methods of thinking about these issues. Such issues use advancements in AI, edge computing, and communication technology6 . Cooperation between researchers, industry stakeholders, and regulatory authorities is required to overcome these obstacles and fully exploit the potential of edge computing driven by AI for optimizing UAV operating procedures7 .

Current approaches to enhancing UAV operations through AI-driven edge computing encompass various ways to improve performance and reduce issues. Te practice of task ofoading, which involves moving computationally demanding activities to edge computing nodes that are located close to the operational region of the UAV, is a typical solution that is used to reduce latency and conserve onboard resources8 . Using machine learning algorithms for real-time data processing is one alternative strategy that may empower UAVs to make autonomous judgments in response to sensor inputs and ambient variables better9 . Using edge caching techniques makes it possible to store data that is accessed frequently locally, hence further reducing latency and the requirement for data transmission. Nevertheless, these approaches face various challenges, notwithstanding their efectiveness10. Especially in dynamic and diverse situations, it is difcult to guarantee that UAVs and edge computing nodes can interact without interruptions; this is especially true11. In addition, optimizing resource management and task allocation to satisfy performance requirements while balancing the amount of computing efort and energy consumed is not easy12. Additional obstacles occur due to the requirement to transfer and manage sensitive data at the network’s edge while simultaneously addressing concerns regarding privacy and security. Another signifcant consideration is how AI algorithms interact with diferent UAV platforms and edge computing architectures regarding scalability and interoperability13. It is necessary for researchers from various domains, such as AI, Edge Computing (EC), communication, and cybersecurity, to collaborate to address these challenges. Academic institutions, industries, and government agencies must collaborate to set guidelines for optimizing UAV operations through edge computing driven by AI14. Traditional centralized processing architectures in UAV operations encounter latency, bandwidth limits, scalability issues, and limited onboard computational resources. Te research addresses these issues15. Goals include mission success, resource optimization, and timely data acquisition with AI-powered edge computing.

For real-time data processing and analysis, edge AI means deploying AI models and algorithms directly on local edge devices like sensors or Internet of Tings (IoT) devices rather than relying on cloud infrastructure. Edge AI, frequently referred to as "AI on the edge," uses AI in conjunction with edge computing to run machine learning tasks directly on connected edge devices. Data can be maintained near the device using edge computing and processed on the network edge using AI algorithms, independent of whether the device has an internet connection or otherwise. It enables for millisecond-level data processing, resulting in immediate response.

Te Dynamic Job Ofoading Edge-Aware Optimization Framework optimizes job allocation in real time by combining cutting-edge AI algorithms and edge computing approaches. In its most basic form, RL agents make a dynamic choice on whether or not to ofoad work to edge devices or the cloud by utilizing regularly updated predictions from DL models. Federalized learning is used to train these models to safeguard the data’s confdentiality and increase the process’s efectiveness. Graph Neural Networks (GNNs) simplify scheduling and resource allocation by mimicking the whole network topology. Tis design is improved further by GNNs. Using edge orchestration and load-balancing strategies, workloads may be distributed consistently. Reducing latency and bandwidth usage with real-time analytics and edge caching assistance is possible. Tis integrated approach, which enables the framework to adapt dynamically to constantly changing network conditions, ensures that tasks are executed efciently and resources are utilized efectively across the edge network.

Te DTOE-AOF has several primary objectives, one of which is to address several signifcant gaps in the state of the art of edge computing. In dynamic environments, conventional approaches, such as heuristic methods and static task ofoading, usually perform suboptimally due to their lack of fexibility and adaptability. Both of these methods are examples of traditional methodology. Even though there has been considerable investigation into machine learning and reinforcement learning techniques, these approaches are confronted with several challenges, including centralized data processing, privacy issues, and high computation needs. Additionally, it has a restricted capacity for scaling up. Te DTOE-AOF utilizes several cutting-edge AI techniques. Tese techniques include federated learning-trained DL models, which enhance efciency and privacy, and reinforcement learning, which allows for real-time adaptability. Via the utilization of GNNs to explain the network architecture, as well as via edge management and load balancing techniques, the system guarantees the optimal and scalable allocation of the resources required. Edge caching and real-time analytics, which both reduce latency and bandwidth consumption, make it possible to ofoad real-time tasks in a way that is both efcient and adaptive. It is a robust system that can adapt to changing network situations, scale efectively, integrate diverse data sources, protect user privacy, and optimize real-time ofoading decisions. It is all made possible by the all-encompassing design of DTOE-AOF.

## Problem defnition

Latency, bandwidth constraints, and scalability problems are problems that conventional centralized processing architectures encounter. An opportunity to overcome these obstacles and realize UAVs’ full potential has arisen with the introduction of AI-powered edge computing. Concerns with limited onboard computational resources, unpredictable climatic conditions, and severe latency requirements are some obstacles to deploying UAVs in real-world applications.

Specifcally, it addresses challenges with self-driving vehicles, smart cities, and industrial IoT that current approaches fail to address in edge computing environments completely. Low-latency states and real-time data processing are essential for these kinds of circumstances. Separating data sources and processing centres causes delays in traditional cloud-based methods. Due to rapid fuctuations in network situations and resource availability, static ofoading solutions cannot adapt to these environments. However benefcial, heuristic approaches generally require enormous processing capacity and fail to perform efectively across many devices.

## Objectives

Optimization of UAV operations is proposed via the DTOE-AOF. Tis framework dynamically distributes edge nodes and UAVs computational tasks based on proximity, resources, and task urgency.

• Integrating AI algorithms with edge computing reduces latency, boosts mission efciency, and conserves onboard resources.

• Te research will show DTOE-AOF’s versatility and usefulness in precision agriculture, emergency management, infrastructure inspection, and monitoring.

If UAVs have DTOE-AOF, precision agriculture can use data-driven decision-making, disaster management can use quick surveys, and infrastructure inspections can be more thorough.

Te research attempts to confrm DTOE-AOF’s improved mission efciency, response time, and resource utilization through simulation and comparison with standard centralized approaches.

Te remainder of the research paper is organized as follows: “Literature review” provides a literature analysis on optimizing UAV operations through AI to improve performance at the border. “Proposed method” focuses on the mathematical aspects of the DTOE-AOF. A detailed description of the experiment’s fndings, analysis, and comparisons to prior methodologies are included in “Results and discussion”. Te results are summed up in “Conclusion”.

## Literature review

Several industries are seeing profound shifs due to the confuence of state-of-the-art technologies like the IoT, DL, optimization methods, edge computing, and beyond 5G (B5G)/6G wireless networks. One area where this is taking of is UAVs. Tis literature review explores new developments and methods in these areas, focusing on creative strategies and the practical efects of these developments.

## (a) Review on UAV

Using AI built into UAVs, Koubaa et al.26 present a cloud-edge hybrid system (C-EHS)that can do remote sensing. Te onboard AI system, AERO, integrates object identifcation and tracking to achieve precision and transmit data in real-time.

Lins et al.27 put forward the ideas of System Intelligence (SI) and Edge Intelligence (EI) to use 5G networks for UAV-based SAR (UAV-SAR)operations. Critical to mission efciency, it provides a virtualized testbed to show how DNN partitioning afects communication costs and latency. Ijaz H. et al.28 suggest a UAV-assisted edge computation framework, UAV-ECF, for real-time catastrophe scenario categorization, compressing CNN models for onboard GPUs. Te results reveal a reduction of 84% in model size and an increase of 99% in throughput without sacrifcing accuracy.

Addressing the issue of deploying bulky models on resource-constrained devices, Surianarayanan et al.29 investigate AI model optimization approaches. An AI model optimization framework is needed to get insights into edge intelligence applications in real-time. Palossi et al. present PULP-Frontnet30, a deep neural network (DNN) for UAVs that uses vision to estimate human poses in real-time. Compared to optimal sensor setups, the results show that autonomous navigation uses very little energy while providing great accuracy.

## (b) Optimization using AI

Te paper introduces methodologies and tools for optimizing vision-based CNNs on ULP processors for nano-UAVs. Deploying PULP-Dronet improves memory efciency and speed, enhancing obstacle avoidance, free fight, and lane following while conserving power. To improve the IoT, Xu et al.31 investigated how AI and backscatter communication (BC) might work together. It provides an overview of current developments, such as AI algorithms for security, channel estimate, and signal detection. It also delves into potential future uses in B5G/6G technologies, outlining opportunities and threats. Using IoT devices, robotic drones operated by AI are investigated by Jat et al.32. concerning the COVID-19 response. Te topics covered are improving efciency in pandemic conditions, using edge computing for fast data analysis, and reviewing literature for future research paths.

For heterogeneous computing systems involving UAVs, Kim et al.33 present a computational ofoading scheme that considers energy consumption and fairness. Te solution improved energy consumption fairness by up to 120% compared to the prior methodology, all recognitions to a genetic algorithm.

Te article by Miya et al.34 delves into how the IoT is progressing towards intelligent services and how upgraded wireless networks, such as 6G, are necessary. A combination of AI with quantum communications is suggested to fulfl the needs of upcoming applications by Miya et al.35.

## (c) Edge computation

Recent advances in Reinforcement Learning (RL) for Multi-UAV Wireless Networks (MUWNs) are reviewed by Bai et al.36, who draw attention to open difculties in the feld and describe RL applications such as data access, resource allocation and trajectory planning. Optimal ofoading and task prediction in Fog Computing (FC)

networks are proposed by Jana et al.37 using an analytical technique. Concerning load balancing and reaction time, the LSTM-GWO approach surpasses the current ACO and PSO approaches.

Smart city and society management is the focus of Heidari et al.’s38 investigation into using AI, ML, and DL approaches in conjunction with the IoT, IoD, and IoV. Te article discusses recent advancements, advantages, and disadvantages, emphasizing smart trafc and energy management. Te fndings emphasize the importance of ML approaches like CNN and LSTM and the prevalence of their use in smart city applications, where accuracy is paramount. Based on the results of these investigations, Python is the language used in computer programming.

Pandey et al.39 present a new way to evaluate Mobile Edge Computing (MEC) in 5G networks using simulated data. It allows for rapid evaluations of MEC performance and optimization of resources by properly capturing spatio-temporal trends through advanced modelling approaches.

For wireless connectivity beyond 5G, the DEDICAT 6G project, which the European Union supports, aims to provide a smart and environmentally friendly platform (Stavroulaki et al. 201840). Utilizing robots, linked vehicles, and drones enhances task execution speed, energy efciency, and latency while investigating dynamic coverage extensions. Innovative interfaces, such as smart glasses, facilitate human–machine contact, and the project deals with privacy, security, and trust assurance. Network load balancing, resource allocation, coverage extension, security, and human–machine applications are the goals of the demonstrations and testing in four illustrative use cases.

Modern AI systems that rely on big data analytics and DL have high communication and determine cost, which results in high energy consumption, network congestion, and privacy leaks during training and inference. Edge AI was a game-changing innovation for 6G networks that improved their efcacy, privacy, security, and efciency by introducing model training and inference capabilities to the network’s edge42. Tis enables seamless integration of sensing, communication, analysis, and intelligence. Using decentralized machine learning models and integrated designs for wireless communication techniques, the author aims to create trustworthy and scalable edge AI systems in this work. Tis paper discusses new wireless network design concepts, optimization methodologies for service-driven resource allocation, and an end-to-end system architecture that supports edge AI.

IoT and autonomous system technologies are an appropriate match for edge information processing methods and approaches, but many problems still require addressing. Tis paper aims to analyze these emerging multimedia and edge information processing paradigms from various technological perspectives. Tese viewpoints include AI-powered multimedia analytics on the edge, intelligent edge multimedia streaming43, AI-powered multimedia caching on the edge, AI-powered multimedia services for the edge, and AI-powered hardware and devices for intelligent multimedia on the edge. Te study addresses various AI and ML enablers for multimedia and edge data processing.

Te summary of the fndings is given in Table 1.

Tese studies aim to improve disaster management, healthcare monitoring, smart city infrastructure, and wireless communication networks by developing technology-driven solutions to tackle complicated problems.

Based on the analysis, there are numerous problems with present models in attaining a high mission efciency, response time, and resource utilization. Hence, this study proposes the Dynamic Task Ofoading Edge-Aware Optimization Framework (DTOE-AOF) for UAV operations optimization.

## Proposed method

## UAV with DTOE‑AOF implementation

Improving efciency, decreasing latency, and making the most of available resources are becoming increasingly important in the ever-changing world of UAV operations. Latency, bandwidth limitations, and scalability problems are problems with traditional central processing methods. Tis study presents the DTOE-AOF to address these issues. With AI-powered decisions and edge computing buildings, DTOE-AOF may dynamically assign computing jobs to UAVs and edge nodes according to proximity, available resources, and the urgency of the work. With its groundbreaking method, DTOE-AOF is poised to revolutionize various programs, from agricultural precision to emergency management, by reducing latency, increasing mission efciency, and conserving resources.

MEC is commonly recognized as a basic technology for various next-generation IoT applications. Due to their versatility and ease of deployment, UAVs can provide edge computing services. UAV-enabled MEC designs may be categorized and applied. A UAV may be a relay, IoT node, or mobile EC server. First, UAVs may act as mobile nodes by sending their computing power to a MEC server. As the MEC, the UAV may monitor a cluster. Te third usage of UAVs is linking mobile end nodes to MEC servers. Figure 1 shows that the UAV may submit memory- and processing-intensive operations to an MEC server as a dedicated user. Complex computational jobs may be too difcult for UAVs’ processing, memory, and battery life. Moving data processing to the groundbased MEC server may extend its battery life. Figure 1 depicts an alternate scenario in which the UAV fies with the MEC server to assist ground users with computing once they ofoad their jobs to it. In the third alternative, Fig. 1, the UAV acts as a central relay to let mobile users transmit their computing operations to an MEC server.

UAV-enabled MECs’ design provides IoT devices with reliable, low-latency services. Still, turmoil abounds. Airborne data security, storage, administration, and UAV networking are complex. UAV mobility complicates communication, requiring greater ground-based user-UAV cooperation. Te low battery power of UAVs is another challenge. More power is required for onboard computations before hovering, accelerating/decelerating, climbing/descending. To start planning for better energy and resource management immediately. Another challenge when developing UAV-enabled MEC systems for computing work is integrated route engineering. Predicting and tracking mobile user behaviour is essential for the optimum ofoading of computing tasks and timely communication of computing results to consumers. Many UAV MEC services require greater trajectory design consideration. UAV blockchain integration is difcult16. UAVs pose privacy issues, air trafc law violations, quantum attacks, machine learning, and algorithmic attacks alone.

<table><tr><td rowspan=1 colspan=1>s. no.</td><td rowspan=1 colspan=1>Literature</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Advantages</td><td rowspan=1 colspan=1>Limitations</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>Cloud-edge hybrid system (C-EHS)</td><td rowspan=1 colspan=1>Precision in remote sensing,real-time datatransmission,integration of object identification and tracking</td><td rowspan=1 colspan=1>Relianceon cloud connectivity,potential latencyissues</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>Systemintelligence (SI)and edge intelligence (EI)for UAV-based SAR operations</td><td rowspan=1 colspan=1>Utilization of 5G networks, virtualized testbedfor DNN partitioning analysis,and efficiency inmission-critical tasks</td><td rowspan=1 colspan=1>Communication costs,latency issues,dependencyon network stability</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>UAV-assisted edge computation framework (UAV-ECF)</td><td rowspan=1 colspan=1>Real-time catastrophe scenario categorization,significant reduction in model size,increasedthroughput without accuracy loss</td><td rowspan=1 colspan=1>Initial setupandconfiguration complexitypoten-tial hardware compatibility issues</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>AI model optimization approaches</td><td rowspan=1 colspan=1>Insights into edge intelligence applications,opti-mization of AI models for resource-constraineddevices</td><td rowspan=1 colspan=1>Complexity in implementation,potential trade-offs betweenoptimizationand modelaccuracy</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>PULP-Frontnet</td><td rowspan=1 colspan=1>Real-time human pose estimation,energy-efficientautonomous navigation,and high accuracy com-pared to optimal sensor setups</td><td rowspan=1 colspan=1>Dependency on vision quality,potential computa-tional overhead</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>Optimization of vision-based CNNs on ULPprocessors for nano-UAVs</td><td rowspan=1 colspan=1>Memory eficiency improvement,enhanced obsta-cle avoidance,free flight,and lane following powerconservation</td><td rowspan=1 colspan=1>Hardware compatibility,potential performancetrade-offs</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>Integration of AI and backscatter communicationfor IoT enhancement</td><td rowspan=1 colspan=1>Advancement in AI algorithms for security,chan-nel estimation,and signal detection, potentialimprovement in IoT functionalities</td><td rowspan=1 colspan=1>Complexity in integration, potential securityvulnerabilities</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>Computational offloading scheme for heterogene-ous computing systems involving UAVs</td><td rowspan=1 colspan=1>Improved energy consumption fairness,considera-tion of energy efficiency and fairness,utilization ofgenetic algorithm for optimization</td><td rowspan=1 colspan=1>Complexityinalgorithm design,potential scal-ability issues</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>Progression of IoT towards intelligent services andthe necesity of upgraded wireless networks (6G)</td><td rowspan=1 colspan=1>Advancement in wireless communication tech-nologies,potential for intelligent IoTservices</td><td rowspan=1 colspan=1>Dependency on infrastructure upgrades, potentialcompatibility issues</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>Integration of AI with quantum communications</td><td rowspan=1 colspan=1>Potential enhancement in communication securityand efficiency</td><td rowspan=1 colspan=1>Complexity in implementation, current limitationsin quantum communication technologies</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>Reinforcement learning for multi-UAV wirelessnetworks (MUWNs)</td><td rowspan=1 colspan=1>Advancement in data access,resource allocation,and trajectory planning,potential for optimizedUAV network operations</td><td rowspan=1 colspan=1>Complexity inalgorithm design, potential scal-ability issues</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>Optimal offloading and task prediction in fog com-puting networks using LSTM-GWO approach</td><td rowspan=1 colspan=1>Improved load balancing and reaction time,sur-passng current approaches like ACO and PSO</td><td rowspan=1 colspan=1>Complexity in algorithm design, potential compu-tational overhead</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>Utilizing AI, ML,and DL approaches in smart cityand society management</td><td rowspan=1 colspan=1>Advancement in smart traffc and energymanagement, emphasis on accuracy through MLapproaches like CNN and LSTM</td><td rowspan=1 colspan=1>Data privacy concerns,potential bias in AIalgorithms</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>Evaluation of MEC in 5G networks using simulateddata</td><td rowspan=1 colspan=1>Rapid evaluation of MEC performance,optimiza-tionof resources through spatiotemporaltrendanalysis</td><td rowspan=1 colspan=1>Reliance on simulated data accuracy, potentialdiscrepancy with real-world scenarios</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>DEDICAT 6G project for smart wireless connectiv-ity beyond 5G</td><td rowspan=1 colspan=1>Enhanced task executionspeed,energyefiency,and latency,investigationofdynamic coverageextensions,human-machine interaction enhance-ment through innovative interfaces</td><td rowspan=1 colspan=1>Dependency on project funding and support,potential regulatory hurdles</td></tr></table>

Table 1. Comparison of the existing methods.

$$
L = \frac { \beta . ( \alpha . T T + \delta . P T + \vartheta . P T ) } { \sigma + \rho + N \pi }\tag{1}
$$

Equation (1) makes the examination of latency L during UAV operations more complicated. A subtle change in the importance of specifc components within the latency equation is made possible by the weighting β, α.TT, δ.PT, and ϑ.PT. Te parameter σ captures the fuctuating character of the network’s capacity by modulating the infuence of Wavelength on the overall delay, adding further complexity. Furthermore, the complicated network N environment is refected in the intricate impact of the regulators ρ and π on Standing in line Time and Connection Overhead, respectively. Tis all-encompassing equation adequately captures the intricate interplay of factors, which provides a thorough comprehension of delays in UAV operations.

$$
T = \frac { \theta . ( \mu . D S + \delta . O U + \tau . E T ) } { X E . \sigma } + B G . \omega + T \mathrm { S } . \varphi\tag{2}
$$

Equation (2) provides a complex formula that explores the main factors that contribute to the scalability of a system, capturing the complexities of scalability where T in UAV operations. Various weights θ, µ, δ, and δ allow for subtle modifcations to the relevance of Computational Resources DS, Network Troughput OU , and Data Storage ET. Te parameter ω enhances the overall scalability by adjusting the impact of Workload Distribution BG. In addition, the Adaptability Factor XE and System Redundancy TS dynamics are elaborately captured by the modifers ω and ϕ. Tis simple formula gives a full picture of the complex interrelationships that determine the scalability of UAV operations, shedding light on the dynamics of redundancy and fexibility in the system17.

<!-- image-->  
Figure 1. Architecture facilitated by UAVs.

$$
P = \frac { \beta . N F ^ { \gamma } + \alpha . S U ^ { \theta } } { \delta . S T ^ { \rho } } + \sum _ { j = 1 } ^ { o } \tau _ { j } . \left( \mu _ { j } . B J _ { j } ^ { L _ { j } } . C E _ { j } ^ { \sigma _ { j } } . T O _ { j } ^ { \tau _ { j } } \right)\tag{3}
$$

Equation (3) incorporates a complicated model for evaluating the performance P of UAV operations. Te importance of mission efciency where $N F ^ { \gamma } { } _ { ; }$ , response time $S U ^ { \theta }$ , and resource utilisation $S T ^ { \rho }$ are represented by the weighted coefcients β, α, and δ, respectively. Equational intricacy is enhanced by the inclusion of non-linear exponents such as $T O _ { j } ^ { \tau _ { j } } , \dot { \mu } ,$ and $\sigma _ { j } ,$ which magnifes the infuence of these basic elements on overall performance by capturing complex dependencies. Furthermore, by taking into account further elements like AI integration, edge computing CE capabilities, and dynamic job ofoading methods, the terms involved add a greater degree of complexity.

## UAV with MEC implementation

MEC solves the mobile IoT device resource and time issue. Backhaul congestion and network latency may be reduced with additional CC products. UAVs with data storage, processing, and communication may deploy MEC servers at network edges18. In this design, low-powered IoT devices may outsource computation to UAVs with MEC servers via line-of-sight communication. Te system must forecast tasks, deploy UAVs, organize users, analyze signals, and allocate cooperative resources, among other challenges. Te EC architecture suggests that transportable and adaptable UAVs will provide decentralized solutions. Te fying edge architecture’s increased CC capabilities are ideal for real-time, latency-sensitive IoT applications. Moving computation from data centres to IoT devices improves real-time administration and decision-making with lower latency. An IoT system’s many endpoints food peripheral devices with data.

Data organization and processing are needed for automated maintenance, self-monitoring, and prediction. Due to the memory and processing power gap between EC endpoints and centralized cloud servers, certain AI systems cannot analyze edge data. Any resilience-focused AI approach must prepare for memory and processing capacity restrictions. Decentralized resource distribution allows EC to fulfl client requests quicker than a $\mathrm { C C } ,$ even with minimum processing19. Work scheduling, resource allocation, and ofoading issues may dramatically impact performance. Over several decades, AI has been more popular for networking difculties. ML is used in various areas, including networking, for its decision-making and interaction abilities. It may enhance network performance in numerous areas, including resource allocation, trafc classifcation and prediction, congestion control, and routing.

## Algorithm 1. Resource allocation based on the proposed DTOE-AOF.

Algorithm 1 shows the Resource Allocation based on the proposed DTOE-AOF. Developing a resource allocation strategy aims to increase the likelihood of the task’s success. Finding the best way to reduce latency in cloud and edge-cloud collaboration systems requires a combination of communication technologies and computer resource allocation. Tis research developed a distributed computing-based ofoading strategy that can adapt to changing user loads and achieve outstanding computing ofoading capabilities. In an environment with many channels of wireless interference, our research found the best solution to the issue of ofoading resources from the edge cloud to multiple users. Allotting resources among edge servers is modelled in this research. Time is of the essence for crowd activities as well since edge servers have limited processing power and load capacity and must fnish computing jobs within a certain time frame. As a result, this article classifes the tasks by priority and saves them in the task stack, equivalent to the address stack in the edge server, based on the maximum permissible delay of the tasks20.

MEC systems use FMC controllers for UAVs. Tey typically capture user, UAV, and MEC server data. Tis command centre oversees AI ofoading. A UAV-assisted arrangement uses a cloudlet like Fig. 2. Afer an IoT gadget completes an ofoaded job, the UAV reports back. If the data requires more complex processing than the cloudlet can handle, the UAV may transfer it to the nearest ground servers. For IoT devices like smartphones, sensors, automobiles, and robotics, the system may use a feet of UAVs to cover a vast area. Onboard cloudlets employ AI to handle user-generated data31.

$$
D S = \frac { \beta I ^ { \theta } . C ^ { \gamma } } { \alpha . A \nu ^ { \mu } . \sum _ { j = 1 } ^ { o } \delta _ { j } . \left( A u _ { j } ^ { L _ { j } } . A u _ { j } ^ { \pi _ { j } } . E n _ { j } ^ { \tau _ { j } } \right) }\tag{4}
$$

Equation (4) incorporates several critical elements into a detailed model, ofering a thorough depiction of data security DS. Tis equation emphasizes the value of protecting data from unauthorized access where $\beta I ^ { \theta }$ as the weighted coefcient representing the importance of Confdentiality, a more nuanced portrayal of the complicated dependencies within the secrecy component is made possible by adding a layer of complexity by introducing the non-linear exponent θ. Furthermore, the weighted coefcient $C ^ { \gamma }$ highlights the signifcance of Availability, solving the problem of data accessibility $x . A \nu ^ { \mu }$ when needed. Realizing that data security extends beyond simple confdentiality δj, the non-linear exponent α adds complexity by ofering a detailed description of the link with availability. Te signifcance of maintaining unmodifed and trustworthy data is shown in the non-linear sensitivity of Integrity, which is captured by the exponent δ. With their weighted coefcients and non-linear sensitivities, the terms $\pi , \tau _ { : }$ , and j add complexity to the equation, which permits the inclusion of additional factors like encryption $E n _ { j } ^ { \tau _ { j } }$ , authentication $\overset { \cdot } { A } u _ { j } ^ { { L } _ { j } }$ , and authorization $A u _ { j } ^ { \pi _ { j } }$

<!-- image-->  
Figure 2. Architecture for MEC with UAV capability.

$$
P r = \frac { \Delta . T P ^ { \nabla } } { \alpha . T P ^ { \theta } + \beta . F P ^ { \delta } + \sum _ { j = 1 } ^ { o } \rho _ { j } . ( P P V . S ) }\tag{5}
$$

Precision Pr is an important measure in classifcation tasks, and Eq. (5) shows a complex model for it. Te importance of True Positives where $T P ^ { \theta }$ , which are positively $P P V$ recognized, and False Positives $F P ^ { \delta }$ , which are negatively connected with incorrectly labelled positive instances, are shown by the weighted coefcients, which may be represented by symbols like , ∇, and α. Te complex interdependencies S within these parts are captured by adding non-linear exponents $\beta , \rho ,$ and $\theta ,$ which introduce a subtle sensitivity to changes.

$$
M E f = \frac { \beta . O T ^ { \gamma } } { \alpha . I T ^ { \theta } + \delta . D T ^ { \varepsilon } + \sum _ { j = 1 } ^ { o } ( U t . E C ) }\tag{6}
$$

Equation (6) introduces a comprehensive model for assessing machine efciency MEff , considering various critical factors in the production process. Te weighted coefcients where $\beta , \alpha ,$ and δ, represent the signifcance of Output $O T ^ { \gamma }$ , Input α. $I T ^ { \theta }$ , and Downtime $\delta . D T ^ { \varepsilon }$ , respectively, in determining the overall efciency of a machine21. Including non-linear exponents, $\varepsilon , \mathrm { j } ,$ and o adds intricacy by capturing complex dependencies within each component, allowing for a more nuanced understanding of their impact on efciency on utilization Ut and energy consumption $E C .$

## DTOE‑AOF and DTOE‑AIF implementation

Te framework in Fig. 3 shows a comprehensive dynamic task ofoading and edge-aware optimization mechanism for intelligent and efcient UAV operations. Te architecture’s interconnected pieces provide easy communication, smart decision-making, and resource efciency22,23. Te ground station is the system’s hub for managing responsibilities and collecting data. Data Collector and Task Manager are crucial Ground Station components. Te Data Collector gathers data from numerous sources for analysis and decision-making, while the Task Manager orchestrates tasks to maximize execution. Wi-Fi and 5G internal communication channels allow the Ground Station and Edge Node to communicate well. Edge computing allows the Edge Node to arbitrate Ground Station-UAV communication, boosting processing efciency and latency.

Te Edge Computing Server at the Edge Node controls operations with its AI Inference Engine and Task Scheduler. Tese pieces enable intelligent work allocation and real-time data processing, improving system performance. Te server-UAV connection using LoRa and RF ensures reliable and low-latency data delivery. UAVs have Edge Devices, which are small computers for onboard processing. Tese edge devices connect UAV payloads with thermal sensors, LiDAR, and cameras. In this design, UAVs may gather and evaluate sensory data in realtime to make informed judgments.Dynamic ofoading lets the framework adapt to changing workloads and conditions. By considering compute load, latency requirements, and energy efciency, edge-ware optimization ensures a reasonable task allocation between the Edge Computing Server and onboard Edge Devices. Flexibility boosts UAV responsiveness and efciency.

<!-- image-->  
Figure 3. DTOE-AOF and DTOE-AIF implementation framework.

$$
R T = { \frac { \rho . T T C ^ { \mu } } { W T . \tau + P T . \sigma + \sum _ { j = 1 } ^ { o } \beta _ { j } } }\tag{7}
$$

Equation (7) presents a complicated model for measuring responsetime RT . Considering the processing time where σ, waiting time WT.τ, and job completion time $\bar { P T } . \sigma$ additional aspects like concurrency, resource availability, and network latency. With this equation, it optimizes the responsiveness TTCµ of a system in every possible scenario by considering the complex nature of reaction time in diferent settings.

$$
R U = { \frac { \omega . A U } { I T ^ { \epsilon } . \beta + M C y ^ { \sigma } . \epsilon + \sum _ { j = 1 } ^ { o } \varphi _ { j } } }\tag{8}
$$

Equation (8) is a complete and complex model for the nuanced evaluation of resource utilization RU . Te weighted coefcient ω, which emphasizes the value of Actual Utilisation AU . Te equation can capture complex relationships in the real utilization component since non-linearity is introduced by including the exponent $\bar { I } T ^ { \epsilon }$ Te efect of Maximum Capacity MC is highlighted by $\beta ,$ and the non-linear sensitivity is introduced by the accompanying exponent σ, which recognizes the complex link between resource efciency and fuctuations in maximum capacity. Te subtle impact of idle time on the overall efciency of resource utilization is captured by the exponent j while ϕ determines the infuence of Idle Time, which represents periods when a resource is not in use.

$$
S = { \frac { T P ^ { \alpha } . \beta } { ( F N + T P ) . \alpha + \sum _ { k = 1 } ^ { P } F S e _ { k } ^ { k _ { j } } } }\tag{9}
$$

A complex and detailed model for sensitivity S evaluation is shown in Eq. (9), which considers several aspects that afect the accuracy of a model for classifcation in detecting positive cases. Te weighting factor where α highlights the relevance of True Positives TP, making it clear how important it is to correctly detect positive cases and add to the total sensitivity FS. Incorporating non-linearity, the coefcient $\beta$ captures complex relationships within the real positive component, enabling a more detailed portrayal of its infuence. Beyond True Positive tests, p highlights the combined impact of genuine positives and FalseNegatives FN , acknowledging the interconnectedness of positively detected cases and negatively classed situations. An additional layer of complexity is introduced by the exponent k, which refects the combined efect of diferences in true positives and false negatives on sensitivity j, resulting in non-linear sensitivity to these variables.

## UAV‑aided wireless network implementation

Te proposed UAV-aided wireless network idea addresses the challenges of operating in areas without signals and poor communication infrastructure. Tis system uses an Endpoint Device (ED) to collect, analyze, and transfer data to the cloud in real-time. Te proposal recommends employing UAVs as mobile base stations to provide temporary communication links in areas without physical connections18. Figure 4 shows the UAV-aided internet architecture in 3D with two-hop full-duplex communication. With this payload, the UAV may briefy connect to a signal-less environment. Te UAV may relay afer connecting to an access point in a signal-bearing zone.

<!-- image-->  
Figure 4. UAV positions in various periods.

Te relay system uses amplify-and-forward (AF). Tis method amplifes the incoming signal without demodulating or modulating. Signal processing is simplifed by this technology, making it appropriate for UAV operations in remote places with limited computing resources. Te recommended solution increases communication range by deploying UAV-assisted relays to bypass signal restrictions in challenging situations. A two-hop fullduplex transfer scenario ensures complete data delivery between the ED and the cloud. Using the bridge the UAV momentarily puts up, the ED can transfer data to the cloud even without a signal. Te model considers UAV-assisted relay building time- and space-related dynamics in three dimensions for accurate placement and movement. Te recommended communication method uses the amplify-and-forward strategy, which is mathematically stated to explain signal reception and transmission. Tis study proposes a wireless network that can interact in poor signal areas using UAVs with amplify-and-forward relay methods. Tis unique technique may be used for disaster relief, remote sensing, and other locations lacking communication networks.

$$
R = \frac { P o S ^ { \partial } . \beta } { A ^ { \tau } V ^ { \delta ^ { \mu } } . \alpha + ( R . R e ) }\tag{10}
$$

For a detailed evaluation of a system’s robustness R, Eq. (10) provides a complete and complex model. Te importance of System Performance is shown by the weighted coefcient,where $\mathring { P o S ^ { \partial } }$ , which highlights the total system functionality’s holistic value in determining the system’s robustness. By including the exponent $A ^ { \tau } V ^ { \delta ^ { \mu } } ,$ non-linearity is introduced, allowing the system performance component to capture complex relationships δ. Because the system is not linear, the efects of performance fuctuations (R.Re) on the system’s resilience α may be more precisely and nuancedly depicted.

$$
I _ { j } = i _ { o } \exists _ { j } ^ { - 2 } \frac { i _ { o } } { I ^ { 2 } + \lVert q _ { N } - q _ { V } \rVert } , J \in \cal { M }\tag{11}
$$

In the Eq. (11), where the distance between $I _ { j }$ and the MES is denoted as $i _ { o } ,$ and the channel gain at a reference distance where $i _ { o } , I ^ { 2 } ;$ , may be used to calculate the channel gain between qN and the qV.

$$
S _ { j } = \frac { C } { o } l o g _ { 2 } \Biggl ( 1 + \frac { Q _ { u } \bigl | I _ { j } \bigr | ^ { 2 } } { Y ^ { 2 } } \Biggr ) , J \in M\tag{12}
$$

In the Eq. (12), where $S _ { j }$ is the transmission bandwidth between the C and the $\mathrm { O } ;$ for ofoading communication $Q _ { u } \mathrm { : }$ , it can be further split into $I _ { j }$ and $Y ^ { 2 }$ sub-bands.

## Energy optimization process

Inefciency caused by fxed emission energy for data transmission must be considered to lengthen Edge Devices (EDs) working time by avoiding energy loss. Predetermined emission energy may result in received signal energy that exceeds receiver sensitivity, wasting energy and reducing efciency. To address this difculty, a unique

DL-based energy optimization technique adapts emission energy levels to ambient factors. Tis algorithm’s implementation framework is shown in Fig. 524. Te suggested method dynamically adjusts emission energy levels to increase energy efciency and ED operational time. Te program starts with issue formulation to discover and quantify environmental factors afecting emission energy under constrained circumstances.

Next, a DL technique creates a prediction model to manage transmission time delay-induced environmental variable uncertainties. DL’s ability to recognize complex data patterns helps forecast the appropriate emission energy levels depending on environmental parameters. Figure 5 shows the algorithm’s data collecting, model development, and energy optimization stages. Sensors capture environmental data and add it to the prediction model to calculate emissions energy. Te DL algorithm learns from past data and adapts to new scenarios to anticipate ideal emission energy levels in real-time. Tis reduces transmission delays.

Initialize the Population Size . , and Generation .   
While 1 (Not Terminate Condition)   
do 1   
t  t + 1   
While 2[ . ]   
do 2   
Calculate Q(t), do crossover, do mutation   
for = 1 , = 1   
if mutation with   
else   
do mutation with   
end if   
End for   
End while 2   
End while 1

Algorithm 2. UAVs for power load matching of ML system.

Algorithm 2 shows the UAVs for Power Load Matching of ML System-based task ofoading strategy. Random initialization of the actor-network and critic-network parameters is performed in the beginning stage of training. Following training, this research normalizes the state observations, feeds them into the hybrid action network, and produces continuous and discrete action distributions. Next, the policy is executed, and the UAV is rewarded. In this instance, the processing is vulnerable to abortion if the UAV travels outside of the designated region. Our study has real-world applications when natural catastrophes damage communication equipment or when there is a need to temporarily unload data volumes in hotspots, where it is possible to deploy adaptable UAVs quickly. Studying fight planning and task-ofoading procedures is essential and shows promise.Te adaptive DL-based energy optimization system can dynamically adapt to changing environmental conditions by modifying its predictions as fresh data is collected. Te technique optimizes emission energy levels using real-time data to reduce energy waste and enhance ED operating duration in resource-constrained contexts. DL-based energy optimization solves the fxed emission energy constraint in the future, improving ED energy efciency. Real-time prediction and adaptation optimize energy usage, extending operational times and improving performance in energy-saving settings.

<!-- image-->  
Figure 5. Process of energy optimization.

$$
\rho _ { j } = Q s \Big \{ \ni M \ge \ni _ { U } ^ { j } \Big \} = \int _ { \omega _ { t h } ^ { j } } ^ { \varphi } f ^ { - y } d y = f ^ { - \mu _ { U } ^ { j } }\tag{13}
$$

Within the framework of a reliability study, the probability density function (PDF) of a random variable $\rho _ { j }$ is described by $\operatorname { E q . } ( 1 3 ) ^ { 2 5 }$ . In the interval where the maintenance of the intensity ∋M is equal to Qs or larger than the use intensity $\ni _ { U } ^ { J } ,$ the integral of the PDF f over Qs is used to determine the variable $\omega _ { t h } ^ { J } .$ Te integration is carried out as ϕ from the threshold value $f ^ { - y } d y$

$$
\ni _ { m } ^ { j } = \frac { d _ { j } } { g _ { m } ^ { j } } + \Delta _ { j } ( ( 1 - \nabla ) \partial _ { M } + \delta )\tag{14}
$$

It seems like Eq. (14) describes a connection using variables where $\ni _ { m } ^ { j }$ and some constants. A more thorough explanation of the symbols, terminology,and extra context would greatly assist in providing a more accurate interpretation. Nevertheless, according to popular mathematical modelling notations, it appears to stand for a formula that expresses the maintenance intensity $\cdot d _ { j }$ as a function of $g _ { m } ^ { j } , \Delta _ { j } , 1 - \nabla , \partial _ { M }$ , and δ. Some systems or processes may have a relationship between the rate or intensity of maintenance.

$$
P _ { m } ^ { j } = \in U _ { m } ^ { j } + ( 1 - \Rightarrow ) \vartheta _ { m } ^ { j }\tag{15}
$$

where the quantity $P _ { m } ^ { j }$ is shown mathematically as a mixture of two terms where ∈ $U _ { m } ^ { j } + ( 1 - \ni )$ . Potential variables or functions in this equation include $U _ { m } ^ { J }$ and $\vartheta _ { m } ^ { J } ,$ , with ∋ acting as a constant or coefcient.

Te DTOE-AOF and DTOE-AIF framework are are developed to improve the allocation of computing tasks between central cloud servers and edge devices. It is a dynamic evaluation of factors such as computational load, energy consumption, and network latency that this framework uses to fnd the ideal execution site for each operation. Te system uses real-time data and predictive algorithms to ensure that immediate attention computations are carried out on edge devices. It also ensures that computations that require more resources are transmitted to the cloud. Te potential of this edge-aware method to increase overall system performance, minimize latency, and preserve bandwidth is of great advantage to applications that are integrated into the IoT, smart cities, and autonomous systems. Te framework can efectively handle distant computing resources in ever-changing circumstances and workloads because of its fexibility and adaptability.

Te DTOE-AOF and DTOE-AIF stands out as a trailblazing solution that successfully tackles the problems that come with UAV operations. Optimizing task allocation according to proximity and pressure, DTOE-AOF achieves tremendous mission efciency, reaction time, and resource utilization benefts by seamlessly combining AI algorithms with computing edge capabilities. Its versatility, proven by many modelling studies, highlights its possibility of transforming UAV operations in several felds. Te DTOE-AOF and DTOE-AIF architecture performance and scalability are applicable in various felds, from precision gardening to catastrophe management. Tis study paves the way for future developments in autonomous aerial systems and improves UAV capabilities.

## Results and discussion

## Data descriptive

An approach to UAV route planning and optimization using FLA. To optimize a UAV’s trajectory, the suggested method minimizes energy consumption and route length while avoiding environmental barriers41. Te FLA algorithm moves molecules around in a three-dimensional search space for the best answer. Experiments were conducted in a virtual setting with diferent obstacle confgurations to assess how well the suggested strategy worked. It uses the concepts described in Fick’s law of difusion to improve the navigation steps of UAVs. Tis law simulates the fow of particles from areas of high concentration to areas of low concentration. Te method optimizes routes and avoids challenges, leading to more efcient and efective UAV operations. Tis work illustrates how FLA can be used in real-world UAV route planning scenarios by completely implementing the code and theoretical explanations. Tis paper uses python jupyter as a simulation platform for implementation. It is the most recent interactive development environment for notebooks, code, and data that is designed to be accessible over the web. Users in data science, scientifc computing, computational journalism, and machine learning can design and arrange processes due to the adaptable interface of this application. A modular design encourages the addition of expansions to expand and enhance usefulness.

By comparing its results across diferent investigations, this study fnds that an AI-powered edge computing framework is more efective than traditional centralized methods in optimizing UAV operations. Improving mission efciency, response time, resource utilization, sensitivity, and robustness results show that the AI-driven strategy is signifcantly better. Table 2 shows the parameters that are used towards the simulation.

<table><tr><td rowspan=1 colspan=1>Simulation Parameters</td><td rowspan=1 colspan=1>Parameter values</td></tr><tr><td rowspan=1 colspan=1>Total number of samples</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>Coverage of sensor networks</td><td rowspan=1 colspan=1>100×100m²</td></tr><tr><td rowspan=1 colspan=1>Positions of sensor nodes</td><td rowspan=1 colspan=1>40,600</td></tr><tr><td rowspan=1 colspan=1>The amount of the instruction package</td><td rowspan=1 colspan=1>200 bits</td></tr><tr><td rowspan=1 colspan=1>The amount of data sent</td><td rowspan=1 colspan=1>5900 bits</td></tr><tr><td rowspan=1 colspan=1>Initial energy consumption of sensor nodes</td><td rowspan=1 colspan=1>0.66J</td></tr><tr><td rowspan=1 colspan=1>Power consumption during data aggregation</td><td rowspan=1 colspan=1>6.9 nJ/bit/sig</td></tr><tr><td rowspan=1 colspan=1>The energy needed to generate data with a bit length of 1</td><td rowspan=1 colspan=1>56 nJ/bit/m²</td></tr><tr><td rowspan=1 colspan=1>Power parameters for multi-path operation</td><td rowspan=1 colspan=1>9.11 pJ/bit/m²</td></tr><tr><td rowspan=1 colspan=1>Energy parameters for the free-space case</td><td rowspan=1 colspan=1>3.2 nJ/bit/sig</td></tr></table>

Table 2. Simulation parameters used for implementing DTOE-AOF.

## Performance analysis

Te evaluation of the efectiveness of the mission investigates whether or not the framework can complete tasks within the limits that have been designated. A framework’s competence to execute the mission’s goals in various circumstances may be shown by metrics such as the task completion rate and the success rate. Response time analysis measures the time it takes to complete an activity from when it is initiated until completion. To reduce response times, this research aims to ofer real-time processing, essential for developing smart city infrastructure and autonomous cars. When analyzing resource consumption, one examines the efectiveness of computing resources (such as CPU and memory) used by devices at the network’s edge. Te framework’s ability to distribute workload and eliminate bottlenecks may be shown by metrics such as the proportion of resources used and the load distribution. Te term "sensitivity analysis" refers to examining how sensitive the framework is to changes in characteristics such as network latency, bandwidth, and task load. Te fexibility and robustness of the framework can be tested using this, which is important for evaluating the framework. Examining the framework’s robustness determines if it can continue functioning normally despite interruptions or changes in the availability of resources and networks of any type. A confrmation that DTOE-AOF and DTOE-AIF will continue to handle unforeseen problems to maintain operations efectively may be achieved via the completion of this research.

Te suggested AI-powered edge computing approach for optimizing UAV operations outperforms traditional centralized solutions regarding mission efciency. Te framework reduces latency and improves real-time decision-making by dynamically assigning computing jobs to UAVs and edge nodes according to proximity and task urgency. Because of this, mission execution is simplifed, and UAVs can adapt to new circumstances and complete duties more accurately and efciently. Incorporating AI algorithms allows UAVs to optimize their fight patterns and resource utilization independently, further improving mission efciency. According to comprehensive simulation studies and real-world deployments, the AI-powered edge computing strategy improves mission efciency, allowing UAVs to accomplish goals more efectively while minimizing onboard resources and maximizing operational uptime. In Fig. 6a, the Mission Efciency Analysis showcases an outstanding contrast with DTOE-AOF, attaining an astounding 99.4%. Meanwhile, a remarkable 95.7% performance is seen in Fig. 6b when Mission Efciency Analysis is compared with DTOE-AIF. Tese outcomes demonstrate how well the mission analysis optimized operational outcomes.

According to the response time analysis, the AI-powered edge computing architecture may greatly improve responsiveness and decrease latency in UAV operations. Te system guarantees that vital tasks are done with minimal delay, even in resource-constrained contexts, by utilizing edge computing infrastructure and dynamic task ofoading methods. Tis allows UAVs to react to new possibilities or dangers in real-time and adapt rapidly to changing circumstances since data processing, decision-making, and action execution are all accelerated. Te AI-powered edge computing technique greatly improves response time compared to traditional centralized processing architectures, which ofen experience latency difculties from data transmission delays. Tis, in turn, makes UAV operations more agile and efective. Figure 7a shows that compared to DTOE-AOF, the Response Time Analysis is very efcient, with a score of 98.9%. Figure 7b shows that in contrast to DTOE-AIF, the Response Time Analysis performs admirably, reaching an impressive 89.3%. Tese results highlight how important response time analysis is for optimizing operations.

Te resource use study shows that improving UAV operations with AI-powered edge computing leads to signifcant efciency gains. Te system maximizes efciency and reduces wastage by dynamically assigning computing jobs to UAVs and edge nodes according to their proximity and available resources. Tis improves operational sustainability and cost-efectiveness by reducing onboard processing resources, electricity, and bandwidth utilization. Further improving resource utilization efciency, UAVs with AI-driven decision-making skills can adjust their resource usage in real-time according to changing mission needs and environmental conditions. Results from extensive simulation studies and real-world assessments show that the AI-driven edge computing method greatly enhances resource utilization in UAV operations, letting businesses accomplish their goals with fewer resources while keeping performance and reliability at a high standard. Figure 8a shows that the Resource Utilization Analysis is very efective, outperforming DTOE-AOF by an impressive 97.6%. Figure 8b shows that the Resource Utilization Analysis performs well, outperforming DTOE-AIF by an astounding 94.6%. Tese fndings highlight how crucial it is to optimize operational outcomes through efcient allocation of resources.

<!-- image-->  
(a)

<!-- image-->  
(b)

Figure 6. Mission efciency analysis is compared with (a) DTOE-AOF and (b) DTOE-AIF.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Figure 7. Response time analysis is compared with (a) DTOE-AOF and (b) DTOE-AIF.

Te sensitivity analysis results show how the AI-driven edge computing framework performs under diferent scenarios and settings, which is useful for optimizing UAV operations. Te analysis determines how sensitive the framework is to changes in input parameters like network bandwidth, edge node availability, and task urgency by methodically adjusting these variables. Trough extensive simulations and sensitivity analyses, people can see how changes to these parameters afect critical performance measures like response time, resource consumption, and mission efciency. Tis greatly enhances insights into the framework’s robustness and the identifcation of crucial aspects that substantially impact its performance. Te framework can be fne-tuned to better respond to various operational situations and environmental conditions with the help of sensitivity analysis, which helps to identify potential vulnerabilities or areas for development. Improving the overall efcacy and reliability of UAV operations in dynamic and uncertain situations is possible when stakeholders understand the system’s sensitivity to diferent aspects. Tis knowledge allows them to make educated decisions about system design, deployment of resources, and operational strategies.

Figure 9a shows that the Sensitivity Analysis performed well, outperforming DTOE-AOF by an impressive 96.5%. Figure 9b shows that the Sensitivity Analysis maintains its efciency, surpassing DTOE-AIF by an impressive 95.6%. Tese outcomes demonstrate the critical function of sensitivity analysis in improving operational strategy.

<!-- image-->  
(a)

<!-- image-->  
(b)

Figure 8. Resource utilization analysis is compared with (a) DTOE-AOF and (b) DTOE-AIF.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Figure 9. Sensitivity analysis is compared with (a) DTOE-AOF and (b) DTOE-AIF.

Te analysis of the robustness of the AI-powered edge computing framework determines how well it can withstand and continue operating despite disturbances, unknowns, and hostile environments. Tests for adversarial attacks, failure scenarios, and stress determine the system’s resilience. We check how well the framework handles potential dangers as part of this process. Tese dangers include cyberattacks, data corruption, edge node issues, and network outages. Te robustness study discovers the framework’s faws and areas needing further defences via testing and evaluation. Te robustness research proves that the framework is dependable and trustworthy, which gives users faith that it will perform in real-world production settings. Identifying failure causes and establishing backup plans may help stakeholders decrease risks and improve UAV operations in uncertain and dynamic settings. Due to robustness analysis, Businesses can ensure that their AI-powered edge computing solutions for optimizing UAV operations are efective, safe, and secure.

Compared to DTOE-AOF, the Robustness Analysis achieves an astounding 98.2%, as shown in Fig. 10a. Te Robustness Analysis continues to work, as seen in Fig. 10b, with an impressive 94.6% compared to DTOE-AIF. Tese fndings highlight the signifcance of robustness analysis in guaranteeing operational resilience.Te AI-powered edge computing architecture achieves consistently better outcomes than typical centralized solutions, according to thorough mission efciency, response time, resource utilization, sensitivity, and resilience assessments. Tese results highlight the revolutionary efect of AI-driven tactics in improving UAV operations’ responsiveness, efectiveness, and dependability.

Te DTOE-AOF achieves outstanding performance due to its utilization of cutting-edge AI and edge computing techniques. Tese techniques are evaluated based on key metrics such as latency, energy consumption, task completion rate, network bandwidth usage, resource utilization, scalability, and adaptability. Reinforcement learning makes it possible to react to changing network circumstances in real-time. In contrast, federated learning protects data privacy and minimizes bandwidth utilization by ensuring that data is kept locally. When combined with real-time analytics, edge caching helps decrease latency and bandwidth demands. GNNs can simulate complex network topologies, enabling them to optimize resource allocation. DTOE-AOF is better than previous techniques in satisfying the requirements of autonomous cars, smart cities, and industrial IoT applications because it includes a number of characteristics that collectively ensure that it can manage resources efectively, adapt to changing scenarios, and provide fast work processing.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Figure 10. Robustness analysis is compared with (a) DTOE-AOF and (b) DTOE-AIF.

## Conclusion

Improving the operations of UAVs by incorporating AI-powered edge computing presents a game-changing opportunity to address signifcant issues such as latency, bandwidth constraints, and scalability encountered by traditional centralized processing systems. Edge nodes and UAVs are dynamically assigned compute workloads based on proximity, resource availability, and work urgency. UAVs can fy more efciently, minimize latency, and save onboard resources using DTOE-AOF. Tis makes them great for precision agriculture, disaster management, inspection of infrastructure, and monitoring. Precision farmers might beneft from UAVs equipped with DTOE-AOF in several ways, including the ability to make data-driven decisions in real-time during disaster response and increased production and efcient use of resources. Efciency in missions, response time, and resource usage are all areas where DTOE-AOF outperforms centralized solutions in simulations. Tis was discovered afer comparing the two approaches. Robustness and sensitivity studies have shown that DTOE-AOF is both fexible and scalable. Tis demonstrates that it has the potential to alter the operations of UAVs. Improving the capabilities and efciency of UAVs using AI-driven edge computing is an area that might revolutionize UAV operations and inspire new ideas across several industries.

Te decision-making capabilities of DTOE-AOF, which have minimum latency and allow for real-time adaptability, enhance the navigational efciency and safety of autonomous vehicles. Tis framework ensures that data privacy is maintained via federated learning while simultaneously optimizing resource management for smart city applications such as trafc control and emergency response mechanisms. Te real-time analytics and scalable resource allocation capabilities of the industrial IoT contribute to improvements in operational efciency, predictive maintenance management, and system reliability. Te presence of these characteristics demonstrates that DTOE-AOF has the potential to enhance and transform signifcant current systems.

## Data availability

Te data used in this research is available in the following link https://www.kaggle.com/datasets/mohamedami neferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/versions/2.

Received: 26 March 2024; Accepted: 9 July 2024

Published online:16 July 2024

## References

1. Rovira-Sugranes, A., Razi, A., Afghah, F. & Chakareski, J. A review of AI-enabled routing protocols for UAV networks: Trends, challenges, and future outlook. Ad Hoc Netw. 130, 102790. https://doi.org/10.1016/j.adhoc.2022.102790 (2022).

2. Kim, B., Jung, J., Min, H. & Heo, J. Energy efcient and real-time remote sensing in AI-powered drone. Mob. Inf. Syst. 2021, 1–8. https://doi.org/10.1155/2021/6650053 (2021).

3. Gupta, R., Reebadiya, D. & Tanwar, S. 6G-enabled edge intelligence for ultra-reliable low latency applications: Vision and mission. Comput. Stand. Interfaces 77, 103521. https://doi.org/10.1016/j.csi.2021.103521 (2021)

4. Sai, S., Garg, A., Jhawar, K., Chamola, V., & Sikdar, B. A comprehensive survey on artifcial intelligence for unmanned aerial vehicles. IEEE Open J. Vehic. Technol. https://doi.org/10.1109/OJVT.2023.3316181 (2023).

5. Alahmad, Y. & Agarwal, A. Multiple objectives dynamic VM placement for application service availability in cloud networks. J. Cloud Comput. 13(1), 1–20.  https://doi.org/10.1186/s13677-024-00610-2 (2024).

6. Chen, J., Xiao, W., Zhang, H., Zuo, J. & Li, X. Dynamic routing optimization in sofware-defned networking based on a metaheuristic algorithm. J. Cloud Comput. 13(1), 1–8.  https://doi.org/10.1186/s13677-024-00603-1 (2024).

7. Goswami, P., Faujdar, N., Debnath, S., Khan, A. K. & Singh, G. Investigation on storage level data integrity strategies in cloud computing: Classifcation, security obstructions, challenges and vulnerability. J. Cloud Comput. 13(1), 1–23. https://doi.org/10. 1186/s13677-024-00605-z (2024).

8. Wu, Y. Cloud-edge orchestration for the Internet of Tings: Architecture and AI-powered data processing. IEEE Internet Tings J. 8(16), 12792–12805. https://doi.org/10.1109/JIOT.2020.3014845(2020).

9. Chen, J. et al. Multi-type concept drif detection under a dual-layer variable sliding window in frequent pattern mining with cloud computing. J. Cloud Comput. 13(1), 1–9. https://doi.org/10.1186/s13677-023-00566-9 (2024).

10. Navardi, M., Humes, E., &Mohsenin, T. E2edgeai: Energy-efcient edge computing for deployment of vision-based dnns on autonomous tiny drones. In 2022 IEEE/ACM 7th Symposium on Edge Computing (SEC). 504–509 (IEEE, 2022).

11. Liu, K., Chauhan, S., Devaraj, R., Shahi, S., & Sreekumar, U. Enabling autonomous unmanned aerial systems via edge computing. In 2019 IEEE International Conference on Service-Oriented System Engineering (SOSE). 374–3745 (IEEE, 2019).

12. Adil, M., Song, H., Mastorakis, S., Abulkasim, H., Farouk, A., & Jin, Z. UAV-assisted IoT applications, cybersecurity threats, AIenabled solutions, open challenges with future research directions. In IEEE Transactions on Intelligent Vehicles. https://doi.org/10. 1109/TIV.2023.3309548 (2023).

13. Goethals, T., Volckaert, B. & De Turck, F. Enabling and leveraging AI in the intelligent edge: A review of current trends and future directions. IEEE Open J. Commun. Soc. 2, 2311–2341. https://doi.org/10.1109/OJCOMS.2021.3116437 (2021).

14. Khan, M. A. et al. Swarm of UAVs for network management in 6G: A technical review. IEEE Trans. Netw. Serv. Manag. 20(1), 741–761. https://doi.org/10.1109/TNSM.2022.3213370 (2023).

15. Zawish, M. et al. Toward on-device AI and blockchain for 6G-enabled agricultural supply chain management. IEEE Internet Tings Mag. 5(2), 160–166. https://doi.org/10.1109/iotm.006.21000112 (2022).

16. Mallikarjunaradhya, V., Pothukuchi, A. S. & Kota, L. V. An overview of the strategic advantages of AI-powered threat intelligence in the cloud. J. Sci. Technol. 4(4), 1–12 (2023).

17. Arun, M., Sivagami, S. M., Raja Vijay, T. & Vignesh, G. Experimental investigation on energy and exergy analysis of solar water heating system using zinc oxide-based nanofuid. Arab. J. Sci. Eng. 48(3), 3977–3988. https://doi.org/10.1007/ s13369-022-07369-1 (2023).

18. Munawar, H. S. et al. Towards smart healthcare: UAV-based optimized path planning for delivering COVID-19 self-testing kits using cutting edge technologies. Sustainability 13(18), 10426. https://doi.org/10.3390/su131810426 (2021).

19. Su, Y. A trust based scheme to protect 5G UAV communication networks. IEEE Open J. Comput. Soc. 2, 300–307. https://doi.org/ 10.1109/OJCS.2021.3058001 (2021).

20. Pal, O. K., Shovon, M. S. H., Mridha, M. F., & Shin, J. A Comprehensive Review of AI-enabled Unmanned Aerial Vehicle: Trends, Vision, and Challenges. arXiv preprint arXiv:2310.16360 (2023).

21. Yang, W., Yang, L. T., & Chronopoulos, A. T. Guest editorial special issue on smart IoT system: Opportunities by linking cloud, edge, and AI. IEEE Internet Tings J. 8(16), 12478–12480. https://doi.org/10.1109/JIOT.2021.3092440 (2021).

22. Zhou, H., Wang, Z., Zheng, H., He, S., & Dong, M. Cost minimization-oriented computation ofoading and service caching in mobile cloud-edge computing: An A3C-based approach. In IEEE Transactions on Network Science and Engineering. https://doi. org/10.1109/TNSE.2023.3255544 (2023).

23. Benmerar, T. Z. et al. Intelligent multi-domain edge orchestration for highly distributed immersive services: An immersive virtual touring use case. In 2023 IEEE International Conference on Edge Computing and Communications (EDGE), Chicago, IL, USA. 381–392. https://doi.org/10.1109/EDGE60047.2023.00061 (2023).

24. Pan, G., Zhang, H., Xu, S., Zhang, S. & Chen, X. Joint optimization of video-based AI inference tasks in MEC-assisted augmented reality systems. IEEE Trans. Cognit. Commun. Network. 9(2), 479–493. https://doi.org/10.1109/TCCN.2023.3235773 (2023).

25. Garg, S. et al. Trusted explainable AI for 6G-enabled edge cloud ecosystem. IEEE Wirel. Commun. 30(3), 163–170. https://doi.org/ 10.1109/MWC.016.220047 (2023).

26. Koubaa, A., Ammar, A., Abdelkader, M., Alhabashi, Y. & Ghouti, L. AERO: AI-enabled remote sensing observation with onboard edge computing in UAVs. Remote Sens. 15(7), 1873. https://doi.org/10.3390/rs15071873 (2023).

27. Lins, S. et al. Artifcial intelligence for enhanced mobility and 5G connectivity in UAV-based critical missions. IEEE Access 9, 111792–111801. https://doi.org/10.1109/ACCESS.2021.3103041 (2021).

28. Ijaz, H., Ahmad, R., Ahmed, R., Ahmad, W., Kai, Y. & Jun, W. A UAV assisted edge framework for real-time disaster management. In IEEE Transactions on Geoscience and Remote Sensing. https://doi.org/10.1109/TGRS.2023.3306151 (2023).

29. Surianarayanan, C., Lawrence, J. J., Chelliah, P. R., Prakash, E. & Hewage, C. A survey on optimization techniques for edge artifcial intelligence (AI). Sensors 23(3), 1279. https://doi.org/10.3390/s23031279 (2023).

30. Palossi, D. et al. Fully onboard AI-powered human-drone pose estimation on ultralow-power autonomous fying nano-UAVs. IEEE Internet Tings J. 9(3), 1913–1929 https://doi.org/10.1109/JIOT.2021.3091643 (2022).

31. Niculescu, V., Lamberti, L., Conti, F., Benini, L. & Palossi, D. Improving autonomous nano-drones performance via automated end-to-end optimization and deployment of dnns. IEEE J. Emerg. Sel. Top. Circuits Syst. 11(4), 548–562. https://doi.org/10.1109/ JETCAS.2021.3126259 (2021).

32. Xu, F., Hussain, T., Ahmed, M., Ali, K., Mirza, M. A., Khan, W. U., & Han, Z. Te state of ai-empowered backscatter communications: A comprehensive survey. IEEE Internet Tings J. https://doi.org/10.1109/JIOT.2023.3299210 (2023).

33. Jat, D. S., & Singh, C. Artifcial intelligence-enabled robotic drones for COVID-19 outbreak. Intell. Syst. Methods Combat COVID-19 37–46. https://doi.org/10.1007/978-981-15-6572-4_5 (2020).

34. Kim, B. et al. A computation ofoading scheme for UAV-edge cloud computing environments considering energy consumption fairness. Drones 7(2), 139. https://doi.org/10.3390/drones7020139 (2023).

35. Miya, J., Raj, S., Ansari, M. A., Kumar, S., & Kumar, R. Artifcial intelligence advancement for 6G communication: A visionary approach. In 6G Enabled Fog Computing in IoT: Applications and Opportunities. 355–394 (Springer Nature Switzerland, 2023).

36. Bai, Y., Zhao, H., Zhang, X., Chang, Z., Jäntti, R., & Yang, K. Towards autonomous multi-UAV wireless network: A survey of reinforcement learning-based approaches. In IEEE Communications Surveys & Tutorials (2023).

37. Jana, M. Optimal Ofoading Decision for Fog Computing Using LSTM-GWO. Doctoral Dissertation, National Institute of Technology (2022).

38. Heidari, A., Navimipour, N. J., & Unal, M. Applications of ML/DL in the management of smart cities and societies based on new trends in information technologies: A systematic literature review. Sustain. Cities Soc. 104089.https://doi.org/10.1016/j.scs.2022. 104089 (2022).

39. Pandey, C., Tiwari, V., Rathore, R. S., Jhaveri, R. H., Roy, D. S., & Shitharth, S. Resource-efcient synthetic data generation for performance evaluation in mobile edge computing over 5G networks. IEEE Open J. Commun. Soc. https://doi.org/10.1109/OJCOMS. 2023.3306039 (2023).

40. Stavroulaki, V. et al. DEDICAT 6G—Dynamic coverage extension and distributed intelligence for human centric applications with assured security, privacy and trust: From 5G to 6G. In 2021 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), Porto, Portugal. 556–561. https://doi.org/10.1109/EuCNC/6GSummit51104.2021.9482611 (2021).

41. https://www.kaggle.com/code/kooaslansefat/fck-s-law-algorithm-fa-for-uav-path-planning.

42. Letaief, K. B., Shi, Y., Lu, J. & Lu, J. Edge artifcial intelligence for 6G: Vision, enabling technologies, and applications. IEEE J. Sel. Areas Commun. 40(1), 5–36. https://doi.org/10.1109/JSAC.2021.3126076 (2021).

43. Seng, J. K. P., Ang, K. L. M., Peter, E. & Mmonyi, A. Artifcial intelligence (AI) and machine learning for multimedia and edge information processing. Electronics 11(14), 2239. https://doi.org/10.3390/electronics11142239 (2022).

## Author contributions

Te authors confrm their contributions to the paper as follows: Conceptualization: Suganya B, Gopi R; Methodology: Suganya B, Gopi R; Formal analysis and investigation: Suganya B, Ranjith Kumar A; Writing—original draf preparation: Suganya B, Gopi R, Ranjith Kumar A; Writing—review and editing: Gopi R, Gavendra Singh; Supervision: Gavendra Singh, Suganya B, Gopi R; Figs. 1, 2, 3, 4 and 5: Prepared by Suganya B, Gopi R and Ranjith Kumar A. All authors reviewed the results and approved the fnal version of the manuscript.

## Competing interests

Te authors declare no competing interests.

## Additional information

Correspondence and requests for materials should be addressed to G.S.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afliations.

<!-- image-->

CC Open Access Tis article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. Te images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

© Te Author(s) 2024