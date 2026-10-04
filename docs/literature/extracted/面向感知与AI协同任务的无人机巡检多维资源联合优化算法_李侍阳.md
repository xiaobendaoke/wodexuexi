# 《电子与信息学报》网络首发论文

题目： 面向感知与AI 协同任务的无人机巡检多维资源联合优化算法

作者： 李侍阳，朱晓荣

收稿日期： 2025-12-03

网络首发日期： 2026-03-31

引用格式： 李侍阳，朱晓荣．面向感知与 AI 协同任务的无人机巡检多维资源联合优化算法[J/OL]．电子与信息学报.https://link.cnki.net/urlid/11.4494.TN.20260330.0940.006

<!-- image-->

<!-- image-->

网络首发：在编辑部工作流程中，稿件从录用到出版要经历录用定稿、排版定稿、整期汇编定稿等阶段。录用定稿指内容已经确定，且通过同行评议、主编终审同意刊用的稿件。排版定稿指录用定稿按照期刊特定版式（包括网络呈现版式）排版后的稿件，可暂不确定出版年、卷、期和页码。整期汇编定稿指出版年、卷、期、页码均已确定的印刷或数字出版的整期汇编稿件。录用定稿网络首发稿件内容必须符合《出版管理条例》和《期刊出版管理规定》的有关规定；学术研究成果具有创新性、科学性和先进性，符合编辑部对刊文的录用要求，不存在学术不端行为及其他侵权行为；稿件内容应基本符合国家有关书刊编辑、出版的技术标准，正确使用和统一规范语言文字、符号、数字、外文字母、法定计量单位及地图标注等。为确保录用定稿网络首发的严肃性，录用定稿一经发布，不得修改论文题目、作者、机构名称和学术内容，只可基于编辑规范进行少量文字的修改。

出版确认：纸质期刊编辑部通过与《中国学术期刊（光盘版）》电子杂志社有限公司签约，在《中国学术期刊（网络版）》出版传播平台上创办与纸质期刊内容一致的网络版，以单篇或整期出版形式，在印刷出版之前刊发论文的录用定稿、排版定稿、整期汇编定稿。因为《中国学术期刊（网络版）》是国家新闻出版广电总局批准的网络连续型出版物（ISSN 2096-4188，CN 11-6037/Z），所以签约期刊的网络版上网络首发论文视为正式出版。

# 面向感知与AI协同任务的无人机巡检多维资源联合优化算法

李侍阳 朱晓荣\*

(南京邮电大学通信与信息工程学院   南京   210003)

摘   要：针对在无人机感知与故障检测并发场景下，无人机巡检过程中任务复杂，带宽、算力、功率等多维资源调度困难的问题，该文提出一种面向感知与AI协同规划的智能无人机巡检多维资源联合优化算法。首先提出一种单无人机与多个计算节点之间联合协同完成多个任务的框架，在无人机与多个边缘计算节点协同工作的系统框架下，无人机在巡检点采集图像与传感器数据，并将其分批传输至多个节点进行分布式处理，完成飞行状态感知与故障检测任务，最终形成了以无人机系统能耗最小化为目标，带宽、功率、算力、节点选择、数据量和压缩率为变量的最优化问题。针对该问题，将原优化问题分解为4个子问题，分别采用双辅助混合整数线性规划(MILP)转化、数据驱动边界学习、基于逐次凸逼近(SCA)的带宽功率联合优化和下界解析分配等方法进行求解，并通过交替优化策略实现整体优化，并进行了复杂度分析。最后仿真结果表明，与其它先进算法相比，所提方法在时延、能耗、精度方面具有更优的性能。

关键词：无人机；智能巡检；资源调度；联合优化

中图分类号：TN929.5

DOI: 10.11999/JEIT251284

文献标识码：A

文章编号：1009-5896(2026)12-0001-12

CSTR: 32379.14.JEIT251284

## 1    引言

随着空中活动需求的日益增长，各类飞行器的作业能力正逐步向全空域、多行业方向拓展。无人机的应用范围已覆盖从低空到高空的多个高度层，涵盖微型、中型及大型等多种机型，广泛应用于公共安全、交通运输、应急管理、物流配送、地理测绘等多个领域，持续推动生产与生活模式的创新变革[1]。与传统的人工巡检方式相比，无人机巡检作为一项新兴业务，能够获取人眼难以捕捉的图像信息，不仅显著降低了人力成本，也提升了巡检作业的精度与效率。然而，无人机巡检也对多维资源的分配与任务调度规划提出了新的挑战。以电力系统巡检为例，输电线路长期暴露于户外，易出现腐蚀、老化甚至破损等问题，需依赖定期巡检以保障运行安全[2]。

无人机巡检业务主要由无人机平台、机载传感器、地面控制站三个部分共同完成[3]，巡检过程需要有带宽、算力等多维资源的共同支持，需要制定合适的资源调度策略，目前关于无人机多维资源联合调度的研究已取得大量进展，包含各种任务场景下的资源规划问题。文献[4]提出了一种通感一体化的全双工通信的波束赋性方法，通过优化波束成形矩阵和发射功率来使得整体功耗最小。文献[5]提出了一种协调多点的多基站通感一体化系统框架，多个基站相互协同在完成目标感知的同时，还与用户进行通信，接受用户卸载的计算任务并分配给边缘计算节点，最终形成总能耗最小化的优化问题。文献[6]提出一种通感算联合优化的深度神经网络拆分的多无人机网络，每个无人机需要与用户进行通信并发射感知波形获取感知数据，用深度神经网络完成感知目标分类的计算任务，并按一定策略拆分网络，一部分在无人机本地进行，另一部分卸载到算力强大的计算节点进行。文献[7]提出一种多无人机辅助的集成感知与通信(Integrated Sensing AndCommunications, ISAC)框架，每个无人机需要在特定时隙内发送传感信号，完成对目标的感知，并在其它时隙与地面用户通信，并以发送感知信号的协方差矩阵为目标，使得无人机能耗最小的同时感知性能最优。优化变量分为用户关联和波束赋形，和无人机飞行轨迹两个部分，通过交替优化进行求解。

目前关于无人机资源分配调度的方法可主要分为基于数学优化的方法和基于神经网络的方法。在数学优化的方法中，文献[8]将原先的非凸优化问题转化为3个凸优化子问题，之后在块坐标下降法和逐次凸近似的基础上提出一种高效的迭代算法来求解问题。文献[9]分别使用粒子群优化、差分进化、穷举法分别求解无人机的部署位置、带宽分配的问题、用户-服务器关联决策问题。文献[10]面向多无人机的场景，提出联盟博弈的方法进行联合优化。

文献[11]基于获得的用户内容请求的预测，分别使用改进的K-means聚类算法、拉格朗日对偶方法、Kuhn-Munkres算法分别解决无人机部署、功率分配、载波分配问题。在基于神经网络的方法中，深度强化学习(Deep Reinforcement Learning, DRL)的方式应用较多，文献[12]使用了深度Q网络(DeepQ-Network, DQN)进行轨迹优化和资源分配，文献[13]使用双深度Q网络(Double Deep Q-Network,DDQN)对无人机决策矩阵进行优化。文献[14]则是采用了联邦学习这一分布式计算的方法对时间和能耗进行优化。文献[15]面向多无人机场景，提出一种多智能体强化学习框架，基于P-DQN的QMIX网络，并在奖励中引入了一个类似熵的公平指标，使总回报可分解。

然而在已有无人机巡检的研究中，通常是单独考虑了无人机的各项任务需求，而鲜有对多维任务的综合性研究。无人机的通信、感知、计算这几方面需要消耗时间和能量，需要考虑如何将多维资源进行联合调度，最终实现在一定约束条件下目标的最优化。因此针对上述问题，本文同时考虑通信、感知、计算的多维资源优化问题，提出一种面向无人机智能巡检的多维资源综合规划的框架，以及种4阶段多维资源巡检调度协同优化算法进行求解，本文主要贡献如下:

(1)提出一种多维资源巡检调度协同优化算法与多个计算节点之间联合协同完成多个任务的框架，并进行系统建模，无人机于巡检点拍摄图像，并将图像和传感器数据分批次发送给多个节点进行分布式计算。同时利用新收集的数据通过联邦学习进行本地模型的更新，形成以能耗为目标，带宽、功率、算力分配、节点选择、数据量和压缩率为变量的综合优化问题。

(2)提出一种4阶段多维资源巡检调度协同优化算法，将原始优化问题根据巡检过程拆分为4个子问题，对每个子问题进行数学分析后提出相应的求解方法。针对节点选择问题，采用双辅助混合整数线性规划(Mixed-Integer Linear Programming,MILP)转化法，对于无人机数据采集问题，使用数据驱动边界学习法，对于无人机通信资源分配问题，采用基于逐次凸逼近(Successive Convex Ap-proximation, SCA)的带宽功率联合优化算法，对于节点算力分配问题，采用下界解析分配法。最后用子问题交替优化的方法求解原问题，形成整个算法。

(3)仿真结果表明，本文所提算法与对比算法相比，无人机的总能耗均有所改善。随着带宽、算力等资源的改变，本文算法在能耗方面均优于对比算法，有效降低了总能耗。除此之外，本文对视觉定位和故障检测业务进行了仿真训练，研究了压缩比例和数据量与业务的模型精度之间的关系。

## 2    系统模型

本文考虑如图1所示的无人机巡检任务场景。q ∈ Q =在巡检区域中，存在一定数量的巡检点{1,2, ..., Q} m ∈ M = {1,以及基站与边缘计算节点

<!-- image-->  
图 1 无人机巡检业务场景示意图

$2 , \cdots , M \}$ ，使用单架无人机进行巡检。无人机抵达巡检点后进行图像拍摄，无人机将拍摄的图像Dimg分割为多个图像数据组。图像自身包含的信息除了巡检点的故障情况外，还含有一定的无人机自身方位信息。各基站收到信号后传递给边缘计算节点处理，一方面根据接收的图像信息、无人机传感器数据、信号协方差矩阵的多模态信息推理出无人机的飞行状况，并将飞行信息汇聚到一个节点上，得到无人机飞行姿态的完整信息，并由该节点进行计算，通知无人机调整飞行姿态和路径规划，同时利用收集到的数据训练并更新本地模型。另一方面，各边缘节点将接收的图像数据先进行图像增强，去除拍摄时抖动造成的模糊效果，之后利用本地的图像处理神经网络计算检测故障情况，并对本地模型进行训练和更新，并通知无人机前往下一个巡检点。

## 2.1 数据分配模型

无人机在巡检点从多个方位拍摄多张图像，设$D ^ { \mathrm { i m g } }$ 是无人机拍摄的数据量。将初始压缩比设置为λ，它表示压缩后的数据量与原数据量的比值。ζ表示压缩比例，它表示进行压缩处理的数据比例，例如 $\zeta = 0 . 4$ 表示有40%的数据会被压缩。考虑前两者则实际压缩的数据比例为[16]

$$
\hat { \lambda } = \zeta \cdot \lambda + ( 1 - \zeta )\tag{1}
$$

考虑到单一种类的数据存在一定的局限性，例如图像数据只反应了无人机当前的瞬时位置，忽略了无人机自身的运动状态，且受到图像质量的影响，我们考虑采用多种类的数据联合进行无人机位置感知，传输的多模态数据包括拍摄的图像，无人机的传感器数据，以及接收信号的协方差矩阵，协方差矩阵是根据具体的信号波形计算出的，不占用额外数据量， $D ^ { \mathrm { s e n s e } }$ 是传感器数据量，总数据量表示为

$$
D ^ { \mathrm { t o t a l } } = \hat { \lambda } D ^ { \mathrm { i m g } } + D ^ { \mathrm { s e n s e } }\tag{2}
$$

数据卸载分配：考虑将所有数据 $D ^ { \mathrm { t o t a l } }$ 划 分M 为 组，并通过无人机传输到 $m$ 个附近的边缘计算节点，每个节点先对子图像进行图像增强处理，去除拍摄时因抖动造成的模糊，再使用图像处理神经网络处理一个增强的子图像，检测对应区域的故障数量，最后将检测结果汇总，通知无人Dtotal 机前往下一个巡检点。将总数据 划分表示为${ \cal D } ^ { \mathrm { t o t a l } } = \{ D _ { 1 } , D _ { 2 } , \cdots , D _ { M } \}$ ， $D _ { m }$ 表示分配给边缘节m点 的一组数据大小，数据量的分配以各节点算力资源总量为权重进行分配

$$
D _ { m } = D ^ { \mathrm { t o t a l } } \frac { c _ { m } } { \displaystyle \sum _ { m = 1 } ^ { M } c _ { m } \beta _ { u , m } }\tag{3}
$$

其中， $\beta _ { u , m }$ 是二进制变量，表示无人机的节点选择策略表示， $\beta _ { u , m } = 1$ m表示无人机与节点 建立连接。 $c _ { m }$ m表示节点 的可用算力

## 2.2 通信模型

在本文场景中，无人机与基站之间采用正交频分复用(Orthogonal Frequency Division Multiple-xing, OFDM)的方式进行通信，带宽以资源块的形式进行分配，无人机与多个基站通信，即将数据卸载到多个边缘节点进行计算。基站接收到无人机的数据信息后，除了后续的数据处理，还会从接收波G形中计算出协方差矩阵 用于感知业务。无人机向边缘节点发射的信号可以表示为通信信号和环境噪声的叠加，无人机发射的波形具体可以表示为

$$
X \left[ t \right] = \sum _ { m = 1 } ^ { M } w _ { m } \mathbf { x } _ { m } [ t ]\tag{4}
$$

其中， ${ \pmb x } _ { m }$ [t]表示通信数据波形， ${ \pmb w } _ { m }$ 表示通信波束赋形矢量，设 $h _ { u , m }$ m是节点 与无人机之间的无线信道增益， $n _ { 0 }$ 表示高斯白噪声， $n _ { 0 } = o _ { u , m } B _ { 0 } N _ { 0 }$ ，$N _ { 0 }$ m为噪声功率谱密度。节点 接收的波形表示为

$$
\pmb { y } [ t ] = \sum _ { m = 1 } ^ { M } h _ { u , m } { \pmb w } _ { m } { \pmb x } _ { m } [ t ] + n _ { 0 }\tag{5}
$$

通过接收波形进行采样进行信号协方差矩阵的L 估计， 为采样窗口大小， $\boldsymbol { y } ^ { \mathrm { H } }$ 为接收信号共轭转置，协方差矩阵可表示为

$$
G = \frac { 1 } { L } { \sum _ { l = 1 } ^ { L } } y y ^ { \mathrm { { H } } }\tag{6}
$$

无人机与边缘节点m的通信速率为

$$
R _ { u , m } = \beta _ { u , m } o _ { u , m } B _ { 0 } \mathrm { l o g } _ { 2 } ( 1 + \gamma _ { m } )\tag{7}
$$

$o _ { u , m }$ m是节点 分配给无人机的资源块个数， $B _ { 0 }$ 是单个资源块的通信带宽， $\gamma _ { u , m }$ 为无人机与节点 $m$ 之间通信链路的信干噪比。 $\gamma _ { u , m }$ 可具体表示为

$$
\gamma _ { u , m } = \frac { \left| \left| \pmb { w } _ { m } \boldsymbol { h } _ { u , m } \right| \right| ^ { 2 } } { \displaystyle \sum _ { m ^ { \prime } \neq m } ^ { M } \left| \left| \boldsymbol { I } _ { m ^ { \prime } , m } \pmb { w } _ { m ^ { \prime } } \boldsymbol { h } _ { u , m ^ { \prime } } \right| \right| ^ { 2 } + o _ { u , m } B _ { 0 } N _ { 0 } }\tag{8}
$$

其中， $I _ { m ^ { \prime } , m } = \{ 0 , 1 \}$ 表示给节点 $m ^ { ' }$ m和 之间的通信信号是否对无人机通信存在干扰，两个节点之间的距离小于一定阈值即 $d _ { m ^ { ' } , m } < d _ { \mathrm { t h } }$ 时，它们之间就存在干扰 $I _ { m ^ { \prime } , m } = 1$ m。节点 接收信号时信噪比必须满足一定阈值才能正确接收，因此存在式(9)的约束

$$
\gamma _ { u , m } > \gamma _ { u , m } ^ { \mathrm { t h } }\tag{9}
$$

无人机的总发射功率不能超过最大值，即

$$
\left| \left| \pmb { w } _ { m } \right| \right| ^ { 2 } \leq P _ { \operatorname* { m a x } } , \forall m \in \mathcal { M }\tag{10}
$$

与此同时，基站分配给无人机通信的带宽资源也是有限的，带宽资源块的个数不能超过一定值，即总带宽的约束，对于任意节点 $m \in \mathcal { M } =$ $\{ 1 , 2 , \ldots , M \}$ ，存在约束

$$
o _ { u , m } < o _ { \mathrm { m a x } }\tag{11}
$$

假设无人机与边缘节点的通信之间存在视距 (Line of Sight, LoS)链路和非视距(Non Line of Sight, NLoS)链路，二者的路径损耗分别为[17]

$$
h _ { \mathrm { L o S } } = 2 0 \log \left( \frac { 4 \pi f d _ { u , m } } { \mathrm { c } } \right) + \eta _ { \mathrm { L o S } }\tag{12}
$$

$$
h _ { \mathrm { N L o S } } = 2 0 \log \left( \frac { 4 \pi f d _ { u , m } } { \mathrm { c } } \right) + \eta _ { \mathrm { N L o S } }\tag{13}
$$

其中， $d _ { u , m }$ m是无人机到边缘计算节点 之间的距离， $\eta _ { \mathrm { L o S } }$ 和 $\lvert \eta _ { \mathrm { N L o S } }$ 是分别为LoS链路和NLoS链路的平f c τ均路径额外损耗， 是载波频率， 是光速， 是路径损耗系数。因此节点 $m$ 与无人机之间的无线信道增益 $h _ { u , m }$ 为

$$
h _ { u , m } = ( h _ { \mathrm { L o S } } + h _ { \mathrm { N L o S } } ) d _ { u , m } ^ { - \tau }\tag{14}
$$

## 2.3 视觉定位模型

无人机对节点m传输的数据量为 $D _ { m }$ 。则所有节点获取无人机信号波束的总时延为

$$
t ^ { \mathrm { s e n s e } } = \sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } }\tag{15}
$$

无人机与各节点的通信是串行过程，因此是累加求和。利用收集的各类数据(位置、姿态、环M境)进行计算，计算任务分配至周围 个节点，其总的时延为

$$
t ^ { \mathrm { c o m p u t e } } = \operatorname* { m a x } _ { m } \left\{ \beta _ { u , m } \frac { f ^ { \mathrm { n e t 1 } } D _ { m } } { c _ { m } ^ { 1 } } \right\}\tag{16}
$$

其中 $c _ { m } ^ { 1 }$ m是节点 分配给感知业务的可用算力。$f ^ { \mathrm { n e t 1 } }$ 表示飞行感知的神经网络计算中处理单位数据所需的CPU周期数， $( f ^ { \mathrm { n e t 1 } } D _ { m } ) / c _ { m }$ 是数据运算产生的时延，每个边缘节点的计算过程是并行的，因此是从中取最大值。

若检测到无人机状态异常，则需要对其飞行进行调整， $D ^ { \mathrm { c o n } }$ m为控制信令的数据量大小。节点 将控制信令传输给无人机的时延为

$$
t ^ { \mathrm { c o n t r o l } } = \frac { D ^ { \mathrm { c o n } } } { R _ { u , m } }\tag{17}
$$

对于感知业务的要求是较低的时延，以避免无人机无法及时调整姿态而发生事故或偏离航线，假T设感知总时延最高为 ，则有

$$
t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T\tag{18}
$$

感知结果融合机制的目标是通过整合多个边缘节点的局部感知信息，生成更准确、鲁棒的全局状m态估计。在此过程中，单个节点 的感知结果可以表示为 $s _ { m } = \left\{ d _ { u , m } , \theta _ { u , m } , \vartheta _ { u , m } \right\} , d _ { u , m } , \theta _ { u , m } , \vartheta _ { u , m }$ 分m别表示无人机到节点 的直线距离、方位角和俯仰角。最终汇聚的感知结果可表示为

$$
s ^ { \mathrm { s e n s e } } = \sum _ { m = 1 } ^ { M } \beta _ { u , m } s _ { m }\tag{19}
$$

本文使用拍摄图像、无人机传感器数据、接收信号协方差矩阵共同进行飞行位置感知。拍摄的图像中隐含了无人机当前的位置信息，机载传感器数据提供了无人机飞行状态的直接信息，协方差矩阵含有无人机到基站信号的接收信号强度(ReceivedSignal Strength Indication, RSSI)值，与路径损耗的距离有关，通过多个基站对无人机信号的接收，可实现利用协方差矩阵的多点联合位置感知[18]。

对于单个节点的感知结果，考虑到它是利用多模态数据联合进行感知，本文为每种数据类型的预测结果引入一个置信度 $\varphi$ ，它反映了该类数据的可靠程度，每次选取置信度最高的一类数据的感知结果作为该节点的预测结果

$$
s _ { m } = \arg \operatorname* { m a x } \{ \varphi _ { m } ^ { \mathrm { i m g } } , \varphi _ { m } ^ { \mathrm { s e n s e } } , \varphi _ { m } ^ { G } \}\tag{20}
$$

开始训练模型时，3种数据的置信度 $\varphi$ 设置为相同的值，随着轮次的不断增加， $\varphi$ 也随之不断更新，在训练过程中提升均方误差较低的数据置信度，降低均方误差大的数据的置信度。更新公式为

$$
\varphi _ { j } ^ { t } = \varphi _ { j } ^ { t - 1 } \frac { \varepsilon _ { j } ^ { t - 1 } } { \varepsilon _ { j } ^ { t } }\tag{21}
$$

当新一轮的训练误差 $\varepsilon _ { j } ^ { t }$ 小于上一轮时，比值$\varepsilon _ { j } ^ { t - 1 } / \varepsilon _ { j } ^ { t } { > } 1$ ，置信度 $\varphi _ { j } ^ { t }$ 增加，反之则减少。并且比值$\varepsilon _ { j } ^ { t - 1 } / \varepsilon _ { j } ^ { t }$ 越大，置信度增加量越多，训练效果越好，因此选择置信度最高的数据类型进行视觉定位。

利用收集的数据对负责感知任务的AI模型进行训练更新，其训练结果误差可定义为

$$
\varepsilon ^ { \mathrm { s e n s e } } = \sum _ { m = 1 } ^ { M } \beta _ { u , m } \varepsilon _ { m } ^ { \mathrm { s e n s e } }\tag{22}
$$

$\varepsilon _ { m } ^ { \mathrm { s e n s e } }$ 表示节点m的误差，可由该节点训练的均方误差(Mean Squared Error, MSE)进行表示。

## 2.4 故障检测模型

无人机将各子数据依次传输至各个边缘节点，之后各节点先对传输的图像进行增强处理，再使用本地的计算资源进行检测任务的神经网络计算，并使用联邦学习更新模型[19]，因此故障检测业务部分的时延取决于传输时间与计算时间之和最长的子图像数据，以及图像压缩时间，具体可表示为

$$
t ^ { \mathrm { i m g } } = \frac { f ^ { \mathrm { c p s } } \hat { \lambda } D ^ { \mathrm { i m g } } } { c _ { u } } + \sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } }
$$

$$
+ \operatorname* { m a x } _ { m } \left\{ \beta _ { u , m } \left( \frac { f ^ { \mathrm { r e p a i r } } D _ { m } } { c _ { m } ^ { 2 } } + \frac { f ^ { \mathrm { n e t 2 } } D _ { m } } { c _ { m } ^ { 2 } } \right) \right\}\tag{23}
$$

其中， $c _ { u }$ 为无人机机载算力， $f ^ { \mathrm { c p s } }$ 表示压缩计算中处理单位数据所需的CPU周期数， $c _ { m } ^ { 2 }$ m是节点 分配给故障检测业务的可用算力， $f ^ { \mathrm { n e t 2 } }$ 表示故障检测神经网络计算中处理单位数据所需的CPU周期数，$f ^ { \mathrm { r e p a i r } }$ 表示图像增强计算中处理单位数据所需的CPU周期数。这一部分与感知时延及其计算类似。在故障检测业务部分中，无人机消耗的能量为

$$
e _ { u } = \frac { \rho f ^ { \mathrm { c p s } } \hat { \lambda } D ^ { \mathrm { i m g } } \mu ^ { 2 } } { c _ { u } } + \sum _ { m = 1 } ^ { M } \beta _ { u , m } | | w _ { m } | | ^ { 2 } \frac { D _ { m } } { R _ { u , m } } + P ^ { \mathrm { f l y } } t ^ { \mathrm { i m g } }\tag{24}
$$

其中， $\mu$ 为无人机机载CPU频率， $\rho$ 为无人机MCPU能耗系数， 是分配的边缘节点的总数，$| | \pmb { w } _ { m } | | ^ { 2 }$ 是分配给无人机与节点 $m$ 之间的通信功率，$P ^ { \mathrm { f l y } }$ 为无人机的飞行功率。计算节点对感知业务和故障检测业务的处理是并行的，故障检测业务中除了有神经网络计算，还有一个图像修复的部分，花费的时间比感知业务更多。无人机消耗的总能量不得超过最大值，即

$$
e _ { u } \le E _ { \operatorname* { m a x } }\tag{25}
$$

边缘节点利用收集的数据进行模型的训练与更新，引入检测误差描述了所有节点对故障结果的预测情况，误差越小，巡检结果越精确，将检测误差定义为

$$
\varepsilon ^ { \mathrm { i m g } } = \sum _ { m = 1 } ^ { M } \beta _ { u , m } \varepsilon _ { m } ^ { \mathrm { i m g } }\tag{26}
$$

$\varepsilon _ { m } ^ { \mathrm { i m g } }$ 表示节点m的误差，可由该节点训练的MSE进行表示。

## 2.5 优化问题形成

根据以上分析，需要使无人机消耗的能量最小，优化变量包括边缘节点选择策略、带宽分配策略、功率分配策略、计算资源分配策略、拍摄图像数据量、实际压缩比例。优化目标为总能耗 $e _ { u }$

$$
\mathrm { P } \colon \operatorname* { m i n } _ { \substack { \beta , o , w , c , \hat { \lambda } , D ^ { \mathrm { i m g } } } } e _ { u }\tag{27a}
$$

$$
\mathrm { s . t . } t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T\tag{27b}
$$

$$
e _ { u } \le E _ { \mathrm { m a x } }\tag{27c}
$$

$$
o _ { u , m } \leq o _ { \operatorname* { m a x } } , \forall m \in \mathcal { M }\tag{27d}
$$

$$
\left. \left. \pmb { w } _ { m } \right. \right. ^ { 2 } \leq P _ { \operatorname* { m a x } } , \forall m \in \mathcal { M }\tag{27e}
$$

$$
\gamma _ { u , m } > \gamma _ { u , m } ^ { \mathrm { t h } } , \forall m \in \mathcal { M }\tag{27f}
$$

$$
c _ { m } ^ { 1 } + c _ { m } ^ { 2 } \leq c _ { m } , \forall m \in \mathcal { M }\tag{27g}
$$

$$
k \leq \sum _ { m = 1 } ^ { M } \beta _ { u , m } \leq K
$$

$$
\beta _ { u , m } = \{ 0 , 1 \}\tag{27h}
$$

(27i)

$$
D ^ { \mathrm { i m g } } \geq D _ { \mathrm { m i n } }\tag{27j}
$$

$$
\varepsilon ^ { \mathrm { s e n s e } } \leq \varepsilon _ { \mathrm { m a x } } ^ { \mathrm { s e n s e } }\tag{27k}
$$

$$
\varepsilon ^ { \mathrm { i m g } } \leq \varepsilon _ { \mathrm { m a x } } ^ { \mathrm { i m g } }\tag{27l}
$$

其中，式(27b)表示感知任务总时延约束，式(27c)表示无人机能耗约束，式(27d)、式(27e)、式(27g)分别表示带宽、功率、算力资源约束，式(27f)表示接受信号信噪比必须大于门限才能准确接收。式(27h)k K表示无人机最少与 个节点建立连接，最多与 个节点建立连接，式(27j)表示对神经网络图像数据量的最低需求，式(27k)和式(27l)分别表示感知误差(距离)和故障检测误差(准确率)上界约束。原优化问题在目标函数中均存在最大值项、对数项以及整数0-1变量，这些都是求解优化问题时的重难点，且目标函数表达式复杂、约束较多，属于混合整数非线性规划问题(Mixed Integer NonLinear Pro-gramming problem, MINLP)，直接求解较为困难，为此本文提出4阶段多维资源巡检调度协同优化算法进行求解。

## 3    4阶段多维资源巡检调度协同优化算法

通过对总能耗 $e _ { u }$ 的分析，可以发现其由以下4个部分组成， $( \rho f ^ { \mathrm { c p s } } \mu ^ { 2 } + P ^ { \mathrm { f l y } } ) \hat { \lambda } D ^ { \mathrm { l m g } } / c _ { u }$ 表示无人机图Dimg 像压缩消耗的能量，由图像数据量 和实际压缩λˆ比例 决定， $\sum _ { m = 1 } ^ { M } \beta _ { u , m } \left( | | \pmb { w } _ { m } | | ^ { 2 } + P ^ { \mathrm { f l y } } \right) D _ { m } / R _ { u , m }$ $+ P _ { \mathrm { \normalfont } } ^ { \mathrm { f l y } } \mathrm { m a x } \{ \beta _ { u , m } \left( f ^ { \mathrm { r e p a i r } } D _ { m } / c _ { m } ^ { 2 } + f ^ { \mathrm { n e t 2 } } D _ { m } / c _ { m } ^ { 2 } \right) \}$ 此 部m  
分能耗受节点选择策略 $\beta$ 影响，应首先根据各节点的资源情况选择出合适的节点，再进行后续的带宽、功率、算力的优化， $\sum _ { m = 1 } ^ { M } \beta _ { u , m } \left( \left| | \pmb { w } _ { m } | \right| ^ { 2 } + P ^ { \mathrm { f l y } } \right)$ $D _ { m } / R _ { u , m }$ 表示无人机向各节点传输信息的通信能耗，受带宽资源块个数 $_ { o _ { u , m } }$ 和功率 $| w _ { m } | | ^ { 2 }$ 影响，$P ^ { \mathrm { f l y } } \underset { m } { \mathrm { m a x } } \{ \beta _ { u , m } \left( f ^ { \mathrm { r e p a i r } } D _ { m } / c _ { m } ^ { 2 } + f ^ { \mathrm { n e t 2 } } D _ { m } / c _ { m } ^ { 2 } \right) \}$ } 表 示 边

缘计算需要的时间造成的无人机悬停消耗的能量，受算力分配 $c _ { m } ^ { 1 }$ 和 $c _ { m } ^ { 2 }$ 的影响。根据以上分析，可以把原始问题转化为以下4个子问题。

## 3.1 子问题1(无人机图像处理优化)：数据驱动边界学习法

$$
\mathrm { P 1 } \colon \operatorname* { m i n } _ { \hat { \lambda } , D ^ { \mathrm { i m g } } } \frac { ( \rho f ^ { \mathrm { c p s } } \mu ^ { 2 } + P ^ { \mathrm { f l y } } ) \hat { \lambda } D ^ { \mathrm { i m g } } } { c _ { u } }\tag{28a}
$$

$$
\mathrm { s . t . } \ D ^ { \mathrm { i m g } } \geq D _ { \mathrm { m i n } }\tag{28b}
$$

$$
\varepsilon ^ { \mathrm { s e n s e } } \leq \varepsilon _ { \mathrm { m a x } } ^ { \mathrm { s e n s e } }\tag{28c}
$$

$$
\varepsilon ^ { \mathrm { { i m g } } } \leq \varepsilon _ { \mathrm { { m a x } } } ^ { \mathrm { { i m g } } }\tag{28d}
$$

λˆ在子问题1中可以看出实际压缩比例 越小和图像Dimg 数据量 越少，目标函数就越小，但是实际压缩比例对后续通信和计算的能耗影响远大于此部分压缩图像的能耗影响，因此考虑尽可能大的压缩率和尽可能少的数据量，然而约束中存在对神经网络的误差约束，由于其解释性差，很难直接得出与能耗λˆ Dimg 相关的数学表达式， 和 都能直接影响训练结λˆ Dimg 果的误差。因此本部分通过用不同的 和 多次训练神经网络得到误差，确定出二者与两个误差之间的联系，把误差约束 $\varepsilon ^ { \mathrm { s e n s e } } \leq \varepsilon _ { \mathrm { m a x } } ^ { \mathrm { s e n s e } }$ ε img ≤ ε imgmax 和 转Dimg λˆ化为 的下界约束和 的上界约束，进行求得本部分的最优解。

## 3.2 子问题2(边缘节点选择优化)：双辅助MILP转化法

$$
\mathrm { P 2 } \colon \operatorname* { m i n } _ { \beta } \sum _ { m = 1 } ^ { M } \beta _ { u , m } \left( | | \boldsymbol { w } _ { m } | | ^ { 2 } + P ^ { \mathrm { f l y } } \right) \frac { D _ { m } } { R _ { u , m } } + P ^ { \mathrm { f l y } }
$$

$$
\cdot \operatorname* { m a x } _ { m } \left\{ \beta _ { u , m } \left( { \frac { f ^ { \mathrm { r e p a i r } } D _ { m } } { c _ { m } ^ { 2 } } } + { \frac { f ^ { \mathrm { n e t 2 } } D _ { m } } { c _ { m } ^ { 2 } } } \right) \right\}\tag{29a}
$$

$$
\mathrm { s . t . } t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T\tag{29b}
$$

$$
k \leq \sum _ { m = 1 } ^ { M } \beta _ { u , m } \leq K\tag{29c}
$$

$$
\beta _ { u , m } = \{ 0 , 1 \}\tag{29d}
$$

这是一个0-1规划问题，关键在于选择出尽可能少的节点参与飞行感知和故障检测的计算。本文考虑通常情况，即参与节点的数量越少，能耗越小，取 $et { } { _ { m } } et { } { _ { M } } et { } { _ { M } } et { } { _ { M } } et { } { _ { M } } et { } { _ { M } } \ s _ { u , m } = k$ tcontrol。 不受节点选择的影响，此处视为一个常量，此外 $T _ { \mathrm { t h } } = T - t ^ { \mathrm { c o n t r o l } }$ 。考虑到目标函数和约束条件中都有最大值项，本文引入辅助两个辅助变量 $z _ { 1 }$ , $z _ { 2 }$ 将max函数线性化，将P2写为

$$
\mathrm { P 2 ^ { \prime } } \colon \operatorname* { m i n } _ { z _ { 1 } , \beta } \sum _ { m = 1 } ^ { M } \beta _ { u , m } \left( \left| \left| \pmb { w } _ { m } \right| \right| ^ { 2 } + P ^ { \mathrm { f l y } } \right) \frac { D _ { m } } { R _ { u , m } } + P ^ { \mathrm { f l y } } z _ { 1 }\tag{30a}
$$

$$
\mathrm { s . t . } \ \sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } } + z _ { 2 } \leq T _ { \mathrm { t h } }\tag{30b}
$$

$$
z _ { 1 } \geq \beta _ { u , m } \Bigg ( \frac { f ^ { \mathrm { r e p a i r } } D _ { m } } { c _ { m } ^ { 2 } } + \frac { f ^ { \mathrm { n e t 2 } } D _ { m } } { c _ { m } ^ { 2 } } \Bigg ) \ \forall m \in \mathcal { M }\tag{30c}
$$

$$
z _ { 2 } \geq \beta _ { u , m } \frac { f ^ { \mathrm { n e t 1 } } D _ { m } } { c _ { m } ^ { 1 } } \forall m \in \mathcal { M }\tag{30d}
$$

$$
\sum _ { m = 1 } ^ { M } \beta _ { u , m } = k\tag{30e}
$$

$$
\beta _ { u , m } = \{ 0 , 1 \}\tag{30f}
$$

引入辅助变量后，目标函数和所有约束都是线性的可以使用标准的MILP求解器进行求解。

## 3.3 子问题3(无人机通信优化)：基于SCA的带宽功率联合优化算法

$$
\mathrm { P 3 } \colon \operatorname* { m i n } _ { o , w } \sum _ { m = 1 } ^ { M } \beta _ { u , m } \left( \left| \left| w _ { m } \right| \right| ^ { 2 } + P ^ { \mathrm { f l y } } \right) \frac { D _ { m } } { R _ { u , m } }\tag{31a}
$$

$$
\mathrm { s . t . } t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T\tag{31b}
$$

$$
o _ { u , m } \leq o _ { \operatorname* { m a x } } , \forall m \in \mathcal { M }\tag{31c}
$$

$$
\left. \left. \pmb { w } _ { m } \right. \right. ^ { 2 } \leq P _ { \operatorname* { m a x } } , \forall m \in \mathcal { M }\tag{31d}
$$

$$
\gamma _ { u , m } > \gamma _ { u , m } ^ { \mathrm { t h } } , \forall m \in \mathcal { M }\tag{31e}
$$

在该问题中，带宽资源块分配 $^ o$ 是离散值，功率分  
w配 是连续值。在约束条件 $t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } }$   
$\leq T$ 中， $t ^ { \mathrm { c o m p u t e } } = \operatorname* { m a x } _ { \omega } \{ \beta _ { u , m } f ^ { \mathrm { n e t 1 } } D _ { m } / c _ { m } ^ { 1 } \}$ 中包含最m  
大值约束，这会使得问题难以求解，因此将 $t ^ { \mathrm { s e n s e } } +$ M  
tcompute + tcontrol ∑ m= βu,mDm/Ru,m + max{βu,m

$$
\sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } } + \beta _ { u , 1 } \frac { f ^ { \mathrm { n e t 1 } } D _ { 1 } } { c _ { 1 } ^ { 1 } } + \frac { D ^ { \mathrm { c o n } } } { R _ { u , m } } \leq T\tag{31f}
$$

$$
\sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } } + \beta _ { u , 2 } \frac { f ^ { \mathrm { n e t 1 } } D _ { 2 } } { c _ { 2 } ^ { 1 } } + \frac { D ^ { \mathrm { c o n } } } { R _ { u , m } } \leq T\tag{31g}
$$

$$
\sum _ { m = 1 } ^ { M } \beta _ { u , m } \frac { D _ { m } } { R _ { u , m } } + \beta _ { u , M } \frac { f ^ { \mathrm { n e t 1 } } D _ { M } } { c _ { M } ^ { 1 } } + \frac { D ^ { \mathrm { c o n } } } { R _ { u , m } } \leq T\tag{31h}
$$

对于该问题中的整数变量和约束，将OFDM中的离散带宽资源块松弛化，引入连续带宽变量$B _ { m } = o _ { u , m } B _ { 0 }$ ，将原先的变量 $_ { \mathbf { \delta } }$ B替换为 ，带宽约束 $o _ { u , m } \leq o _ { \mathrm { m a x } }$ 变为连续值约束 $B _ { u , m } \leq B _ { \mathrm { m a x } }$ 。对于表达式中的对数项采用SCA凸近似算法，将非凸项转化为凸函数，使得该问题转换为凸优化问题，具体过程如下：

在 对 数 项 的 表 达 式 中 ， 信 噪 比||wmhu,m||2γu,m = ||I m′ ,mw m′ hu,m ′ ||2 + ou,mB0N0 m′̸=m将 干 扰 项 与 噪 声 项 用 一 个 常 量 代 替 $I _ { m } ^ { ( t ) } =$ $\sum _ { m ^ { \prime } \neq m } \big | \big | I _ { m ^ { \prime } , m } \pmb { w } _ { m } ^ { ( t - 1 ) } h _ { u , m ^ { \prime } } \big | \big | ^ { 2 } + B _ { m } ^ { ( t - 1 ) } N _ { 0 }$ ， 其 中t是SCA的迭代轮次， $w _ { m } ^ { ( t - 1 ) }$ 和 $B _ { m } ^ { ( t - 1 ) }$ 是上一轮迭代得到的结果，之后 $\gamma _ { u , m }$ 就转换成了关于 ${ \pmb w } _ { m }$ 的2次表达式，是凸的。之后将对数项泰勒展开

$$
\begin{array} { r l r } {  { R _ { m } = B _ { m } \mathrm { l o g } _ { 2 } ( 1 + \gamma _ { u , m } ) \approx B _ { m } ( \log _ { 2 } { ( 1 + \gamma _ { u , m } ^ { ( t ) } ) }  } } \\ & { } & \\ & { } & {  + \frac { ( \gamma _ { u , m } + \gamma _ { u , m } ^ { ( t ) } ) } { \ln { 2 ( 1 + \gamma _ { u , m } ^ { ( t ) } ) } } ) } & { ( 3 ) } \end{array}\tag{2}
$$

其中泰勒展开的位置 $\gamma _ { u , m } ^ { t }$ 是用 $w _ { m } ^ { ( t - 1 ) }$ 和 $B _ { m } ^ { ( t - 1 ) }$ 代 入计算得到的，将对数项近似转换为凸函数。对于信噪比约束项，首先用近似值 $\chi _ { u , m } = \big | \big | \pmb { w } _ { m } ^ { ( t - 1 ) } h _ { u , m } \big | \big | ^ { 2 }$ 替 换 $| | \boldsymbol { w } _ { m } h _ { u , m } | | ^ { 2 }$ ， 将 分 式 转 换 为 $\gamma _ { m } ^ { \mathrm { t h } } ( \sum _ { m ^ { \prime } \neq m }$ $| | I _ { m ^ { \prime } , m } w _ { m ^ { \prime } } h _ { u , m ^ { \prime } } | | ^ { 2 } + o _ { u , m } B _ { 0 } N _ { 0 } \big ) < \chi _ { u , m } ,$ 此约束是B w凸的，因此此时对于单个变量 或 是一个凸优化问题，即可用CVX等常用工具对二者进行交替求解。设P3的目标函数表示为 $e _ { 2 }$ ，迭代的收敛阈值为 ${ \delta _ { 2 } }$ 。 0

3.4 子问题4(边缘节点算力优化)：下界解析分配法

$$
\mathrm { P 4 } { \colon } \operatorname* { m i n } _ { c } P ^ { \mathrm { f l y } } \operatorname* { m a x } _ { m } \left\{ \beta _ { u , m } \left( \frac { f ^ { \mathrm { r e p a i r } } D _ { m } } { c _ { m } ^ { 2 } } + \frac { f ^ { \mathrm { n e t } 2 } D _ { m } } { c _ { m } ^ { 2 } } \right) \right\}\tag{32a}
$$

$$
\mathrm { s . t . } t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T\tag{32b}
$$

$$
c _ { m } ^ { 1 } + c _ { m } ^ { 2 } \leq c _ { m } , \forall m \in \mathcal { M }\tag{32c}
$$

子问题4是关于每个节点算力资源分配的优化，从优化目标中可以看出，该优化目标取决于所有选取的节点中最大的计算时间，因此需要使得每个节点的计算时间最小化。从表达式中可以看出，优化该问题，只需在满足感知时延和总算力的约束下，每个节点为故障检测业务分配尽可能多的计算资源，即使得 $c _ { m } ^ { 2 }$ 尽可能大。对于感知时延约束而言，在固定其它优化变量的条件下，感知时延最终取决于 $t ^ { \mathrm { c o m p u t e } } = \operatorname* { m a x } _ { m } \{ \beta _ { u , m } f ^ { \mathrm { n e t 1 } } D _ { m } / c _ { m } ^ { 1 } \}$ ，因此有 $\operatorname* { m a x } _ { m } \{ \beta _ { u , m } f ^ { \mathrm { n e t 1 } } D _ { m } / c _ { m } ^ { 1 } \} \leq T - t ^ { \mathrm { s e n s e } } - t ^ { \mathrm { c o n t r o l } }$ ， 设$T ^ { t \mathrm { h } } = T - t ^ { \mathrm { s e n s e } } - t ^ { \mathrm { c o n t r o l } }$ 。可以最大值函数的约束转化为对于 $\forall m \in { \mathcal { M } }$ ，均有 $\beta _ { u , m } f ^ { \mathrm { n e t 1 } } D _ { m } / c _ { m } ^ { 1 } \leq T -$ $t ^ { \mathrm { s e n s e } } - t ^ { \mathrm { c o n t r o l } }$ ， 将 等 式 代 换 可 得 到 $c _ { m } ^ { 1 } \geq \beta _ { u , m }$ $f ^ { \mathrm { n e t 1 } } D _ { m } / ( T - t ^ { \mathrm { s e n s e } } - t ^ { \mathrm { c o n t r o l } } )$ ，因此分配给感知业务的 算 力 存 在 一 个 下 界 $c _ { m , \mathrm { m i n } } ^ { 1 } = \beta _ { u , m } f ^ { \mathrm { n e t 1 } } D _ { m } /$ $( T - t ^ { \mathrm { s e n s e } } - t ^ { \mathrm { c o n t r o l } } )$ ，每个节点的 $c _ { m } ^ { 1 }$ 必须大于等于该下界才能满足时延约束。每个节点取 $c _ { m } ^ { 1 } =$ $c _ { m , \mathrm { m i n } } ^ { 1 } , c _ { m } ^ { 2 } = c _ { m } - c _ { m } ^ { 1 }$ ，该优化目标取决于推理计算出的最小的 $c _ { m } ^ { 2 }$ 。

## 3.5 4阶段多维资源巡检调度协同优化算法

对于子问题P1，可以使用深度学习的方法，通过在不同的数据量和实际压缩比例的条件下反复训练，最终得到一组数据量、实际压缩比例与模型误差的关系，再根据给出的误差阈值确定出数据量和实际压缩比例的最小值并以此值固定，从而减少能耗。在后面3个子问题中，本文采用交替优化的方法进行求解，在求解每个子问题时固定其它子问题的变量。在子问题耦合方面，存在一个共同的约束 $t ^ { \mathrm { s e n s e } } + t ^ { \mathrm { c o m p u t e } } + t ^ { \mathrm { c o n t r o l } } \leq T$ ，交替优化算法不受其影响，因为求解每个子问题时都会满足此项约束，则最终的解也一定满足。设迭代的收敛阈值为δ，本文的优化算法可由算法1进行描述。

算法中存在内外两层交替优化，假设内层的迭代次数为 $n _ { 1 }$ ，外层的迭代次数为 $n _ { 2 }$ 。对于子问题P 2，这是一个MILP问题，其复杂度通常为 $O ( 2 ^ { M } )$ ，子问题P3通过SCA近似后成为凸问题，其复杂度为 $O ( n _ { 1 } M ^ { 3 . 5 } )$ ，子问题P4的复杂度是线性的 $O ( M )$ 。综合起来，整个问题的复杂度为$O ( n _ { 2 } ( 2 ^ { M } + n _ { 1 } M ^ { 3 . 5 } + M ) )$

## 4    仿真分析

本文使用python3.7下的pytorch1.13以及mat-labR2020a进行仿真，场景中总边缘计算节点的数Dtotal 量M设置为20，总数据量 为15 MB，无人机飞行功率设置为100 W，飞行高度为100 m[20]，飞行速度为10 m/s，基站为通感一体的5G基站， $\mathrm { L o S }$ 链路损耗 $\eta _ { \mathrm { L o S } }$ 设置为1， $\mathrm { N L o S }$ 链路损耗 $\eta _ { \mathrm { N L o S } }$ 设置为τ 20， 设置为2，感知业务计算强度 $f ^ { \mathrm { n e t 1 } }$ ，故障检测业务计算强度 $f ^ { \mathrm { n e t 2 } }$ ，图像修复计算强度 $f ^ { \mathrm { r e p a i r } }$ ， 压缩计算强度 $f ^ { \mathrm { c p s } }$ 均设置为800 flop/bit，单位CPU周期的能量消耗 $\rho$ 为 $2 \times 1 0 ^ { - 1 0 }$ J/circle, CPU频率 $\cdot \mu$ 为2 GHz。本文使用 $\mathrm { R e s N e t { - } } 1 8$ 作为图像处理神经网络完成深度学习，故障检测数据集使用中国电力线绝缘子数据集(CPLID)，视觉定位数据集使用UAV-VisLoc数据集。本文使用了4种优化算法进行对比实验，对比算法包括启发式学习的PSODD算法[13]，博弈论的TSSDG算法[21]，以及深度强化学习的DAM-MAC-DDQN算法[22]。

图2和图3展示了故障检测任务中不同数据量(百分比数目)和实际压缩比的影响。从图2中可以看出，适当减少用于训练神经网络的图像数据量，可以一定程度地提升模型精度，图中使用60%的数据量效果最优，收敛时的模型精度达到了0.922，这是由于适当减少数据量减轻了数据过拟合的影响。而当数据量减少到40%时，最终的精度只有0.823，这是由于样本数量不足导致的欠拟合。由图3中可以看出，实际压缩比对模型精度的整体影响较小。使用原始高清图像进行训练，准确度达到了0.941，而即使将图像压缩至原始大小的40%，模型的训练精度仍有0.891，这是由于故障检测只需关注图像中的设备情况且更为明显，环境信息的影响较小。以上两图说明，在故障检测这方面，无人机可以适当减少图像数据量的传输，在保障准确率的同时减少能耗，而图像压缩率这方面，可在保障图像不失真的情况下尽可能使用较高的压缩率。

图4和图5展示了视觉定位任务中不同数据量和实际压缩比的影响。图4可以看出，在初始100%数据用于训练时，稳定状态下的模型误差大约有16.12 m。而使用80%数据进行训练，模型误差下降到了13.76 m，性能得到了提升。继续减少数据量，模型性能开始下降，使用40%数据进行训练时，误差高于初始100%数据训练下的结果，这说明欠拟合对视觉定位的影响较大，使用80%左右的数据进行模型训练最为合适。相比于故障检测，实际压缩比对视觉定位的影响更为明显，图5可以看出，在实际压缩比为80%时模型性能最优，模型误差为12.58 m。而随着实际压缩比继续提升，模型性能开始变差，压缩率为40%时尽管收敛速度有所提升，但误差却达到了17.23 m。比起故障检测，视觉定位对图像的全局信息更为敏感，因为其中包含无人机所处环境，需要更高的图像质量。以上两图说明，在视觉定位这方面，应谨慎选择合适的数据量和压缩率，在保障飞行安全的前提下减少能耗。

算法 1  4阶段多维资源巡检调度协同优化算法  
Dimg λˆ Dimg λˆ εsense εimg1：使用不同的 和 训练神经网络，确定出 , 与两种误差 , 的关系图  
Dimg2：根据关系图得到 的下界和ζ的上界并以之作为二者的解  
3：初始化其余优化变量 ${ \bf \beta } ( 0 ) , B ^ { ( 0 ) } , { \bf w } ^ { ( 0 ) } , { \bf c } ^ { ( 0 ) }$   
4：while $( e _ { u } ^ { ( t ) } - e _ { u } ^ { ( t - 1 ) } > \delta )$   
5： 固定其余变量 $B ^ { ( t - 1 ) } , \pmb { w } ^ { ( t - 1 ) } , \pmb { c } ^ { ( t - 1 ) }$ β(t)，使用MILP求解器求解  
6： 固定其余变量 $\cdot \beta ^ { ( t ) } , c ^ { ( t - 1 ) }$ ，使用SCA近似算法求解 $B ^ { ( t ) } , \boldsymbol { w } ^ { ( t ) }$   
7： 固定其余变量 $\mathbf { \nabla } \cdot \beta ^ { ( t ) } , B ^ { ( t ) } , \mathbf { w } ^ { ( t ) }$ tsense + tcompute + tcontrol ≤ T c1m，根据 求解每个节点 的下界 $\cdot c _ { m , \mathrm { m i n } } ^ { 1 }$ in c1m(t) =并取 c1m,min ,  
${ c _ { m } ^ { 2 } } ^ { ( t ) } = c _ { m } - { c _ { m , \mathrm { m } } ^ { 1 } }$ in  
β(t), B(t), w(t), c(t) e(t)8： 将 代入目标函数求得 u  
9：end while

<!-- image-->

图 2 数据量对故障检测精度影响  
1.0  
-100%  
0.8 60% 80%  
40%  
0.6  
0.4  
0.2  
0  
100200300400500  
训练时间 (s)  
图 3 实际压缩比对故障检测精度影响

图 4 数据量对视觉定位精度影响  
<!-- image-->  
图 5 实际压缩比对视觉定位精度影响

图6展示了不同算法下对故障检测准确率和视觉定位误差距离的对比，可以看出，本文提出的算法在故障检测和视觉定位的AI模型训练方面均优于其它算法，其中故障检测准确率达到了0.922，误差距离达到了15.735 m，相比其它3种算法均有较大的提升，有效提升了两项AI任务完成的准确程度，提升了性能。

图7和图8分别展示了无人机总能耗与节点平均算力和总带宽之间的关系。随着可调用的算力和带宽资源的增加，边缘节点的计算时间、无人机传输数据的时间得到下降，对应的无人机在巡检点悬停等待的时间也下降，从而使得无人机总能耗下降。并且在二者之间，总带宽的对能耗的影响更大。在4种不同算法的对比中，本文所提算法性能最优。在平均算力关系图中，本文算法与其余3种能耗分别下降了0.63%, 1.11%和1.59%的百分比。在总带宽关系图中，本文算法与其余3种能耗分别下降了约1 2.19%, 2.91%和3.45%的百分比。以上两图说明，本文所提方法对无人机的总能耗下降有所改善，与对比算法相比具有一定的优越性。

<!-- image-->  
图 6 不同算法下准确率与误差距离对比

<!-- image-->  
图 7 无人机能耗与节点平均算力关系图

<!-- image-->  
图 8 无人机能耗与总带宽关系图

## 5    结束语

本文提出了一种面向感知与AI协同优化的智能无人机巡检多维资源联合优化算法，形成了以无人机能耗最小化为目标，带宽、功率、算力、节点选择、数据量和实际压缩比例为变量的优化问题，完成了故障检测与视觉定位两项AI业务的同时最小化无人机的能耗。仿真结果表明，本算法能够降低无人机的总能耗，提升了模型训练的精度。此研究针对单架无人机巡检的应用场景，未来可进一步研究更加复杂的多无人机协同巡检场景，并引入更多业务进行综合研究。

## 参 考 文 献

谷美颖, 李航, 张家伟, 等. 基于视觉的无人机定位与导航方法[1]研究综述[J]. 电子学报, 2025, 53(3): 651–685. doi: 10.12263/DZXB.20240699.GU Meiying, LI Hang, ZHANG Jiawei, et al. A review ofvision-based UAV localization and navigation methods[J].Acta  Electronica  Sinica, 2025, 53(3): 651–685. doi: 10.12263/DZXB.20240699.

[2] YANG Yuwei, HE Minheng, LIU Juan, et al. The UAV

intelligent  inspection  technology  in  the  transformer substation inspection[C]. 2025 5th Power System and Green Energy Conference, Hong Kong, China, 2025: 215–219. doi: 10.1109/PSGEC66102.2025.11151059.

肖国德, 张贺. 基于深度学习的无人机输变配一体化巡检系统[3][J]. 自动化应用, 2024, 65(23): 4–6. doi: 10.19769/j.zdhy.2024.23.002.XIAO  Guode  and  ZHANG  He. Deep  learning-basedunmanned aerial vehicle integrated inspection system forpower  transmission  transformation  and  distribution[J].Automation Application, 2024, 65(23): 4–6. doi: 10.19769/j.zdhy.2024.23.002.

HE  Zhenyao,  XU  Wei,  SHEN  Hong, et al.  Integrated[4] sensing and full-duplex communication: Joint transceiver beamforming  and  power  allocation[C].  2023  IEEE International Conference on Acoustics, Speech and Signal Processing,  Rhodes  Island,  Greece,  2023:  1–5.  doi: 10.1109/ICASSP49357.2023.10097111.

DONG  Huanyu,  LI  Peichun,  DAI  Minghui, et al.[5] Coordinated  multi-point  aided  integrated  sensing, communication  and  computation  system:  An  energy efficient design[C]. 2024 IEEE/CIC International Conference on  Communications  in  China,  Hangzhou,  China,  2024: 54–59. doi: 10.1109/ICCC62479.2024.10681790.

DENG  Cailian,  FANG  Xuming,  and  WANG  Xianbin.[6] Integrated sensing, communication, and computation with adaptive DNN Splitting in multi-UAV networks[J]. IEEE Transactions on Wireless Communications, 2024, 23(11): 17429–17445. doi: 10.1109/TWC.2024.3453650.

ZHANG Ruizhi, ZHANG Ying, TANG Rui, et al. A joint[7] UAV Trajectory, User association, and beamforming design strategy  for  multi-UAV-assisted  ISAC  systems[J]. IEEE Internet of Things Journal, 2024, 11(18): 29360–29374. doi: 10.1109/JIOT.2024.3430390.

ZHANG  Xin,  CHANG  Zheng,  ZHANG  Guopeng, et al.[8] Trajectory optimization and resource allocation for time minimization in the UAV-enabled MEC system[C]. 2022 IEEE  Wireless  Communications  and  Networking Conference,  Austin,  USA,  2022:  333–338.  doi: 10.1109/ WCNC51071.2022.9771719.

王怡, 覃团发, 韦睿, 等. SAG-MEC网络下支持WPT的无人机[9] 动态任务卸载与资源分配[J/OL].  计算机工程,  1–10. https://doi.org/10.19678/j.issn.1000-3428.0070030, 2024. WANG  Yi,  QIN  Tuanfa,  WEI  Rui, et al.  Dynamic  task unloading and resource allocation of UAVs supported by WPT in SAG-MEC network[J/OL]. Computer Engineering, 1–10.  https://doi.org/10.19678/j.issn.1000-3428.0070030, 2024.

王轶宇, 钱鹏智, 张余, 等. 基于联盟博弈的无人机集群任务分[10]配与频谱资源联合规划方法[J]. 中国电子科学研究院学报,2024, 19(7): 647–657. doi: 10.3969/j.issn.1673-5692.2024.07.009.WANG  Yiyu,  QIAN  Pengzhi,  ZHANG  Yu, et al. Taskallocation and joint spectrum resource planning for UAVcluster based on alliance game theory[J]. Journal of ChinaAcademy of Electronics and Information Technology, 2024,19(7): 647–657. doi: 10.3969/j.issn.1673-5692.2024.07.009.

BAYESSA  G  A,  CHAI  Rong,  LIANG  Chengchao, et al.[11] Content  fetching  delay  optimization-based  caching  and resource  allocation  for  UAV-enabled  networks[J]. IEEE Access, 2024, 12: 62429–62447. doi: 10.1109/ACCESS.2024. 3395279.

GAO  Yuan,  DING  Yu,  WANG  Ye, et al.  Deep[12] reinforcement learning-based trajectory optimization and resource  allocation  for  secure  UAV-enabled  MEC n e t w o r k s [ C ] .   I E E E   C o n f e r e n c e   o n   C o m p u t e r Communications  Workshops,  Vancouver,  Canada,  2024: 1–5. doi: 10.1109/INFOCOMWKSHPS61880.2024.10620895.

WANG Jiawei and SUN Haifeng. Joint resource allocation[13] and trajectory optimization for computation offloading in UAV-enabled  mobile  edge  computing[C].  2024  6th International Conference on Communications, Information System  and  Computer  Engineering,  Guangzhou,  China, 2024: 302–307. doi: 10.1109/CISCE62493.2024.10653394.

HU Bintao, ISAAC M, AKINOLA O M, et al. Federated[14] learning empowered resource allocation in UAV-assisted edge intelligent systems[C]. 2023 IEEE 3rd International Conference  on  Computer  Communication  and  Artificial Intelligence,  Taiyuan,  China,  2023:  336–341.  doi: 10.1109/CCAI57533.2023.10201325.

YIN Sixing and YU F R. Resource allocation and trajectory[15] design in UAV-aided cellular networks based on multiagent reinforcement learning[J]. IEEE Internet of Things Journal, 2022, 9(4): 2933–2943. doi: 10.1109/JIOT.2021.3094651.

PENG Sicong, LI Bin, LIU Lei, et al. Trajectory design and[16] resource  allocation  for  multi-UAV-assisted  sensing, communication, and edge computing integration[J]. IEEE Transactions on Communications, 2025, 73(4): 2847–2861. doi: 10.1109/TCOMM.2024.3478115.

AL-HOURANI  A,  KANDEEPAN  S,  and  LARDNER  S.[17] Optimal  LAP  altitude  for  maximum  coverage[J]. IEEE Wireless Communications Letters, 2014, 3(6): 569–572. doi: 10.1109/LWC.2014.2342736.

XIE  Hao,  ZHANG  Tiankui,  XU  Xiaoxia, et al. Joint[18] sensing, communication, and computation in UAV-assisted systems[J]. IEEE Internet of Things Journal, 2024, 11(18):

29412–29426. doi: 10.1109/JIOT.2024.3362937.

MCMAHAN  B,  MOORE  E,  RAMAGE  D, et al.[19] Communication-efficient learning of deep networks from decentralized data[C]. Proceedings of the 20th International Conference  on  Artificial  Intelligence  and  Statistics,  Fort Lauderdale, USA, 2017: 1273–1282.

ISMAIL M and LE Longbao. Computation offloading and[20] resource  allocation  for  deep  neural  network  inference  in UAV wireless networks[C]. IEEE International Conference on Communications, Montreal, Canada, 2025: 832–837. doi: 10.1109/ICC52391.2025.11161364.

WANG Die, JIA Yunjian, LIANG Liang, et al. Resource[21] allocation in blockchain integration of UAV-enabled MEC networks: A stackelberg differential game approach[J]. IEEE

Transactions  on  Services  Computing, 2024, 17(6): 4197–4210. doi: 10.1109/TSC.2024.3418330.

CAO Xiaolan, YAN Feng, XIA Weiwei, et al. DDQN based[22] adaptive  multi-channel  MAC  protocol  for  UAV  ad-hoc networks[C].  2024  10th  International  Conference  on Computer  and  Communications,  Chengdu,  China,  2024: 1680–1685. doi: 10.1109/ICCC62609.2024.10942236.

李侍阳：男，硕士生，研究方向为无人机、6G网络、多维资源调度等.

朱晓荣：女，博士，教授，研究方向为6G通信系统、物联网、区块链、网络大数据等.

责任编辑：余　蓉

# Multi-dimensional Resource Joint Optimization Algorithm for UAV Inspection of Collaborative Tasks of Perception and AI

LI Shiyang ZHU Xiaorong

(School of Communication and Information Engineering, Nanjing University of Posts and Telecommunications, Nanjin 210003, China)

## Abstract:

Objective With increasing demand for aerial operations, the capabilities of various aircraft are steadily expanding across all airspace levels and multiple industries. The application of Unmanned Aerial Vehicles (UAVs) now spans multiple altitude layers, from low to high altitudes, and covers micro, medium, and large models. UAVs are widely used in public safety, transportation, emergency management, logistics and distribution, geographic surveying and mapping, and other fields, thereby promoting innovation and transformation in production and daily life. Compared with traditional manual inspection, UAV inspection, as an emerging operational approach, can acquire image information that is difficult for the human eye to capture. Labor costs are therefore significantly reduced, and the accuracy and efficiency of inspection operations are improved. However, UAV inspection also creates new challenges for multidimensional resource allocation and task scheduling. In power system inspection, for example, transmission lines are exposed to outdoor environments for long periods and are vulnerable to corrosion, aging, and even damage. Regular inspections are therefore required to ensure operational safety.

Methods A four-stage multidimensional resource inspection and scheduling collaborative optimization algorithm is proposed. The original optimization problem is decomposed into four subproblems according to the inspection process. After mathematical analysis of each subproblem, a corresponding solution method is proposed. For the node selection problem, a dual-aided Mixed-Integer Linear Programming (MILP) transformation method is used. For the UAV data acquisition problem, a data-driven boundary learning method is adopted. For UAV communication resource allocation, a bandwidth-power joint optimization algorithm based on Successive Convex Approximation (SCA) is used. For node computing power allocation, a lower-bound analytical allocation method is adopted. Finally, the original problem is solved by an alternating optimization method across the subproblems, thereby forming the complete algorithm.

Results and Discussions Simulation results show that the proposed algorithm reduces overall UAV energy consumption compared with the benchmark algorithms. Simulation training is conducted for visual positioning and fault detection services to examine the relationship among compression ratio, data volume, and service performance. Figures 2-5 show that fault detection accuracy reaches its optimum at 60% data volume and 60% compression ratio. Visual positioning accuracy reaches its optimum at 80% data volume and 80% compression ratio. Figure 6 shows that the proposed algorithm achieves higher accuracy than the benchmark algorithms for AI services. As shown in Figures 7 and 8, under varying bandwidth, computing power, and other resource conditions, the proposed algorithm consistently performs better than the benchmark algorithms in terms of energy consumption and effectively reduces total energy consumption.

Conclusions A multidimensional resource joint optimization algorithm is proposed for intelligent UAV inspection with collaborative perception and AI tasks. An optimization problem is formulated with the objective of minimizing UAV energy consumption, using bandwidth, power, computing power, node selection, data volume, and actual compression ratio as variables. The algorithm jointly minimizes UAV energy consumption for two AI services, fault detection and visual localization. Simulation results show that the algorithm reduces total UAV energy consumption and improves model training accuracy. This study focuses on the application scenario of single-UAV inspection. More complex multi-UAV collaborative inspection scenarios can be examined in future work, and additional services can be incorporated for a more comprehensive analysis. Key words: Unmanned Aerial Vehicle(UAV); Intelligent inspection; Resource scheduling; Joint optimization