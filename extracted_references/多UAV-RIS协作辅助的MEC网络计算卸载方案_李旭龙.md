doi 10 19665  issn1001-2400 20250903

# 多 UAV-RIS协作辅助的 MEC网络计算卸载方案

李 旭 龙1, 高 子 慧1, 霍 佳 皓1, 皇 甫 伟1,2

( 北京科技大学 计算机与通信工程学院,北京 ;

摘要:基于移动边缘计算技术,车联网通过将海量数据在边缘节点进行实时处理,满足了车辆对超低时延的需求。在移动边缘计算系统中利用无人机搭载可重构智能表面来增强计算卸载效能已成为研究热点。然而,现有研究多集中于单一无人机搭载可重构智能表面辅助的移动边缘计算架构,难以充分反映多无人机搭载可重构智能表面协同作业的复杂性和多样性。所以,提出了一种多无人机搭载可重构智能表面协同辅助的移动边缘计算系统,通过联合优化无人机飞行轨迹和可重构智能表面相移配置,以最小化计算任务卸载时延。针对无人机搭载可重构智能表面间通信受限的挑战,文中引入了一种融合多智能体强化学习与合作博弈论的分布式协作机制。考虑到计算卸载时延依赖于所有无人机搭载可重构智能表面在通信环境优化上的协同工作,这导致团队奖励共享,使得个体贡献难以评估,进而阻碍策略优化。为此,文中采用合作博弈论来公平分配个人奖励,有效解决了信用分配难题,促进了无人机搭载可重构智能表面单元间的高效协同。为验证所提方案的有效性,对不同无人机数量和可重机智能表面反射单元配置下的性能进行了综合评估。实验结果显示,所提优化框架在多种场景下均优于基线方案,实现了计算卸载时延的最小化,充分展示了其在实际应用中的潜力和价值。

关键词:移动边缘计算;无人机;可重构智能表面;多智能体强化学习;合作博弈论中图分类号: 文献标识码：A文章编号：1001-2400(2025)06-0058-12

# ComputationaloffloadingschemeformultipleUAV-RIS collaboration-assistedMECnetworks

1 GAO Zihui1 HUO Jiahou1 HUANG FU Wei1 2

1 SchoolofComputer& CommunicationEngineering UniversityofScienceandTechnologyBeijing Beijing100083 China 2 ShundeInnovationSchool Universit ofScienceandTechnolo Beiin Foshan528399 China

Abstract Basedon MobileEdgeComputing MEC technology Vehicleto Vehicle V2X meetsthe demandforultra-lowlatencyinvehiclesbyprocessingmassiveamountsofdatainrealtimeatedgenodes UtilizingUnmannedAerialVehicle-carriedReconfigurableIntelligentSurfaces UAV-RIS inMECsystems toenhancecom utationaloffloadin erformancehasbecomeahotresearchtoic However mostofthe existingresearchfocusesonasingle UAV-RIS-assisted MEC architecture which willfullyreflectthe complexityanddiversityofmulti-UAV-RISco-operationwithdifficulty Therefore thispaperproposesa multi-UAV-RIScooperative-assisted MECsystemtominimizethecomputationaltaskoffloadingdelayby jointlyoptimizingtheUAVflighttrajectoryandRISphase-shiftconfiguration Toaddressthechallengeof

limitedcommunicationbetweenUAV-RIS this aerintroducesadistributedcollaborationmechanismthat incororatesMulti-AentReinforcementLearnin MARL andcooerative ametheor Considerin that computing the offloading latency relies on the collaborative work of all UAV-RISs on the optimization of the communicationenvironmentwillleadtothesharin ofteam rewards makin itdifficulttoevaluate individualcontributionsandthushinderin theotimizationofstrateies Forthisreason this aeradots thecooperativegametheorytoequitablydistributeindividualrewards whicheffectivelysolvesthecredit distribution roblemand romotesefficientcollaborationamon UAV-RISunits Inordertoverif the effectiveness of the proposed scheme,this paper comprehensively evaluates the performance under different numbers of UAVs and configurations of RIS reflection units.Experimental results show that the proposed otimizationframework outerformsthebaselineschemeinavariet ofscenariosandachievesthe minimization of computational offloading delay,fully demonstrating its potential and value in practical applications.

KeyWords mobileedgecomputing unmannedaerialvehicles reconfigurableintelligentsurfaces multiagentreinforcementlearning cooperativegametheory

## 1 引 言

移动边缘计算(MobileEdeCom utin ,MEC)作为一种新型计算范式,在第5代(FifthGeneration,5G)及未来第6代(SixthGeneration,6G)移动通信网络中扮演着至关重要的角色。通过将计算和存储资源下沉至靠近用户的网络边缘, 不仅有效缓解了核心网络的数据传输压力,避免了潜在的网络拥塞问题,而且大幅度降低了任务执行的端到端延迟,从而满足了自动驾驶等时延敏感型应用场景的需求,为车联网提供了必要的技术支撑,推动智慧交通发展[1]。

然而,在复杂的城市环境中,建筑遮挡等因素常导致物联网( , )设备与边缘服务器之间的直接通信链路受阻,严重制约了计算任务的即时卸载,进而影响了 系统的整体效能[2]。为解决这一问题,将智能超表面( , )技术融入 系统中以辅助通信,成为了一个极具潜力的解决方案。 是一种由大量低成本的被动无源反射元件组成的新型无线通信辅助设备[3]。通过编程控制每一个反射元件, 能够精准调控反射信号的幅度与相位,进而在物理层面对无线传播环境进行主动干预与改善,显著提升信道容量,并有效扩大网络的覆盖范围与可靠性[4]。因此,将 技术集成于 系统中,能够为 设备构建一个更加灵活多变、高效可靠的通信环境,实现计算任务更快速、更稳定的卸载,进而提升系统的综合效能[3]。

关于 系统中 的部署策略,现有研究可大致划分为两大类:地面固定 和空中移动 。对于前者,一系列研究[5-7]深入探讨了在建筑物表面等地面固定位置部署 的策略,旨在解决特定 场景下的通信质量优化问题。然而,虽然地面固定 部署策略在提升通信质量方面展现出潜力,但受限于成本、城市景观规划等因素[8-9],实际部署中难以根据不断变化的网络环境实时调整 的位置[10-11]。

近年来,凭借高移动性、部署灵活、成本低等优势,无人机( , )在通信领域的应用日益受到关注[12]。作为一个灵活、可移动的通信平台, 通过搭载 (即 )可以有效地提高 部署的灵活性,这种融合方式巧妙地结合了 的灵活部署和 的高效反射特性,为系统性能优化提供了新思路[13-14]。 等人针对多 携带 辅助的可重构无线中继系统设计了一种基于交替优化的方法,旨在通过联合调整 轨迹、功率分配、主动与被动波束成形策略,实现加权和速率的最大化[15]。 等人则聚焦于通过联合优化空中 的部署、相移配置、设备发射功率及数据更新时间,以最小化基站接收数据的平均信息年龄[16]。文献[ ]研究了 作为中继和空中基站辅助的车联网通信,提出了基于逐次凸近似的方案,在确保满足最低通信速率要求的同时,最大化车辆数据传输的速率。在文献[ ]和[ ]中,作者针对 辅助 系统中的能量效率最大化问题,分别提出了基于逐次凸逼近的无人机轨迹、 相移和资源分配联合优化方案。

尽管现有研究已验证 通过架构协同发挥空间机动性与电磁波调控优势,能够动态调整无线信道,可显著提升 系统的任务卸载效率与能效[20]。例如,部分工作聚焦于 轨迹优化或 相移设计的单一维度,或通过集中式控制实现多 协同。然而,这些方法通常假设完美通信环境或依赖全局状态信息 未充分考虑实际场景中恶劣的通信环境和部分观测对协作的影响 与此同时 多智能体强化学习( , )在无线协作领域的应用(如多无人机路径规划、资源分配)已展现其处理分布式决策问题的潜力[21],但传统 框架多采用集中式训练—分布式执行模式,依赖预设的全局奖励函数,难以直接适配 协作中个体贡献差异化的需求[22]。因此,构建面向多协作的差异化贡献评估体系,并基于此设计动态适配的个体化奖励反馈机制,是实现高效协作与优化系统性能的核心挑战[23-24]。

基于以上动机,文中研究了多 辅助的 系统,其中多 分布式协作以辅助地面设备与 服务器之间的计算任务卸载。文中的主要创新性和贡献包括以下 个方面: 构建了多分布式协作架构,通过联合优化 飞行轨迹与 相移矩阵,实现了 计算任务卸载时延的最小化; 提出基于合作博弈论的多智能体强化学习算法,通过设计公平的个体奖励分配机制,有效解决了多 协作中的探索效率与利益冲突问题 显著提升了多 间的协同性能 通过多场景仿真实验,从时延、鲁棒性等维度全面验证了所提方案的有效性。实验结果表明,相比现有方法,该方案在不同网络规模和信道条件下均表现出显著性能优势。

## 2 系统模型和优化目标

## 2.1 系统模型

如图 所示,文中考虑了一个多 辅助的 系统,其中在地面上有一个配备高性能边缘服务器和 K 根天线的基站,M 个 设备,空中有 N 个配备 的 ,每个 有 L 个反射单元。定义$K { = } \{ 1 , 2 , \cdots , K \} , \mathcal { U } { = } \{ 1 , 2 , \cdots , M \} , \mathcal { N } { = } \{ 1 , 2 , \cdots , N \}$ 和 { ,,…,L}分别为基站天线、 设备、和 反射单元的数量集合。文中采用离散时间模型,将决策时间划分为 T 个相等的时隙,其集合定义为 $T = \{ 1 , 2 , \cdots , T \} ^ { [ 2 3 ] }$ 。假设在时隙 t, 设备 m 生成计算任务Dt无法独立完成任务,需要将其卸载到基站处理。然而,由于建筑物的遮挡,导致 设备与基站之间的链路被阻断,所以 设备的计算任务数据需要借助 上的 进行反射并卸载至基站,实现在复杂通信环境下的任务高效处理。

<!-- image-->  
图 系统模型

## 2.2 通信模型

文中在边长为 $L _ { \mathrm { m a x } }$ 的正方形目标区域内构造三维笛卡尔坐标系,基站的位置坐标为 $Q _ { B } = \{ 0 , 0 , z _ { B } \}$ ,其中 $\mathcal { Z } _ { B }$ 为天线的高度。假设IoT设备在目标区域内随机移动,第 m 个IoT设备在时隙 t 的坐标为Qm(t)=$\{ x _ { I } ^ { m } \left( t \right) , y _ { I } ^ { m } \left( t \right) , 0 \}$ 。无人 机 可 以 在 目 标 区 域 上 空 飞 行,第 n 个 无 人 机 在 时 隙 t 的 坐 标 为Qn (t)

$\{ x _ { U } ^ { n } ( t ) , y _ { U } ^ { n } ( t ) , z _ { U } ^ { n } ( t ) \}$ 其中无人机飞行的高度 $z _ { U } ^ { n } \left( t \right)$ )的范围为 $Z _ { \mathrm { m i n } }$ 到 $Z _ { \mathrm { m a x } }$ 。为了叙述简单起见,以下内容在无歧义的情况下省略时间坐标 t。

将 设备与 和 与基站间的信道建模为瑞利衰落模型。考虑路径损耗和小尺度衰落,第 m 个 设备与第 n 个 和第 n 个 与基站的信道增益 ${ \pmb { H } } _ { I U } ^ { n , n } \in { \bf C } ^ { 1 \times L }$ 和 $\pmb { H } _ { U B } ^ { n } \in \mathbf { C } ^ { L \times K }$ 分别可以表示为

$$
{ \bf H } _ { \cal U } ^ { m , n } = ( \rho d _ { \cal U } ^ { m , n - { \widehat { \sf \tau } } } ) ^ { 1 / 2 } \Biggl ( ( { \frac { \beta } { 1 { + } \beta } } ) ^ { 1 / 2 } h _ { \cal U } ^ { m , n } + ( { \frac { 1 } { 1 { + } \beta } } ) ^ { 1 / 2 } g _ { \cal U } ^ { m , n } \Biggr ) \qquad ,\tag{1}
$$

$$
{ \bf H } _ { U B } ^ { n } = ( \rho d _ { U B } ^ { n } \bar {  { ~ \sigma ~ } } \bar {  { ~ \sigma ~ } } ) ^ { 1 / 2 } \left( ( \frac { \beta } { 1 + \beta } ) ^ { 1 / 2 } h _ { U B } ^ { n } + ( \frac { 1 } { 1 + \beta } ) ^ { 1 / 2 } g _ { U B } ^ { n } \right) \qquad ,\tag{2}
$$

其中, $, d _ { I U } ^ { m , n } = \Vert Q _ { I } ^ { m } - Q _ { U } ^ { n } \Vert \mathfrak { K } \Vert d _ { U B } ^ { n } = \Vert Q _ { U } ^ { n } - Q _ { B }$ 分别为第m 个 设备与第 n 个 和第 n 个与基站的距离 $, \rho$ 为参考距离 处的功率增益 $, \beta$ 为瑞利因子 ξ 为路径损耗因 $\vec { \mathcal { F } } , g _ { I U } ^ { m , n } { \sim } C N ( 0 , 1 )$ )和 $g _ { U B } ^ { n } \sim$ CN(O,1）为信道的非视距（Non-Light of Sight,NLoS）链路部分, $\pmb { h } _ { I U } ^ { m , n } = \pmb { a } _ { I U } ^ { \mathrm { T } } \left( \varphi _ { I U } ^ { D } \right)$ 和 $\pmb { h } _ { U B } ^ { n } = a _ { U B } ^ { R } ~ ( \varphi _ { U B } ^ { D } ~ ) \pmb { a } _ { U B } ^ { \mathrm { \tiny ~ T } }$ $( \varphi _ { U B } ^ { A } )$ H 为信 道 的 视 距 (LihtofSiht,LoS)链 路 部 分,其 中 ${ \pmb a } _ { I U } ^ { \mathrm { T } } \left( \varphi _ { I U } ^ { D } \right) = \Big [ 1 , \exp { ( - \mathrm { j } 2 \pi \frac { d } { \lambda } \mathrm { c o s } ( \varphi _ { I U } ^ { D } ) ) } , \cdots$ $\exp ( - \mathrm { j } 2 \pi { \frac { d } { \lambda } } ( L - 1 ) \cos ( \varphi _ { I U } ^ { D } ) ) \Big ]$ 为第 n 架无人机的接收阵列响应,其中 d 为天线间距,λ 为载波波长, $\varphi _ { I U } ^ { D } =$ $\arcsin ( { \frac { z _ { U } ^ { n } } { \left\| Q _ { I } ^ { m } - Q _ { U } ^ { n } \right\| } } )$ 是从第m 个 设备到第 n 个 的通信链路的离开角( ,$\mathrm { A o D } ) , a _ { v B } ^ { R } ~ ( \varphi _ { v B } ^ { D } ) = \left[ 1 , \exp { ( - \mathrm { j } 2 \pi \frac { d } { \lambda } \cos ( \varphi _ { v B } ^ { D } ) ) } , \cdots , \exp { ( - \mathrm { j } 2 \pi \frac { d } { \lambda } ( L - 1 ) \cos ( \varphi _ { v B } ^ { D } ) ) } \right] .$ 和 $\pmb { a } _ { U B } ^ { \top } ~ ( \varphi _ { U B } ^ { A } ) = \left\lceil ~ 1 \right\rceil$ $\exp ( - \mathrm { j } 2 \pi { \frac { d } { \lambda } } \mathrm { c o s } \left( { \varphi } _ { U B } ^ { A } \right) ) , \cdots , \mathrm { e x p } ( - \mathrm { j } 2 \pi { \frac { d } { \lambda } } ( K - 1 ) \mathrm { c o s } \left( { \varphi } _ { U B } ^ { A } \right) ) \Big ]$ 分别为第 n 架无人机和基站的反射和接收阵列响应, $\varphi _ { U B } ^ { D } = \operatorname { a r c c o s } \left( \frac { \textstyle ( z _ { U } ^ { n } - z _ { B } ) } { \textstyle \left\| Q _ { U } ^ { n } - Q _ { B } \right\| } \right)$ 和 $\varphi _ { U B } ^ { A } = \frac { \pi } { 2 } - \varphi _ { U B } ^ { A } \left( Q _ { U } ^ { n } , Q _ { B } \right)$ 分别为从第n 个 UAV-RIS到基站的通信链路的离开角和到达角 $( \mathrm { A n g l e \mathrm { - } o f \mathrm { - } A r r i v a l , A o A } )$

定义 $\pmb { \Theta } _ { n } = \mathrm { d i a g } ( \omega _ { n , 1 } \mathbf { e x p } ( \mathrm { j } \ \theta _ { n , 1 } ) , \cdots , \omega _ { n , 1 } \mathbf { e x p } ( \mathrm { j } \ \theta _ { n , L } ) ) \in \pmb { \mathbf { C } } ^ { 1 \times L }$ 为第n 个 的有效相移的反射系数对角矩阵,其中 $\omega _ { n , l } \mathopen { } \mathclose \bgroup \left( t \aftergroup \egroup \right) \in [ 0 , 1 \mathopen { } \mathclose \bgroup \left( t \aftergroup \egroup \right) ]$ ]为振幅反射系数, $\theta _ { n , l } \left( t \right) \in \left[ 0 , 2 \pi \right]$ 为相移角度。假设 $\omega _ { n , l } = 1 , \forall n \in \mathcal { N } , l \in \mathcal { L }$ 。令$x _ { m }$ 和 $w _ { m }$ 分别为第m 个IoT设备的信息序列和基站的接收预编码矩阵,并假设基站接收IoT设备所卸载的计算任务时采用迫零接收预编码技术。

由于用户与基站间的直接信号链路被阻塞,所以可以将基站接收到的第 $m$ 个 设备所发送的信号表示为

$$
y _ { m } = \sum _ { n = 1 } ^ { N } { \bf H } _ { U B } ^ { n } \Theta _ { n } { \cal H } _ { U } ^ { m , n } ( w _ { m } \phi ^ { 1 / 2 } x _ { m } + \sum _ { m ^ { ' } = 1 , m ^ { ' } \ne m } ^ { M } w _ { m ^ { ' } } \phi ^ { 1 / 2 } x _ { m ^ { ' } } ) + n _ { m }\tag{3}
$$

其中, $\boldsymbol { \phi }$ 为传 输 功 率, $n _ { m }$ 为 第 m 个 设 备 处 均 值 为 、方 差 为 $\sigma _ { m }$ 2 的 加 性 复 高 斯 噪 声 (Gaussian Noise,AWGN)。

所以,第 $m$ 个 设备卸载计算任务到基站的时延可以表示为 $T _ { m } = \frac { S _ { m } } { ( W \ln ( 1 + \mathrm { S I N R } _ { m } ) ) }$ ,其中 $S _ { m }$ 为计算任务的大小,W 为 可 用 带 宽, $\mathrm { S I N R } _ { m }$ 为 基 站 接 收 到 信 号 的 信 干 噪 比 (SignaltoInterferenceplusNoise, ),即

$$
\mathrm { S I N R } _ { m } = \frac { \Big | \displaystyle \sum _ { n = 1 } ^ { N } { \cal H } _ { U B } ^ { n } \ \Theta _ { n } { \cal H } _ { I U } ^ { m , n } \ w _ { m } \Big | ^ { 2 } \rlap / p } { \displaystyle \sum _ { m ^ { \prime } = 1 , m ^ { \prime } \ne m } ^ { M } \Big | \displaystyle \sum _ { n = 1 } ^ { N } { \cal H } _ { U B } ^ { n } \ \Theta _ { n } { \cal H } _ { I U } ^ { m ^ { \prime } , n } \ w _ { m ^ { \prime } } \Big | ^ { 2 } \rlap / p + \sigma _ { m } \vphantom { \cal H } _ { I } ^ { 2 } }\tag{4}
$$

由于基站配备高性能服务器,所以忽略基站处理计算任务的时延。此外,对于大多数场景的应用,如指纹识别、人脸识别和虹膜识别等,处理后的计算结果的大小远小于输入数据的大小,因此计算结果从基站返回 设备的时间开销和能量开销也可忽略不计。

## 2.3 问题公式化

针对 辅助的 系统,旨在通过联合优化多个 的飞行轨迹 $\varXi$ 相移矩阵,在满足计算任务时延约束的前提下,最小化所有 设备在整个决策周期内的总任务卸载时延。基于该优化目标,文中的优化问题公式可表示为

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \displaystyle \mathrm { P l } : \operatorname* { m i n } _ { \boldsymbol { q } _ { \boldsymbol { \nu } } } \sum _ { t = 1 } ^ { M } \sum _ { m = 1 } ^ { M } T _ { m } \left( \boldsymbol { t } \right) , } \\ { \mathrm { s . t . } \quad } & { \mathrm { C l } : 0 \leqslant x _ { \boldsymbol { U } } ^ { n } \left( \boldsymbol { t } \right) , y _ { \boldsymbol { U } } ^ { n } \left( \boldsymbol { t } \right) \leqslant L _ { \operatorname* { m a x } } , \forall n \in N , t \in \mathcal { T } } \\ { \quad } & { \mathrm { C 2 } : Z _ { \operatorname* { m a x } } \leqslant z _ { \boldsymbol { U } } ^ { n } \left( \boldsymbol { t } \right) \leqslant Z _ { \operatorname* { m a x } } , \forall n \in N , t \in \mathcal { T } } \\ { \quad } & { \mathrm { C 3 } : \left\| Q _ { \boldsymbol { U } } ^ { n } \left( \boldsymbol { t } \right) - Q _ { \boldsymbol { U } } ^ { n } \left( \boldsymbol { t - 1 } \right) \right\| \leqslant V _ { \operatorname* { m a x } } \quad , } \\ { \quad } & { \mathrm { C 4 } : 0 \leqslant \theta _ { n , i } \left( \boldsymbol { t } \right) \leqslant 2 \pi , \forall n \in N , t \in \mathcal { T } , l \in \mathcal { L } } \\ { \quad } & { \mathrm { C 5 } : T _ { m } \left( \boldsymbol { t } \right) \leqslant L _ { m } \left( \boldsymbol { t } \right) , \forall n \in N , t \in \mathcal { T } } \end{array} \right. , } \end{array}\tag{5}
$$

其中 和 为无人机飞行范围的约束 限定其只能在目标区域上空 $Z _ { \mathrm { m i n } }$ 到 $Z _ { \mathrm { m a x } }$ 高度范围的飞行 为无人机飞行速度的约束,规定其最大飞行速度为 $V _ { \mathrm { m a x } }$ ; 为 的相移角度的约束,要求其相移角度范围为到 $2 \pi ; C 5$ 为计算任务的时延约束,即计算任务必须在其最大时延约束 $L _ { m } \left( t \right)$ )内完成。问题 为一个非凸的多时隙序列决策优化问题,并具备马尔可夫特性和多重约束条件,难以用传统方法解决。此外,由于通信条件恶劣或成本高昂,无人机往往独立运行,无人机之间无法进行信息共享,因此需要基于不完整的观测信息进行分布式决策,这使得优化问题更具挑战性。

## 3 提出的方案

本节首先将优化问题 建模为马尔可夫博弈,随后提出了基于合作博弈论和多智能体强化学习算法的方案。

## 3.1 问题重构

首先,将通过多 协作来最小化计算任务卸载时延的过程(即问题 )建模为马尔可夫博弈,用元组N,O,A,R,P表示,其中 N 为智能体的数量,每个 为一个智能体 ${ \mathfrak { s o } } = \{ o _ { 1 } , \cdots , o _ { N } \}$ 为观测空间, $A = \{ a _ { 1 } , \cdots , a _ { N } \}$ 为动作空间 $R = \{ r _ { 1 } , \cdots , r _ { N } \}$ 为奖励函数 P 为系统的状态转移概率 文中将的观测空间、动作空间以及奖励函数作如下定义。

()观测空间 $o _ { n } \in O$ :定义 $O _ { n }$ 为 n 的观测空间。假设无人机可以利用现有的信道估计方法获取从 设备到无人机、从无人机到基站的完美信道状态信息,则无人机n 的观测信息包括无人机的坐标以及相关的信道状态信息可表示为

$$
o _ { n } { = \{ Q _ { U } ^ { n } ( t ) , H _ { I U } ^ { m , n } ( t ) , H _ { U B } ^ { n } ( t ) \} } \qquad \circ\tag{6}
$$

()动作空间 $a _ { n } \in A$ :定义 $a _ { n }$ 为 n 的动作空间,其中包括无人机的飞行位移和 的相移矩阵,即

$$
a _ { n } = \{ \Delta Q _ { U } ^ { n } ( t ) , \pmb { \mathscr { O } } _ { n } ( t ) \}\tag{}
$$

其中, $, \Delta Q _ { U } ^ { n } \left( t \right) = Q _ { U } ^ { n } \left( t \right) - Q _ { U } ^ { n } \left( t - 1 \right)$

()奖励函数 $r _ { n } \in R$ :定义 $r _ { n }$ 为 n 的奖励函数。奖励是智能体采取行动后环境给予的反馈,用于评估动作的好坏。奖励函数的设计是至关重要的,它需要与优化目标相吻合,以获取更好的优化效果。在文中所有 共享一个团队奖励 $\boldsymbol { r } _ { T }$ ,其被定义为

$$
r _ { n } = r _ { T } = \sum _ { m = 1 } ^ { M } \left\{ \frac { L _ { m } } { T _ { m } } , \quad T _ { m } \leqslant L _ { m } \right. \qquad ,\tag{8}
$$

其中, $, \kappa > 0$ 为计算卸载任务未在一个时隙内处理完成的处罚。

## 3.2 基于合作博弈论和多智能体深度强化学习算法的方案

## 基于合作博弈论的个体奖励

所有无人机共享一个团队奖励函数使得量化每架无人机对团队奖励的具体贡献变得困难。在这种情况下,每架无人机只能基于共享的团队奖励来优化自身策略,这通常会导致“懒惰智能体(无人机)”的出现,即部分无人机通过有效策略显著提升了团队奖励,而其余无人机可能会担心自身探索行为对团队整体奖励产生不利影响 而抑制自身在策略探索与学习上的积极性 进而陷入一种 懒惰 状态 减少了对新策略的探索与学习 因此 为了促进无人机之间的高效协作 避免 懒惰智能体 无人机 的出现 根据每个无人机对团队奖励的贡献,公平分配个人奖励是至关重要的。

为此,文中提出了基于合作博弈论的个体奖励分配方案。具体来说,在每个时隙,将多无人机协作辅助设备计算卸载的过程建模为 个玩家(无人机)的合作博弈,表示为 $G = \{ \mathcal { N } , \mathcal { C } , v \}$ ,其中 为玩家的集合;是 的子集,表示多个玩家组成的联盟; $v ( \mathcal { C } )$ ), 是联盟收益的特征函数[25]。用 值[26]计算参与合作博弈的无人机 n 对联盟 的贡献,表示为

$$
C _ { S } ( \mathcal { N } , n ) = \sum _ { \mathcal { C } = \mathcal { N } \backslash n } \frac { | \mathcal { C } | \dag ( | \mathcal { N } | - | \mathcal { C } | - 1 ) ! } { | \mathcal { N } | \dag } \mathrm { M C } ( \mathcal { C } , n ) \qquad ,\tag{9}
$$

其中, 和 为联盟 和 中无人机的数量, $\operatorname { M C } \left( { \mathcal { C } } , n \right) { = } v ( { \mathcal { C } } ) - v ( { \mathcal { C } } \backslash n )$ )为无人机 n 对联盟的边际效益。直观上, 值是不包含无人机 n 的所有联盟的边际贡献的加权和。类比于团队奖励,可以将无人机联盟 C 的特征函数v()定义为

$$
v ( \mathcal { C } ) = \left\{ \begin{array} { l l } { \displaystyle \sum _ { m = 1 } ^ { M } \frac { L _ { m } } { T _ { m } ( \mathcal { C } ) } , \quad T _ { m } ( \mathcal { C } ) \leqslant L _ { m } } & { \quad , } \\ { \displaystyle \sum _ { m = 1 } ^ { M } - \kappa , \quad T _ { m } ( \mathcal { C } ) > L _ { m } } & { , } \end{array} \right.\tag{10}
$$

其中, $T _ { n } \left( \mathcal { C } \right) = \frac { S _ { m } } { W \ln ( 1 + \mathrm { S I N R } _ { m } \left( \mathcal { C } \right) ) }$ 为无人机联盟 辅助 系统时 设备 m 卸载计算任务的时延,其中

$$
\mathrm { S I N R } _ { m } ( { \mathcal C } ) = \frac { \Big | \displaystyle \sum _ { n \in C } H _ { U B } ^ { n } \displaystyle \Theta _ { n } H _ { U } ^ { m , n } \displaystyle w _ { m } \Big | ^ { 2 } \rlap / p } { \displaystyle \sum _ { m ^ { \prime } = 1 , m ^ { \prime } \neq m } ^ { M } \Big | \displaystyle \sum _ { n \in C } H _ { U B } ^ { n } \displaystyle \Theta _ { n } H _ { U } ^ { m ^ { \epsilon } , n } \displaystyle w _ { m ^ { \prime } } \Big | ^ { 2 } \rlap / p + \sigma _ { m } ^ { 2 } }\tag{11}
$$

最后,将无人机 n 基于合作博弈论的个体奖励定义为无人机 n 对无人机联盟 的 贡献值,即 $r _ { I , n } { = } C _ { S } ( { \mathcal { N } } , n )$ )。

## 基于合作博弈论和 的方案

在本节中提出了基于合作博弈论的个体奖励和多智能体强化学习算法 (即 )来进行无人机轨迹规划和 相移矩阵优化的方案,其流程如算法 所示。

算法1 基于 的无人机轨迹规划和 相移矩阵优化方案。

初始化主网络参数 $\varphi _ { m , 1 } \ldots \varphi _ { m , 2 } \ldots \theta _ { m } { \mathrm { \# H } } \psi _ { m }$ ,目标网络参数 $\varphi ^ { \prime } { } _ { m , 1 }$ 和 $\varphi _ { \ m , 2 } ^ { ' }$ ,经验回放缓冲区 D

② $\mathrm { ) f o r } ~ j = 1 , 2 , \cdots , J ~ \mathrm { d o }$

③ 重置环境,并获取所有无人机的初始观测 $O ^ { \prime } { = } \{ o ^ { \prime } { _ { 1 } } , \cdots , o ^ { \prime } { _ { N } } \}$

④ $\mathrm { f o r } \ t = 1 , 2 , \cdots , T \ \mathrm { d o }$

⑥ 所有无人机 $n \in \mathcal N$ ,策略网络 $\pi _ { \theta _ { n } }$ 根据局部观测on获取并执行动作an

? 基于合作博弈论计算个体奖励 $R _ { I } = \{ r _ { I , 1 } , \cdots , r _ { I , N } \}$ ,获取下一时刻观测 $O ^ { \prime }$

令 $O {  } O ^ { \prime }$ ,并将 $( O , A , R _ { I } , O ^ { \prime } )$ 存储到经验回放缓冲区 D 中

所有无人机 $n \in \mathcal N$ ,从 D 中随机采样一批样本 B,并更新网络参数

主网络参数 $: \varphi _ { n , i } \cdots \nabla J \left( \varphi _ { n , i } \right) , i = 1 , 2 , \theta _ { n } \cdots \nabla J \left( \theta _ { n } \right)$

温度因子: $: \psi _ { n }  \nabla J ( \psi _ { n } )$

目标网络参数: ${ } _ { : \varphi ^ { ' } { } _ { n , i } } = \tau \varphi _ { n , i } + ( 1 - \tau ) \varphi ^ { ' } { } _ { n , i } , i = 1 , 2$

① endfor

Dend for

软演员—批评家( , )是一种随机策略强化学习算法,多智能体 (, )将其扩展到采用集中训练分布式执行框架的多智能体环境。在 算法中,每个智能体主要由批评家网络和策略网络两部分构成,其中批评家网络包括两个主批评家网络 $Q _ { \varphi _ { n , 1 } }$ 和 $Q _ { \varphi _ { n , 2 } }$ ,两个目标批评家网络 $\cdot Q _ { \varphi _ { n , 1 } }$ 和 $Q _ { \varphi _ { n , 2 } }$ ,它们用于预测无人机 n 在观测on时采取动作an后的预期回报;策略网络包括一个主策略网络 $\pi _ { \theta _ { n } }$ ,它以无人机的本地观测信息 $. o _ { n }$ 为输入,获取动作 a 。 $: a _ { n }$

在每个时隙开始时,所有无人机根据策略网络生成动作,其中包括无人机的轨迹规划和 的相移矩阵 在每个时隙结束后 根据计算任务最终的执行情况 利用合作博弈论计算每个无人机相应的个体奖励$r _ { I , n }$ ，以及生成所有移动设备下一时隙的观测 $\boldsymbol { o ^ { \prime } } _ { n }$ ，并将全局观测 $O = \{ o _ { 1 } , \cdots , o _ { N } \}$ 、联合动作 $A = \{ a _ { 1 } , \cdots , a _ { N } \}$ 、基于合作博弈论的个体奖励 $R _ { I } = { \{ r _ { I , 1 } , \cdots , r _ { I , N } \} }$ 以及下一时刻的全局观测 $\mathcal { O } ^ { \prime } = \{ o ^ { \prime } { } _ { 1 } , \cdots , o ^ { \prime } { } _ { N } \}$ 组成的经验元组$( O , A , R _ { I } , O ^ { \prime } )$ 放入经验回放缓冲区 D 随后无人机 n 从经验回放缓冲区 D 中随机抽取小批量的经验元组以更新其网络参数,其中经验元组的个数为 $\mid B \mid$

在 算法中,策略网络在最大化累计折扣奖励的同时也在最大化策略的熵,所以策略网络的损失函数可以表示为

$$
J \left( \theta _ { n } \right) = \frac { 1 } { \left| B \right| } \sum _ { \sigma _ { s } , \alpha _ { n } \in B } \left[ \psi _ { n } \log \pi _ { \theta _ { n } } \left( \widetilde { a } _ { n } \mid o _ { n } \right) - \underset { i = 1 , 2 } { \operatorname* { m i n } } Q _ { \tau } , _ { i } \left( O , a _ { 1 } , \cdots , a _ { N } \right) \right] , \widetilde { a } _ { n } = \pi _ { \theta _ { n } } \left( o _ { n } \right) .\tag{12}
$$

其中, $\psi _ { n }$ 是温度因子,代表策略熵在优化目标中占的比重。

评价网络可以通过最小化 残差来更新参数,即

$$
J \left( \varphi _ { n , i } \right) = \frac { 1 } { \mid B \big \vert } \sum _ { O , a _ { n } \in B } \left[ Q _ { \varphi , i } \left( O , a _ { 1 } , \cdots , a _ { N } \right) - y _ { n } \right] ^ { 2 } , i = 1 , 2 \qquad ,\tag{13}
$$

其中, $y _ { n }$ 为无人机n 的主批评家网络的目标值,可表示为

$$
y _ { n } = \frac { 1 } { \lvert B \rvert } \sum _ { \sigma \in B } \gamma [ \operatorname* { m i n } _ { i = 1 , 2 } Q _ { \sigma _ { n - 1 } } ( O ^ { \prime } , \tilde { a } ^ { \prime } _ { 1 } , \cdots , \tilde { a } ^ { \prime } _ { N } ) - \psi _ { n } \log \pi _ { \sigma _ { n } } ( \tilde { a } ^ { \prime } _ { \textit { n } } | \mathrm {  ~ \sigma ~ } _ { \alpha } ^ { \prime } ) ] + r _ { I , n } , \tilde { a } ^ { \prime } _ { \textit { n } } = \pi _ { \sigma _ { n } } ( \mathrm {  ~ \sigma ~ } _ { \prime } ^ { \prime } )\tag{14}
$$

其中, $\gamma \in$ [0,1] 为折扣系数。

目标批评家网络的更新方式为软更新,即

$$
\varphi ^ { \prime } { } _ { n , i } = \tau \varphi _ { n , i } + ( 1 - \tau ) \varphi ^ { \prime } { } _ { n , i } , i = 1 , 2 \qquad ,\tag{}
$$

其中, $, \tau$ 是软更新参数。此外,文中对温度因子采用自适应的更新方式,可以表示为

$$
J \left( \psi _ { n } \right) = \frac { 1 } { \mid B \big \vert } \\\\sum _ { o _ { n } \in B } \left[ - \psi _ { n } \log \left( \pi _ { \theta _ { n } } \left( \bullet \mid o _ { n } \right) \right) - \psi _ { n } H \right] \qquad ,\tag{16}
$$

其中,H 为目标策略熵。

值得注意的是,尽管奖励信息需要精确的时延反馈,但得益于所提方案采用的集中式训练 分布式执行的框架和策略网络的离线训练特性,在实际部署后无人机的决策无需奖励信息,从而确保方案在现实场景中的有效性。

## 复杂度分析

针对所提出的基于 的方案进行复杂度分析。文中提出的方案的训练过程包括 J 个训练集,每个训练集有 T 个时隙。单个时隙的计算复杂度主要来源于两个核心模块:基于合作博弈论的个体效用分配机制(算法 第 行)和基于梯度反向传播的神经网络参数更新(算法 第 行)。首先,在计算个体奖励阶段,需为全部 N 个无人机求解 值。根据式()的定义,计算无人机的 值需要遍历所有不包含该无人机的所有可能的联盟。从组合数学角度分析,对于 N 个无人机组成的系统,排除特定无人机 n的联盟组合总数为 $2 ^ { N - 1 }$ ,所以基于合作博弈论计算个体奖励的复杂度为 $O ( N 2 ^ { N } )$ )。然后,网络参数更新的复杂度为 $O ( \mid B \mid ( O _ { p } + O _ { C } ) )$ ),其中 $O _ { p }$ 和OC分别为策略网络和评论家网络参数更新的复杂度,其主要取决于神经网络的拓扑结构,包括隐藏层维度、参数矩阵规模等超参数设置。因此,该方案训练过程的复杂度为$O ( J T ( N 2 ^ { N } + \vert B \vert ( O _ { \phi } + O _ { c } ) ) )$ 。

需特别强调的是 尽管 值在理论层面面临指数级增长的计算复杂度挑战 但通过引入基于蒙特卡洛的近似采样方法替代精确遍历策略 可有效将实际计算复杂度约束于多项式时间范围内 值得注意的是,在工程实践场景中,无人机集群规模通常受限于物理约束条件,这使得所提框架在训练阶段展现出良好的算法可扩展性。更为关键的是,在分布式决策执行阶段,各无人机能够依托离线训练获得的策略网络,基于局部观测信息实现动态协作决策。这种分布式架构设计使得系统协作复杂度与集群规模呈现解耦特性,即随着无人机数量的增加,决策复杂度不产生级联增长,从而在执行层面为系统提供了可验证的规模扩展保障。

## 4 仿真实验与分析

## 4.1 仿真参数设置

文中考虑了一个边长为L 的正方形目标区域,其中多个 设备随机分布,无人机在目标区域上空 到 的范围内飞行。每个时隙开始时所有 设备都有一个要执行的计算任务,计算任务数据大小在 到 之间随机生成,最大时延约束在 至 之间随机生成, 设备的数据传输功率为 。文中的仿真实验采用 作为深度学习框架,使用的评价网络和策略网络均为三层全连接层网络,其隐藏层神经元的个数分别为 和 ,学习率分别为 -4和 -5。经验回放缓冲区大小为 5,批更新的大小为 ,软更新参数 τ 为 ,折扣系数 γ 为 。

为了验证所提出方案的有效性和优越性,选取基于 、多智能体深度确定性策略梯度(, )算法的方案以及随机方案作为基线方案,与所提方案相比较不同之处在于,基于 、 算法的方案中的奖励优化策略是基于整个团队奖励进行优化的,随机方案中所有策略均是随机的。此外,文中构建了一种基于个体无人机奖励的 基线方案( )。在该方案中,无人机 n 的个体奖励被定义为其个体特征函数 $\upsilon \left( C _ { n } \right)$ ),其中 $C _ { n }$ 为只包含无人机的联盟。该个体奖励的含义为特定无人机 n 单独为 设备提供服务时所获取的奖励。

## 4.2 实验结果

图 展示了提出的基于 方案和基线方案的平均累计奖励随训练集数的变化趋势。由图 可知,与几种基线方案相比,文中提出的 方案的平均累积奖励的最终收敛结果方面具有明显的优势。这主要得益于提出的方案将每个 建模为 的代理,并基于合作博弈论为每个公平地分配个体奖励 使得 可以获取自身对团队奖励的贡献 并以此来评估自身策略的优劣,从而有效地提高了算法的探索效率,加快了算法的收敛速度。然而,基于 和 算法的方案分别将每个 建模 和 的代理,并根据团队奖励优化其策略,因此无法量化自身对团队奖励的贡献,也无从评估自身策略的优劣,进而导致算法探索效率低。此外,仅基于团队奖励的优化策略通常会导致团队中的一些 获得有效的策略并显著提高团队奖励,而其他策略不太有效的 开始变得懒惰,降低了探索和学习的意愿,因为它们意识到自身的探索可能会对团队奖励产生负面影响。 方案最终收敛后的累计奖励低于 和 方案,这主要归因于在 方案中, 仅根据以最大化个体无人机边际效用为目标优化其策略,使得间无法进行高效协作。当多个 独立调整 相位参数时,可能引发反射信号的相消干涉现,这种非协作相位调控最终导致接收端合成信号的信噪比显著下降。因此,智能体(即 )利用所提出的基于 方案可以进行有效的无人机轨迹规划和 相移决策,从而使 系统获得最低的计算任务卸载时延。

图 展示了所提方案在测试阶段 设备的位置以及无人机的轨迹 图 中显示 架无人机根据所提方案获取的最优策略进行分布式合作的飞行轨迹规划,从中心点出发分别飞向不同的方向,为不同区域的设备提供服务。这说明提出的算法使 架无人机能有效地分布式合作策略。

图 展示了 种算法在测试阶段 设备的平均计算任务卸载时延随时隙数的变化趋势。与图 类似,提出的基于 的方案的性能最优,即在大部分时隙中,采用该方案进行训练的 设备的平均计算任务卸载时延均最低,采用 、 和 方案的训练结果次之,采用随机方案的训练结果最差。这进一步验证了所提方案的有效性和优越性。该方案可以使 学到高效的无人机轨迹规划和 相移决策,从而有效地改善 设备与基站间的通信环境,以辅助 设备计算任务的卸载,降低计算任务卸载时延。

<!-- image-->

图 平均累计奖励随训练集数的变化曲线  
<!-- image-->  
(a)二维

<!-- image-->  
(b)三维  
图 无人机的最优轨迹

图 展示了在 反射单元数量为 、 、 、 、 的情况下,采用文中所提方案进行计算任务卸载的时延随训练集数的变化趋势。可以看出,随着 的反射单元数量的增加,最终收敛后的计算任务卸载时延逐渐降低。这是因为 反射单元的增加使得其通过改变反射信号的幅度和相位来改善通信环境的能力增强,因此在 的辅助下, 设备与基站间的信道容量增加,计算任务卸载时延也随之减小。

<!-- image-->  
图 设备的计算任务卸载时延随时隙数的变化趋势

<!-- image-->  
图 不同 反射单元数量场景下,计算任务卸载时延随训练集数的变化曲线

图 展示了所提方案和 种基线方案下无人机数量对 设备计算任务卸载时延的影响 可以看出种方案的 设备计算任务卸载时延均随无人机数量的增加而逐渐减小 这是因为无人机数量的增加不仅增加了 反射单元总数量,而且可以随无人机更加灵活的部署,使得 通过反射信号改善通信环境的能力进一步得到了增强。此外,对于不同的无人机数量,采用文中提出的方案相比于其他 种基线方案所得到的计算任务卸载时延均最低,而且随着无人机数量的增加这种优势更加明显。这主要是因为随着无人机数量的增加,它们基于团队奖励进行分布式合作的难度也随之增大。而所提方案中合作博弈论的方法将团队奖励公平地分配给每个无人机,使得其能够量化对团队奖励的贡献,从而可以针对性地改进策略,提高了合作效率。

上述仿真实验均是基于信道状态信息( , )完美已知的理想假设下开展,忽略了环境噪声 设备的位置信息偏差等导致的信道估计误差 为验证所提方案在非理想条件下的有效性,图 展示了在非完美 场景中提出的基于 的方案和几种基线方案的平均累计奖励随训练集数的变化曲线。相较于图 所示完美 场景,几种基于强化学习的方案的平均累计奖励均呈现显著下降,这主要是因为 相移优化依赖精确的 信息,基于强化学习的方案,难以对存在估计误差的 进行相移的设计和无人机轨迹的规划,导致计算任务卸载时延增加及累积奖励下降。值得注意的是,信道估计误差未对非优化随机方案产生性能影响,侧面印证了其对环境参数的低敏感性。尽管如此,文中所提方案在非完美 条件下仍保持对基线方案的显著性能优势,充分验证了该方案在信道不确定性环境中的稳健性与优越性。

<!-- image-->  
图 无人机数量对计算任务卸载时延的影响

<!-- image-->  
图 非完美 场景中,平均累计奖励随

## 5 结束语

文中研究了多 分布式协作辅助 系统中的无人机轨迹规划和 相移矩阵联合优化问题。为了最小化物联网设备数据的卸载时延,首先将复杂的优化问题转化为马尔可夫博弈框架下的决策问题,并在 环境中进行求解。同时,为了解决 间的信用分配问题,创新性地提出了一种融合与合作博弈论的解决方案。该方案巧妙利用合作博弈论的理论基础,根据各 对团队奖励的实际贡献公平、合理地为其分配个体奖励,从而激励 间更加紧密和高效的协作。通过一系列仿真实验验证,文中所提方案相较于现有基线方案展现出了显著优势。在不同 UAV和 RIS数量的多种场景下,该方案均能使 学习到更加高效的分布式合作策略,最终实现了物联网设备计算任务卸载时延的显著降低。此外,文中提出的 分布式协同方案显著降低了计算任务的卸载时延,为车联网等超低时延需求场景的实时决策提供关键支撑。

## 参考文献:

尤肖虎 尹浩 邬贺铨 与广域物联网 物联网学报YOU Xiaohu,YINHao,WU Hequan.6G and Wide Area IoT[J].Chinese Journal on Internetof Things,202o,4(1):3-11

张平 牛凯 田辉 等 移动通信技术展望 通信学报ZHANGPin NIU Kai TIAN Hui etal Technolo Prosectof6G Mobile Communications J JournalonCommunications,2019,40(1) :141-148.

[3]EIMOSSALLAMY M A,ZHANG H,SONG L,et al.Reconfigurable Inteligent Surfaces for Wireless Communications : Princiles Challenes andO ortunitiesJ IEEETransactionsonConitiveCommunicationsandNetworkin 2020 6(3):990-1002.

WANGYi DENGYu XU Yaohua etal EnergyEfficiencyOptimizationoftheRIS-UAVCommunicationSystemBased onanImrovedTD3AlorithmJ JournalofXidianUniversit 2025 524 226-234

5 LUIH L LUOJS WANGSL etal EffectiveSecrecyCapacityforRIS-AssistedNOMACommunicationNetworks J IEEE Transactions on Vehicular Technology,2025,74(1):1379-1384.

高建邦 高国旺 一种智能超表面辅助的非视距安全通信方法 西安电子科技大学学报GAOJianbang GAO Guowang ReconfigurableIntelligentSurface-Assisted Non-Line-of-SightSecureCommunicationSchemeJ JournalofXidianUniversit 2023 502 64-70

[7]LIU Y,WANG R,HAN Z,etal.Passive Beamforming for PracticalRIS-Asisted Communication Systems with Non-Ideal Hardware[J].IEEE Transactions on Vehicular Technology,2024,73(11) :17743-17748.

8 LIU X DENGY HANC etal Learning-BasedPrediction RenderingandTransmissionforInteractiveVirtualRealityin RIS-Assisted Terahertz Networks[J].IEEE Journal on Selected Areas in Communications,2022,40(2):710-724.

9 WANGZ WEIY FENGZ etal ResourceManaementandReflectionOtimizationforIntellientReflectin Surface Assisted Multi-Access Edge Computing Using Deep Reinforcement Learning J IEEE Transactions on Wireless Communications,2023,22(2):1175-1186.

[10]MEI H,YANG K,SHENJ,etal.Joint Trajectory-Task-CacheOptimization with Phase-Shift Design of RIS-Assisted UAV for MEC[J].IEEE Wireless Communications Letters,2021,10(7) :1586-1590.

[11] ZHAOJ,ZHU Y,MU X,etal.Simultaneously Transmitingand Reflecting Reconfigurable Inteligent Surface（STAR-RIS AssistedUAVCommunications J IEEEJournalonSelectedAreasinCommunications 2022 40 10 3041-3056

12 MOZAFFARIM SAADW BENNISM etal ATutorialonUAVsforWirelessNetworks A lications Challenes and OenProblemsJ IEEECommunicationsSurves& Tutorials 2019 213 2334-2360

[13] GUO K,WANG C,LI Z,et al. Multiple UAV-Borne IRS-Aided Milimeter Wave Multicast Communications:A Joint Optimization Framework[J].IEEE Communications Letters,2021,25(11) :3674-3678.

14 SHAFIQUET TABASSUM H HOSSAIN E OptimizationofWirelessRelayingwithFlexibleUAV-BorneReflecting Surfaces[J].IEEE Transactions on Communications,2021,69(1) :309-325.

15 GEL ZHANG H WANGJB etal Reconfigurable WirelessRelayingwith Multi-UAV-CarriedIntelligentReflecting Surfaces J IEEETransactionsonVehicularTechnolo 2023 72 4 4932-4947

16 JIANG W AIB LIM etal AveraeAeofInformationMinimizationinAerialIRS-AssistedDataDeliver J IEEE InternetofThingsJournal 2023 10 17 15133-15146

17 WANGP LID ZHANGY etal UAV-AssistedVehicularCommunicationSstemOtimizationwithAerialBaseStation andIntellientReflectin Surface 2025 J OL 2025-07-07 htts doi or 10 1109 TIV 2023 3324385

18 ZHAIZ DAIX DUOB etal Energy-EfficientUAV-MountedRISAssistedMobileEdgeComputing J IEEE Wireless

Communications Letters，2022,11(12) :2507-2511.

19 ZHOU Y MAZ LIU G etal SecureMulti-LayerMECSystemswithUAV-EnabledReconfigurableIntelligentSurface Against Full-Duplex Eavesdropper[J].IEEE Transactions on Communications,2024,72(3):1565-1577.

[20] YUN WJ,PARK S,KIMJ,et al.Cooperative Multiagent Deep Reinforcement Learning forReliable Surveillnce via Autonomous Multi-UAV Control[J].IEEE Transactions on Industrial Informatics,2022,18(10):7086-7096.

21 WANGS SONG X SONG T etal Fairness-AwareComputationOffloadingwithTrajectoryOptimizationandPhase-ShiftDesinin RIS-Assisted Multi-UAV MEC Network J IEEEInternetof ThinsJournal 2024 11 11 20547-20561.

王若男 董琦 基于学习机制的多智能体强化学习综述 工程科学学报WANGRuonan DONG Qi MultiaentGameDecision-Makin MethodBasedontheLearnin Mechanism J ChineseJournalofEnineerin 2024 467 1251-1268

刘娅汐 李旭龙 霍佳皓 等 面向工业场景的无人机时空众包资源分配 工程科学学报LIU Yaxi,LI Xulong,HUO Jiahao,etal.Spatial-Temporal Crowdsourced Resource Allcation for UAVs in IndustrialScenarios[J].Chinese Journal of Engineering,2025,47(1):91-100.

24 LIX QINY HUOJ etal ComputationOffloadingandTrajectoryPlanningofMulti-UAV-EnabledMEC AKnowledge-AssistedMultiagentReinforcementLearningApproach J IEEE TransactionsonVehicularTechnology 2024 73 5 7077-7088.

25 CHENR YANGZ ACooerativeMerin Strate forConnectedandAutomatedVehiclesBasedonGameTheorwith TransferableUtilit J IEEETransactionsonIntellientTransortationSstems 2022 2310 19213-19223

26 XIEH LUIJCS CooperationPreferenceAwareShapleyValue Modeling AlgorithmsandApplications J IEEE ACM TransactionsonNetworking 2023 31 6 2439-2453

(编辑:齐淑娟)