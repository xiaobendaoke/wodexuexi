# 第三章补证运行包

此目录是独立扩展及发布工具，不是封存源码。协议见 `docs/plans/ch3_evidence_20261103.json`，写作判断见 `docs/plans/ch3_attention_ablation_20261102.md`。

- `prepare.py`：从只读基座复制新发布目录；锚点不匹配即退出，不能修改封存副本。
- `runtime.py`：主实验默认观测及环境物理沿用基座；边界干预、固定输入契约、被动请求记录。
- `historical_gate.py`：服务器重读旧确认的原始回合与摘要，先通过1e-6复现门槛。
- `seed_audit.py`：扫描历史文本和路径中的拟用种子，冲突即停止。
- `remote_tests.py`：默认等价、被动诊断、请求集合、延迟/缺失及输入契约测试，只在服务器运行。
- `controller.py`：预检、40次新训练、200任务确认、固定序列容量对照、请求诊断及432回合边界评估。
- `analysis.py`：完整矩阵核验、以训练种子为单位的配对区间与crossed bootstrap、逐单元代数分解及请求集合诊断。

服务器连接本轮核实为 `ssh -p 22 PengYanghan@100.69.44.85`。旧记录2222端口拒绝连接。专用Python为 `/home/PengYanghan/miniconda3/envs/drone/bin/python`，仅为调用设置 `LD_LIBRARY_PATH=/home/PengYanghan/miniconda3/envs/drone/lib`，不修改共享依赖。新目录 `/home/PengYanghan/Lunwen/ch3-evidence-20261103/`，可编辑发布副本 `release_v1`。首次尝试缺少方法注册的`source`目录保留为失败准备，不执行科研实验。

运行顺序：先 `historical_gate.py` 与 `seed_audit.py`，然后进入发布目录执行 `ch3_controller.py --mode preflight --protocol ... --root ... --validation ...`。预检冻结`release.json`记录代码和协议SHA256及容量宽度；通过后同参数以`--mode run`启动。控制器在每次任务前检查发布摘要。仅已核验完整任务可跳过，未完成目录不自动覆盖。运行记录不等于实验已完成。

结果`status.json`区分阶段、已完成任务数与失败。只有`status=COMPLETE`、全部预定矩阵验证通过后才形成新实证结论。若边界单机请求超过检查点100个下层槽，记录`INTERFACE_NOT_APPLICABLE`；不能静默截断。主比较不通过不自动启动容量训练，不追加随机种子。

所有科学执行在Linux实验服务器；本机只做静态文件核对、写作、编译。不要在Windows导入此目录的训练模块运行实验。不修改原环境物理，不提交或推送Git。

完成后可在服务器运行`summarize.py --root <results> --protocol <protocol.json> --source <release_v1> --output <report.md>`生成结果报告；该入口在status不为COMPLETE时拒绝生成实证报告。它不修改冻结训练发布。

技术修订v2：首次主训练在任何主确认评估开始前统一停止，保留results/validation/release_v1。修正延迟遥测的发送候选与实际接收槽位计数，增加全部12种边界设置的命令行加载短流程。四组不筛选地从头重启，固定种子与检验规则不变。当前发布release_v2，结果results_v2，门槛validation_v2，日志controller_v2.log；v1不形成实证结论。
