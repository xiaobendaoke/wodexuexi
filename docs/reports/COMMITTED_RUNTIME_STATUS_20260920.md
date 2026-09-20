# Git 提交与服务器验收状态

代码提交：`bfb030099096d56d2349df9562a4b06a289537da`。
源码摘要：`eb34d0b0e59a056276c2310a98a1dac05875d1ed19b94cc5699fac0efd6b70d3`。

已将该提交通过 Git bundle 传到实验服务器，并从确切提交创建干净工作目录：
`/home/PengYanghan/Lunwen/committed-repair-bfb0300`。
未推送 GitHub，也未修改服务器原工作目录中的用户改动。

已验收发布目录：`/home/PengYanghan/Lunwen/canonical-releases/20260920-bfb0300`。
验收为 61 项测试、30 个单方法短验证通过。清除 Conda 激活标记和 LD_LIBRARY_PATH 后，统一入口仍可运行；两种 NumPy/PyTorch 导入顺序均通过。依赖检查未发现损坏依赖。

在发布目录使用正式门槛，Joint 完成 1 回合 × 1000 步，PPO 8 epochs，batch size 256，保存检查点，状态 PASS。实际输出 manifest 记录上述 Git 提交及 `git_clean=true`。这是运行验收，不是完整论文结果。

正式入口的计划检查通过，现有 MARL 矩阵仍为 150 次训练、1650 次评估。没有启动全矩阵。

## 使用已修复的运行环境

```bash
ssh 100.69.44.85
cd /home/PengYanghan/Lunwen/canonical-releases/20260920-bfb0300
bash run_remote_python.sh 需要运行的脚本.py 相应参数
```

使用 `run_linear_v2_full.sh` 时会自动调用统一运行环境。旧 v2 目录未被覆盖，不应继续使用旧启动说明。

## 尚未完成的第四章范围

快照评估器和接受显式目标的通用最佳改进/穷举模块已经实现并通过测试。第四章优化目标尚待用户确认，因此尚未启用章节级 MC-ITR、Greedy/Iterative 对照及正式评估矩阵。建议目标为：先最少超时请求，再最小平均归一化时延，最后最小卸载相关 UAV 能耗；采用冻结的全批次最终负载均分模型，不添加 MBS 惩罚。

此状态不能称为“全论文全部实验已准备完成”。本次 Git 操作仅提交修复文件，原有无关暂存区逐项核对保持一致。
