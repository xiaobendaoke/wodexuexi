**SF-HLM运行实现优化：开发比较**

8状态、224分支实例、672次完整费用测量。图逐位核对300次；稀疏logits最大差2.38e-07。

legacy为原实现，fast为公共计算加速且保留原密集网络，sparse再使用合法边/动作稀疏推理及近并列密集回退。G/R8/R16共享同一公共加速。原始分配与物理结果完全相同后才复用原逐请求时延。所有输入矩阵与特征构造计入逐次时间。

| 方法 | 原始成功率/% | 费用后成功率/% | 相对fast_G/pp | 均值/ms | 块p95均值/ms | 匹配次数 | 稀疏回退/调用 | 能耗比fast_G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_G | 18.976 | 18.675 | +0.000 | 11.28 | 13.13 | 0.000 | 0.000 | 1.0000 |
| legacy_L16 | 19.578 | 17.771 | -0.904 | 69.00 | 85.08 | 13.125 | 0.000 | 0.7517 |
| legacy_L8 | 19.578 | 18.373 | -0.301 | 42.04 | 48.53 | 7.125 | 0.000 | 0.7517 |
| legacy_R16 | 19.398 | 17.530 | -1.145 | 53.73 | 59.13 | 16.000 | 0.000 | 0.8378 |
| legacy_R8 | 19.277 | 18.414 | -0.261 | 31.67 | 35.35 | 8.000 | 0.000 | 0.8804 |
| fast_G | 18.976 | 18.675 | +0.000 | 10.03 | 11.25 | 0.000 | 0.000 | 1.0000 |
| fast_L16 | 19.578 | 18.373 | -0.301 | 40.52 | 47.65 | 13.125 | 0.000 | 0.7517 |
| fast_L8 | 19.578 | 18.976 | +0.301 | 26.47 | 29.92 | 7.125 | 0.000 | 0.7517 |
| fast_R16 | 19.398 | 19.096 | +0.422 | 14.42 | 16.36 | 16.000 | 0.000 | 0.8378 |
| fast_R8 | 19.277 | 18.916 | +0.241 | 12.34 | 13.66 | 8.000 | 0.000 | 0.8804 |
| sparse_L16 | 19.578 | 18.373 | -0.301 | 40.12 | 48.74 | 13.125 | 0.000 | 0.7517 |
| sparse_L8 | 19.578 | 18.976 | +0.301 | 26.23 | 29.15 | 7.125 | 0.000 | 0.7517 |

冻结开发闸门：

```json
{
  "candidates": {
    "fast_L16": {
      "practical": false,
      "fee_difference_vs_fast_G": {
        "mean": -0.00301204819277108,
        "ci95": [
          -0.04128374920612071,
          0.03525965282057855
        ],
        "values": [
          0.0,
          -0.00602409638554216
        ],
        "n": 2,
        "positive": 0,
        "exact_signflip_p": 1.0
      },
      "energy_ratio_fast_G": 0.751731477441173,
      "within_time": true,
      "best_fast_nonlearning": "fast_R16",
      "learning_increment": -0.0072289156626506035,
      "learning_increment_target_met": false
    },
    "fast_L8": {
      "practical": false,
      "fee_difference_vs_fast_G": {
        "mean": 0.003012048192771094,
        "ci95": [
          -0.03525965282057871,
          0.0412837492061209
        ],
        "values": [
          0.0,
          0.006024096385542188
        ],
        "n": 2,
        "positive": 1,
        "exact_signflip_p": 1.0
      },
      "energy_ratio_fast_G": 0.751731477441173,
      "within_time": true,
      "best_fast_nonlearning": "fast_R16",
      "learning_increment": -0.0012048192771084432,
      "learning_increment_target_met": false
    },
    "sparse_L16": {
      "practical": false,
      "fee_difference_vs_fast_G": {
        "mean": -0.00301204819277108,
        "ci95": [
          -0.04128374920612071,
          0.03525965282057855
        ],
        "values": [
          0.0,
          -0.00602409638554216
        ],
        "n": 2,
        "positive": 0,
        "exact_signflip_p": 1.0
      },
      "energy_ratio_fast_G": 0.751731477441173,
      "within_time": true,
      "best_fast_nonlearning": "fast_R16",
      "learning_increment": -0.0072289156626506035,
      "learning_increment_target_met": false
    },
    "sparse_L8": {
      "practical": false,
      "fee_difference_vs_fast_G": {
        "mean": 0.003012048192771094,
        "ci95": [
          -0.03525965282057871,
          0.0412837492061209
        ],
        "values": [
          0.0,
          0.006024096385542188
        ],
        "n": 2,
        "positive": 1,
        "exact_signflip_p": 1.0
      },
      "energy_ratio_fast_G": 0.751731477441173,
      "within_time": true,
      "best_fast_nonlearning": "fast_R16",
      "learning_increment": -0.0012048192771084432,
      "learning_increment_target_met": false
    }
  },
  "best_fast_nonlearning": "fast_R16",
  "goal_complete": false,
  "not_independent_confirmation": true
}
```

统计单位为复用历史训练块，报告块级配对t区间与精确符号翻转；不是独立确认。结果是指定状态集合的即时成功率，不是整回合DSR；软件等待未回写环境。cProfile时间未混入正式费用。

本轮完成不等于持续目标完成。后续根据已冻结门槛推进未用负载闭环或下一项成本感知学习实验，保留本轮全部结果，不放宽原门槛。
