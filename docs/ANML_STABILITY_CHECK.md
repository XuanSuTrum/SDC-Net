# 固定种子的 A/B/仅源域关系复查

这个入口用于已完成数据审计及 seed 0 试跑后的初始化敏感性检查。
保留已有 seed 0，新增 seed 1、2，不按目标准确率选择种子、epoch 或参数。

## 在 PyCharm 中运行

1. 更新 `experiment/anml-pilot` 分支。
2. 确认 `run_anml_pycharm.py` 中的 `DATA_ROOT` 和 `DEVICE` 仍为当前可用设置。
3. 右键项目根目录的 `run_anml_stability_pycharm.py`，选择 Run。

新入口只复用旧入口中的数据路径和设备。它直接使用训练模式；
旧入口的 `MODE`、`VARIANTS`、`SUBJECTS` 和 `SEEDS` 不控制这轮运行。
通常无需修改新入口中的默认设置。

## 默认运行内容

| Case | 训练器中的 variant | 配置文件 | 源域关系项 | 目标域关系项 |
| --- | --- | --- | --- | --- |
| A | A | `configs/anml_pilot.json` | 关闭 | 关闭 |
| B | B | `configs/anml_pilot.json` | DSCL | DSCL |
| source_relation_only | C | `configs/anml_source_relation_only.json` | DANML | 关闭 |

三个 case 都保留共同的分类、MMD/CMMD 框架；“仅源域关系”不表示只使用源域数据。

默认目标受试者为 1，每个 case 分别运行 seed 1 和 seed 2，共 6 次。
每次从新模型开始，训练 200 个 epoch，batch size 为 32。
提交的配置使用学习率 0.001，关闭 SS-Mix；DANML 参数沿用组件诊断配置。
各 case 按顺序运行，不同时占用多份 GPU 训练资源。

请保持这轮配置不变。尤其不能把 B 指向 source_relation_only 的配置，
否则会关闭 B 的目标域 DSCL 项，改变基线定义。
如果本地曾自行修改这两个 JSON，请先核对与分支中的配置一致。

## 输出和提交结果

控制台开头会打印这一批结果的绝对路径：

`outputs/anml_stability/batch_<UTC时间>_<唯一编号>/`

每个 case 在该批目录下单独保存：

- `A/train_<时间编号>/`
- `B/train_<时间编号>/`
- `source_relation_only/train_<时间编号>/`

每个 `train_...` 目录都有实际生效的 `config.json`、`environment.json`、
数据审计、完成状态、`results.csv`、汇总，以及两个 seed 各自的训练记录、
最终检查点、预测和诊断结果。
训练器始终在最终检查点保存后评估目标标签。

source_relation_only 的 CSV 中仍记录 `variant=C`，
必须结合所在 case 目录及保存的配置解读，不能与源域、目标域关系均开启的 C 混用。

完成后压缩整个 `batch_...` 文件夹，保留三个 case 的父目录和配置记录。
每次重新运行都会创建一个新批目录，不会自动续训。
若中途失败，已经完成的结果保留；先查看报错及 `completion.json`，
避免不加判断地重跑已完成的六次实验。

## 这一轮回答的问题

把新增结果与既有 seed 0 按相同 seed 配对，报告全部三次的 Accuracy 和 Macro-F1，
同时报告每个 seed 上 source_relation_only 相对 A、B 的差值。
不要只展示最高结果，也不要把不同 seed 的最好结果拼成一组比较。

这一轮用于检查同一受试者上的初始化敏感性。
它不能替代独立受试者验证，也不能把连续 EEG 窗口当作独立重复实验。
若差值方向反复变化或平均收益消失，应先重新评估该路线；
若结果方向一致，再预先固定其他受试者的验证方案。

当前变更只提供运行入口和输出目录覆盖参数，不改变模型、优化器或损失计算。
