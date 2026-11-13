# ANML 源/目标关系分项诊断

本入口用于拆开源域、目标域 ANML 关系项的作用。它是探索性诊断，
不保证改善准确率，也不是完整的新方法或独立验证实验。

## 1. PyCharm 运行

在工程根目录更新 experiment/anml-pilot 分支后，右键运行：

```text
run_anml_components_pycharm.py
```

入口复用原 run_anml_pycharm.py 顶部的 DATA_ROOT 和 DEVICE。导入这些设置不会启动原脚本。
新的入口默认依次训练 source_relation_only、target_relation_only 两项，
目标被试 1、seed 0、固定 200 epoch、batch size 32。
每一项重新初始化模型和随机种子；不是加载另一项的权重继续训练。
每次启动会生成新的时间戳结果目录，当前不支持续训。

本入口显式调用 train 模式，运行时仍先检查数据。
若只做第一项，可设置 CASES = ["source_relation_only"]。
需要保留一次只改一个实验变量的原则；先使用原 pilot 的数据、epoch、种子和损失参数。

## 2. 对照矩阵

| 设置 | 源域关系 | 目标域关系 | source_scale | target_scale |
|---|---|---|---:|---:|
| 已有 A | 关闭 | 关闭 | 对 A 无影响 | 对 A 无影响 |
| source_relation_only | DANML | 关闭 | 1 | 0 |
| target_relation_only | 关闭 | DANML | 0 | 1 |
| 已有 C | DANML | DANML | 1 | 1 |

这两个新配置的模型仍标记为 variant C；目录及保存的
protocol.diagnostic_case 区分具体组件设置。

这里的 source_relation_only 只表示关闭目标关系损失。
两项都保留共同的 CE、MMD、CMMD、伪标签筛选、网络、采样和调度，
仍是 transductive UDA；不应将其报告为不接触目标数据的 source-only 模型。
配置保留 gamma_pos=-2、gamma_neg=20、radius_pos=0.3、
radius_neg=0.5、margin=0.3、学习率0.001、ss_mix=false。

## 3. 输出位置与归档

```text
outputs/anml_components/source_relation_only/train_时间戳/
outputs/anml_components/target_relation_only/train_时间戳/
```

每个目录中的 config.json 和 environment.json 保存实际设置及环境；
C_subject_1_seed_0 内保存 history.csv、metrics.json、
final_pair_diagnostics.json、predictions.csv 和最终模型。

两个目录的 results.csv 都会写 variant=C，应同时查看父目录和 diagnostic_case。
它们不会自动合并为一个 A/B/C 配对表。可直接打包 outputs/anml_components
进行复核，保留父目录的配置及环境文件。检查是否有多个时间戳运行，避免混用旧输出。

## 4. 本次增加的训练日志

训练目标、优化器和反向传播流程保持原实现。本次只在记录值时增加：

- weighted_cmmd：该 batch 的 CMMD 权重乘 CMMD 损失。
- weighted_relation_source：该 batch 的实际源关系权重乘源关系损失。
- weighted_relation_target：该 batch 的实际目标关系权重乘目标关系损失。
- weighted_source_similarity_sensitivity_l1：该 batch 的源关系权重乘源相似度敏感度。

以上先逐 batch 加权，再汇总为 epoch 均值。
源权重依赖 batch 的 CE，不能用两个 epoch 均值相乘替代这个过程。
原 weighted_target_similarity_sensitivity_l1 已按相同原则记录。

关闭某一关系项时，raw loss/hinge/敏感度仍可作为未加权诊断计算；
实际训练贡献应查看 weighted 字段。相似度敏感度是对相似度矩阵条目的偏导，
不是模型参数梯度归因。最终诊断中的 null 表示比例分母为零，不能解释成没有错误关系。

## 5. 结论范围

这组 A 与 C 的 2×2 关系开关对照，用于区分源项、目标项及二者交互。
若某个单独开启的关系项已使当前折退化，它及其与共同适配框架的作用应被优先检查；
若两项单独表现接近 A 而共同开启退化，应进一步考虑交互和训练稳定性。
单被试、单 seed 的这些观察不能证明普遍因果机制。

保留共同的历史调度：源关系与 CMMD 权重由源 CE 决定；
打印的第87轮起 CMMD 目标置信阈值为1且使用严格大于，CMMD 目标集合为空。
两项使用同一规则，不能把这个阶段的总损失骤降直接当成性能提高。

不要为了匹配目标结果直接放大损失系数、扫描margin或选择目标best epoch。
本轮固定最终epoch，目标真标签只用于模型保存后的评价与事后诊断。
查看目标结果后开展的这些对照属于开发阶段；后续效果验证需要未参与选择的数据。

## 6. 验证状态

已完成源码、配置一致性和接口的静态复核。准备过程未执行项目、未运行训练，
没有新的性能结果。由用户在 PyCharm 手动运行后核对实际输出。
