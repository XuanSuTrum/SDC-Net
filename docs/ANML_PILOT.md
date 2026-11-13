# SDC-Net × ANML：第一阶段可运行探索分支

本分支用于 A/B/C 对照及机制诊断准备。尚未执行训练，也没有性能结果。
原始基准提交：84e1deeade0c6e1e7993f6d0c00592326d44486e。

## 1. 在 PyCharm 中开始

推荐单独下载本分支，保留已有工程和结果。终端可用：

```bash
git clone --branch experiment/anml-pilot --single-branch https://github.com/XuanSuTrum/SDC-Net.git SDC-Net-ANML
```

在 PyCharm 打开 SDC-Net-ANML 文件夹，选择现有 paper1 Python 解释器（建议 Python 3.9+）。
缺依赖时，在该解释器对应的终端安装：

```bash
python -m pip install -r requirements-anml.txt
```

依赖文件只包含本入口使用的 NumPy、SciPy、PyTorch；GPU 是否可用取决于所安装的 PyTorch 构建。
入口会打印 device；不强制要求 GPU。

打开根目录 **run_anml_pycharm.py**，只需先改 DATA_ROOT 为真实数据目录。
默认 MODE="audit"，右键运行，会检查数据、标签编码、被试编号和试次元数据。
出现 DATA AUDIT PASSED 后，将 MODE 改为 "train"，再次运行。

默认只跑 B、被试 1、种子 0、固定 200 epoch。这里的 1 指文件 1.mat 或 1_日期.mat 的编号，
不再依赖 os.listdir 的顺序，也不是旧代码的零基下标。

```python
MODE = "train"
VARIANTS = ["B"]
SUBJECTS = ["1"]
SEEDS = [0]
EPOCHS = 200
```

确认单折可运行后，可在相同配置下比较：

```python
VARIANTS = ["A", "B", "C"]
SUBJECTS = ["1", "2", "3"]
SEEDS = [0, 1, 2]
```

这只是开发用示例，不代表三个被试足以证明方法有效。全部被试使用 SUBJECTS=["all"]。
每次启动生成独立时间戳目录，不能把两次启动解释为续训；当前版本不支持断点续训。
原论文 main_5.py 不作为本分支新实验的启动入口。

也支持终端参数；优先级为命令行参数 > launcher 顶部设置 > JSON 配置。
命令行只覆盖明确提供的选项，例如只传 --mode train 时会保留你已填写的 DATA_ROOT。
如要直接完全按 JSON 配置运行，可在项目根目录用 python -m experiments.anml.train。

```bash
python run_anml_pycharm.py --mode audit --data-root "E:/your_data/feature_for_net_session1_LDS_de"
python run_anml_pycharm.py --mode train --data-root "E:/your_data/feature_for_net_session1_LDS_de" --variants A B C --subjects 1 2 3 --seeds 0 1 2
```

## 2. 数据格式与 SS-Mix

### 已处理 MAT：默认方式

DATA_ROOT 指向同一个 session 的 15 个被试 MAT 文件。默认读取：

```text
1.mat -> dataset_session1.feature: [N, 310]
         dataset_session1.label:   [N, 1]
2.mat -> ...
15.mat -> ...
```

其他 session 修改 configs/anml_pilot.json 的 data.session。
支持三类 SEED 和四类 SEED-IV（后者 num_classes=4；使用正确的 session 文件夹）。
本入口没有实现 FACED 数据适配。不要把其他格式的文件混入该目录。

label_encoding=auto 在完整类集合存在时区分：SEED -1/0/1、0/1/2、1/2/3。
不完整/歧义标签集合会报错，此时应依据数据真实定义设置 seed_signed、zero_based 或 one_based，
不要通过尝试不同编码来挑选准确率。严格 one-hot 标签也可读。
每个被试单独以其特征做 [-1,1] MinMax；目标完整无标签特征参与归一化与适配。

默认 ss_mix=false。原 processed 导出没有保存试次边界，不能仅凭总行数重建边界。
本分支没有沿用原仓库后定义的 feature*alpha 占位增强。
因此 A/B/C 默认均不增强，B 是修订的关系学习基线，不是完整论文 SS-Mix 结果的复现。

要启用真实的源域同被试同试次 Mixup，设置 experiment.ss_mix=true，并满足下列任一项：

- MAT 结构中提供逐行 trial_id 或 trial_ids；
- 提供来源明确的 trial_metadata JSON，逐被试给出真实 trial_lengths 或 trial_ids；
- 使用下述原始 de_LDS 模式，直接从每个试次字段提取分组。

trial_metadata 文件示意（长度必须替换为从原始数据取得的整数，不能保留占位符）：

```json
{
  "confirmed": true,
  "source": "Trial lengths extracted from the original de_LDS fields",
  "subjects": {"1": {"trial_lengths": [235, 233]}}
}
```

上面的两段长度只是说明 JSON 结构，不是可直接用于完整数据的边界。
各被试长度总和必须等于该文件样本数；每个源试次内标签必须一致。
元数据键使用 audit 中打印的被试 ID。JSON 中的 confirmed 只是来源声明，不替代对原始数据的核对。
启用时为每个源试次生成等量 Mixup 样本并与原始源数据拼接；目标只使用原始无标签样本。

### 原始 de_LDS

将 data_format 设为 raw_de_LDS，DATA_ROOT 指向单 session 的原始特征文件夹。
每个文件需有 de_LDS1...de_LDS15（SEED）或 ...24（SEED-IV），shape=[62,T,5]。
另设置 data.label_file 和 data.label_key，提供对应 session 的 15 或 24 个试次标签向量。
若标签文件为多 session 矩阵，请先提供选定 session 的向量；本入口不猜矩阵的 session 轴。
相对 label_file/trial_metadata 路径以 DATA_ROOT 为基准。

原始特征转换使用 transpose(1,2,0).reshape(T,310)，对应仓库 MATLAB 导出的频带优先、频带内通道顺序。
不支持 MATLAB v7.3/HDF5 MAT，需重新保存成普通 MAT v7。

## 3. A/B/C 改什么

| 设置 | 关系损失 | 共同设置 |
|---|---|---|
| A | 源、目标关系损失均为 0 | 同数据、主干、分类器、MMD/CMMD、采样、epoch 与种子 |
| B | 数值安全的 DSCL BCE | 同一候选关系池，排除自身对角关系 |
| C | DANML Eq. (22) 的独立适配 | 在相同概率余弦空间中替换关系目标 |

主干复用 backbone.py 的 simple_net：310→64→64，ReLU；不启用原 forward 未调用的 Dropout。
分类器为 64→32→C，ReLU 与 Dropout(0.5)。三组均优化主干和分类器。
关系空间保持为 softmax 分类概率的余弦相似度，不悄悄改用 64 维 embedding。

源关系由源真标签产生。目标候选由当前模型的相似度 detach 后产生：
sim>upper 或 sim<lower 的关系被保留；二元关系由 sim>pair_threshold 决定。
初始 pair_threshold=0.9；第 0 个 epoch 结束后为上下阈值中点。
阈值递推使用正确的总 epoch 数，所有变体共用。

B 对所选有向关系取平均；C 对同时具有实际正、负邻居的 anchor 取平均。
C 的虚拟半径不能替代缺失的真实邻居；空候选/无效 anchor 返回连通计算图的零损失。
这两种平均方式与尺度不同，日志同时报告候选覆盖率、有效 anchor 和实际相似度敏感度。

### ANML 实现与初始参数

参考：[Adaptive Neighborhood Metric Learning, VI-A, Eq. (22)](https://arxiv.org/html/2201.08314v1)。
本分支是根据论文公式编写的适配代码，不是作者官方实现。

距离 d=1-cosine(prob_i,prob_j)，范围 [0,1]；正负集合分别加入固定虚拟半径。
b_gamma(a)=-log(mean(exp(-gamma*a)))/gamma；使用稳定 logsumexp 与 |neighbors|+1 归一化。
外层取 relu(margin+b_pos-b_neg)。

初始固定值：gamma_pos=-2、gamma_neg=20、radius_pos=0.3、radius_neg=0.5、margin=0.3。
这些值仅是探索初值，未调优。margin 明确为本次适配设置：目标关系已经由相似度阈值排序，
margin=0 容易使其 hinge 恒为零；同时，虚拟半径会限制小邻域可达到的最大间隔。
以仅1正1负、正距离0/负距离1计算，当前参数的最大聚合间隔约0.362，故预先选用0.3而非0.5。
这只是边界可达性检查，不是性能验证或参数最优性证明。仍需检查 C 的 hinge-active anchors 和敏感度日志；
损失为零不等于关系已正确，也不说明机制有效。

当前没有实现新的关系可靠性估计或先前讨论的 D/E 机制；本阶段只做 A/B/C 与诊断记录。

## 4. 相对原仓库的统一修正

本分支采用独立实验模块，原 main_5.py 等文件保留。共同修正包括：

1. 避免 Transfer_Net 的 num_class/max_iter 位置参数互换。
2. optimizer 包含分类器参数，统一清零其梯度。
3. 区分 MAT 实际标签编码，修复 MATLAB 导出已+1而旧加载器仍按-1/0/1处理的风险。
4. 以文件名编号固定被试映射，不使用目录枚举位置；不调用 os.chdir。
5. 修复目标首 batch 后错误重置迭代器，使用 zip，步数为两个 loader 长度的较小值。
6. CMMD 张量跟随输入设备；目标置信集合为空时返回 0。
7. 所有关系损失排除对角，并处理空候选与数值边界。
8. 固定最后 epoch，保存同一模型后才做目标准确率评估；不选目标 best epoch。
9. 去除占位增强，真实 SS-Mix 需可靠试次分组。
10. 默认 Adam 学习率设为 0.001（旧入口为 0.05），三组相同，配置与输出中明确保存。

这些共同改变意味着 B 不是旧结果的逐数值复现。比较应在本分支同一配置下重新运行 A/B/C。

为减少其他变量，保留原环形 MMD 估计器（可为负，不截断），
保留 cmmd_1 非空情况下按总 batch 大小归一的公式（不额外除以类别数）。
同时保留以下原代码调度，epoch e 从 0 开始：

| 项目 | 规则 |
|---|---|
| MMD 权重 | e≤10:1；10<e≤40:2/(1+exp(-10))-1；40<e≤85:exp(-0.6e)；e>85:0.01 |
| CMMD 伪标签置信阈值 | e<10:0；10≤e<40:0.5；40≤e≤85:0.75；e>85:1，严格大于 |
| CMMD/源关系权重 w | 源 CE≤0.1:1；0.1<CE<0.15:0.5；CE≥0.15:0 |
| 目标关系权重 | 2e/E |

总损失为 CE + beta*MMD + w*CMMD + w*source_scale*R_source + (2e/E)*target_scale*R_target。
因此 e>85 后 CMMD 无高置信目标，修正后为 0；源 CE 较高时源关系权重也是 0。
这些是继承的调度行为，日志会显示实际生效情况，不宣称是最优设计。
三组共享权重规则；依赖各自源 CE 的权重数值可能不同，并非每一步的数值完全相同。

## 5. 输出与机制诊断

每次启动输出到 outputs/anml_pilot/audit_时间戳 或 train_时间戳，不覆盖先前结果。

| 文件 | 内容 |
|---|---|
| config.json / environment.json | 完整配置、配置哈希、代码版本及运行环境 |
| data_audit.json / 各折 manifest.json | 文件→被试映射、标签编码、行数、归一化范围、SS-Mix 状态 |
| 各折 history.csv | 每 epoch 的 batch 均值损失、有效关系数量/比例、阈值、权重与梯度诊断 |
| 各折 final.pt | 固定最后 epoch 的模型、配置与阈值 |
| 各折 metrics.json / predictions.csv | 最终 accuracy、macro-F1、混淆矩阵、逐样本预测 |
| 各折 final_pair_samples.csv | 冻结模型后抽样关系的预测/真实同类关系、选择情况与敏感度 |
| 各折 final_pair_diagnostics.json | 所选关系精度、覆盖率、错误关系敏感度份额 |
| results.csv / paired_comparison.csv | 每折结果以及同被试同种子的 B−A、C−B |
| summary.json / completion.json | 按被试先平均种子的汇总；完成/中断/错误状态 |

CSV 中 accuracy 与 macro_f1 取值 0–1；终端 accuracy 显示为百分数。
summary 的标准差在每被试先平均种子后计算；只有一个被试时为 null，不把窗口当独立重复。

训练阶段仅记录无标签目标统计。优化函数只接收源 loader 和目标 x/index loader。
目标真标签存在评估对象中，用于最后模型保存后的评估与诊断，不进入反向传播、选 epoch 或调参。

final_pair_diagnostics 在冻结模型后用与训练相同的 batch_size 分组，抽样顺序仅由 seed 决定。
不足一个完整 batch 的尾部样本不进入关系诊断；JSON 记录请求、抽取、实际使用和丢弃的样本数。
若可用诊断样本不足一个 batch，相关比例为 null；最终分类指标仍使用完整目标评估集。
wrong_similarity_sensitivity_share 是错误关系占 |dL_relation/dsimilarity_ij| 总量的比例。
它不是参数梯度归因，也不能单独证明因果机制；总敏感度为0时报告 null。
训练期的 counts 字段为每 batch 的均值，不是全数据去重关系计数。

## 6. 实验结论边界

这是 transductive UDA：全部目标特征参与无标签归一化和适配，再用对应真标签最终评估。
本分支未实现 held-out target、online adaptation 或 nested-LOSO 超参数搜索。
当前固定配置可用于探索；若查看目标结果后反复修改方法，这些结果就是开发证据。
方法冻结后，需要新会话/新被试/其他未参与选择的数据进行确认。
不要把本分支的固定最终 epoch 结果与旧代码目标 best epoch 的数值直接比较。

## 7. 已完成与尚未验证

已完成：仓库静态核对、接口与损失公式人工审查、A/B/C 实现、数据校验、日志和 PyCharm 入口。
本次没有可用执行环境，因此没有运行 Python、编译、单元测试或训练，也没有访问你的本地 EEG 数据。
远端文件与提交可核验；实际依赖、数据载入、梯度与训练表现需要在你的环境手动运行验证。
先运行 audit，再运行单折 B，之后比较 A/B/C。

若报错，请保留完整 traceback、data_audit.json（若已生成）和配置；不需要把完整 EEG 数据上传。
若 C 的 valid_anchors/hinge_active_anchors/敏感度长期为0，应先检查关系覆盖，而不是直接报告机制成功。
