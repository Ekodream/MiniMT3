# MiniMT3-Piano 项目技术落地与论文交付方案

## 项目判断

综合你们现在的约束，我的判断非常明确：**最稳、最像论文、最能按时交付的路线，不是复现 full MT3，也不是去碰多乐器/多数据集，而是做一个 piano-only、PyTorch 重写、以单主模型为中心的 MiniMT3-Piano 系统**。原因有三层。第一，entity["company","NVIDIA","technology company"] 的 8×4090 从你给的截图看当前都处于高利用率状态，这意味着你们不能把项目建立在“持续占满 8 卡慢慢调”的假设上。第二，Seq2Seq Piano 和 MT3 已经证明了“频谱输入 + 自回归事件输出”的通用 encoder-decoder 路线是成立的；但官方 MT3 开源仓库至今仍基于 T5X，并且 README 明确写了训练“目前不容易支持”，这对一个月内的学生项目非常不友好。第三，MAESTRO v3 已经提供了约 200 小时、带踏板与力度信息、且官方 train/val/test 划分清晰的钢琴数据，足够支撑一个论文级的 piano AMT 项目，不需要再扩线到更复杂的数据混训。citeturn4view0turn2view5turn9view0turn1view1turn3search1

因此，我建议你们把项目**收敛成“一个主模型 + 三个轻创新 + 一个离线一站式 UI”**。这里的三个创新，不要都放在训练里做，而要尽量放到**推理和后处理**上：  
**约束解码**负责减少非法事件；**双模式输出**负责把同一份转录结果同时变成“演奏保真”的 MIDI 和“可读优先”的五线谱；**pedal-aware cleanup**负责在踏板区域修正 offset、抑制伪重复音、提升乐谱可读性。这样做的最大好处是：**你们只需要把一个 checkpoint 训稳**，后面的创新大多不再消耗大训练预算。这个思路，也和钢琴转录文献里“规则约束/后处理非常重要”的经验是对齐的。citeturn1view2turn10view0turn1view1

我建议你们把项目的最低验收线，收紧成下面这三项。一旦这三项都达成，这个项目就已经是“能交、能演示、能写论文”的状态。

| 交付件 | 最低通过标准 | 最终展示形态 |
|---|---|---|
| 主模型 | 能把 10～20 段钢琴音频稳定转成可听 MIDI，且对示例曲目能输出可读五线谱 | checkpoint + 批量推理脚本 |
| 离线系统 | 本地上传一段音频后，能导出 MIDI、MusicXML、PDF/PNG 预览 | 本地 Web UI |
| 小论文 | 至少包含 Abstract、Introduction、Data&Model、Experiments、Results、Future Work & Conclusion、References，并有 1 张核心 Figure、2～3 张 Table | 论文 PDF |

## 系统架构

我建议你们的系统主链固定为：**音频输入 → 16 kHz 单声道重采样 → 128-bin log-mel → 轻量声学编码器 → 自回归 Transformer 解码器 → 约束解码 → 事件流重建 → 双模式渲染 → 五线谱/文件导出**。Seq2Seq Piano 与 MT3 已经证明，把 spectrogram 直接“翻译”为 MIDI-like 事件序列是可行的；而 torchaudio 原生支持高效重采样和 MelSpectrogram，所以你们没有必要在特征提取层再额外造轮子。citeturn4view0turn0search0turn4view2turn4view3

在工程上，**模型只负责“尽量准确地给出事件”**，不要让模型直接承担完整 MusicXML 语义生成。因为真正麻烦的不是“音高和时间值”，而是“怎样把它排成一份人看得舒服的谱子”。这个问题更适合放到后处理和渲染层解决。你们的双模式输出，建议就定义成：  
**Performance Mode**：尽量保留微时值、velocity、pedal，输出 MIDI；  
**Score Mode**：做 pedal-aware cleanup、节拍量化、左右手分配、MusicXML/PDF/SVG 渲染。  
这样既符合“输入乐曲，输出五线谱”的产品目标，也不会大幅增加训练复杂度。citeturn1view1turn14view0turn14view1

前端我建议用 entity["company","Gradio","python ui library"] 做**本地离线 UI**。它本身就是给 Python 函数包一层 Web GUI，默认本地跑在 localhost；如果需要局域网演示，再把 `server_name` 改成 `0.0.0.0`。渲染层建议采用 **music21 + MuseScore CLI + Verovio** 的组合：music21 用来组织 well-formed 的 Score/Part/Measure 结构并导出 MusicXML；MuseScore 命令行负责导出 PDF/PNG，且支持图像分辨率与白边裁切；Verovio 适合做更清晰的 SVG 预览。这条链路非常适合你们“简洁实用、离线优先”的目标，也能直接解决你们之前图里出现的**文字粘连、乐谱发糊、位图显示差**的问题。citeturn12view0turn12view2turn14view0turn14view1turn11view0turn11view1turn5search22

我建议的核心模块配置如下，已经是“能做、能训、能跑 UI”的折中点，而不是追大模型。

| 模块 | 建议配置 | 这样定的原因 |
|---|---|---|
| 音频前处理 | 16 kHz、mono、128 mel、`n_fft=1024`、`hop=160` | 时间分辨率够用，工程简单 |
| 输入切块 | 训练 20 s 随机裁剪；推理 30 s 滑窗、2 s overlap | 长曲可处理，显存可控 |
| 声学编码器 | 2 层卷积下采样 + 8 层小型 Conformer，`d_model=512` | 比纯 Transformer 更稳，参数仍可控 |
| 解码器 | 6 层 Transformer decoder，8 heads，FFN 2048 | 与 MT3/Seq2Seq 家族一致，但规模收敛 |
| 词表 | `NOTE_ON` 88、`NOTE_OFF` 88、`VELOCITY` 32、`TIME_SHIFT` 100、`PEDAL_ON/OFF`、特殊符号 | 专注 piano-only，词表短、训练稳 |
| 约束解码 | 动态 mask + 小 beam（开发期 greedy，最终 beam=4） | 训练不开销，推理收益高 |
| 后处理 | pedal-aware cleanup + quantization + RH/LH split | 创新集中在可见效果上 |
| 输出 | MIDI、MusicXML、PDF、PNG/SVG、调试 JSON | 演示、论文、复现三用途兼顾 |

仓库结构建议一开始就固定，不然后期会非常乱：

```text
MiniMT3-Piano/
  configs/
    model.yaml
    train.yaml
    infer.yaml
    ui.yaml
  data/
    maestro/
    cache/
  src/
    audio/
      preprocess.py
      features.py
    symbolic/
      events.py
      midi_io.py
      cleanup.py
      score_render.py
    model/
      encoder.py
      decoder.py
      loss.py
    decode/
      constraints.py
      beam_search.py
      merge_windows.py
    eval/
      metrics.py
      qualitative.py
  scripts/
    prepare_maestro.py
    train.py
    infer.py
    run_eval.py
  app/
    app.py
  outputs/
    ckpt/
    midi/
    musicxml/
    pdf/
    png/
    logs/
```

你们论文的**核心 Figure**，也建议直接围绕这个结构来画。版式上我建议采用和 RT-1 类似的“**中间主链 + 右侧图例 + 底部输出**”风格，但不要再把整个系统塞得过满。最稳的布局是：最上方放 `Audio waveform` 与 `Log-mel spectrogram`；中间放 `MiniMT3-Piano` 主框，其中上半是 `Audio Encoder`，下半是 `Autoregressive Decoder`；在主框右侧单独拉出一个 `Constraint Decoder` 小框；最下方用虚线框包住 `Performance Mode` 和 `Score Mode` 两条输出路径，最后只展示一小段**矢量化**的清晰乐谱 snippet，而不是整页缩略图。所有文字至少 8.5 pt，线条 1～1.25 pt，导出统一用 PDF/SVG，再贴到论文里。citeturn11view0turn11view1turn5search22turn14view0turn14view1

## 数据与训练

数据上我建议**只用 MAESTRO v3**。它是目前最适合你们任务的单一主数据集：大约 198.7 小时、1276 段钢琴音频与 MIDI 对齐数据，训练集 159.2 小时、验证集 19.4 小时、测试集 20 小时；官方 split 保证同一作品不会同时出现在不同子集；更重要的是，它不仅有 note 和 velocity，还有 sustain / sostenuto / una corda 等踏板信息，这正好支撑你们的 pedal-aware 设计。citeturn1view1

训练目标建议只做**一个 piano-only 事件建模任务**。具体地说，先从 MIDI 里解析出事件序列：`<BOS> -> TIME_SHIFT -> NOTE_ON(p) -> VELOCITY(v) -> NOTE_OFF(p)`，并插入 `PEDAL_ON/OFF`。这里故意不做 program token、不做多乐器、不做复杂文本指令，因为这些都会显著增加建模负担，却不直接提升你们“钢琴音频转五线谱”的主任务价值。你们真正的新意，不要押在更大的词表，而要押在**如何让解码结果更合法、更干净、更适合渲染成谱**。这个取舍和 MT3/Seq2Seq 的主思想是一致的，但范围更可控。citeturn4view0turn0search0turn1view1

训练配置上，我建议第一版就按“**24 GB 单卡可稳跑**”来设计，而不是反过来逼自己只能在 8 卡上才能活。具体建议如下。

| 项目 | 推荐值 | 备注 |
|---|---|---|
| micro-batch | 2 / GPU | 如果 OOM 就降到 1 |
| grad accumulation | 2 | 8 卡时有效 batch=32 |
| optimizer | AdamW | `lr=3e-4`, `wd=1e-2` |
| scheduler | 4k warmup + cosine decay | 足够稳定 |
| precision | 优先 bf16；否则 fp16 + GradScaler | 4090 友好，显存压力小 |
| dropout | 0.1 | 不要太大 |
| grad clip | 1.0 | 防抖 |
| eval interval | 每 2k step | 用 val note F1 选 ckpt |
| checkpoint | 每 1k step | 防意外中断 |
| 早停依据 | `note_with_offsets_f1` 优先，其次 `velocity_f1` | 最贴近最终效果 |

在算力预算上，这个项目其实是可行的。因为 MAESTRO 训练集是 159.2 小时，如果按 **20 s 随机裁剪 + 有效 batch 32** 来粗算，一遍训练集大约就是 **900 step 左右**。也就是说，**30k～50k step** 对一个 piano-only 小模型已经是很充足的预算了；你们完全没必要一上来就追求 100k+ step。再加上 PyTorch 的 DDP 推荐单机一 GPU 一进程，而 AMP 可以用 `autocast` 和 `GradScaler`/bfloat16 降低显存与算力压力，这个规模在你们的一周窗口里是现实的。citeturn1view1turn1view4turn17view1turn17view2

更关键的是，**你们的三个创新里，真正需要再训练的只有“是否把 pedal token 纳入目标”这一层；约束解码和双模式输出几乎都是 inference-time / post-processing 的工作**。这意味着你们最重的训练只要完成**一个主 checkpoint**即可，后续对比实验大多可以通过“同一 checkpoint，不同 decode / cleanup 策略”完成。这是整个项目最重要的风险控制点。citeturn1view2turn10view0turn1view1

结合你们当前服务器状态，我给出的训练执行原则是：**准备期绝不碰现有任务与文件，只在新目录、新环境里做单卡调试；真正拿到保留窗口时，再切到 8 卡主训**。实际操作上，请从第一天开始就保持以下规则：  
一是新建独立环境和独立输出目录，不修改现有工程；  
二是所有调试脚本先保证 1 卡可用，再上 8 卡；  
三是所有配置都做成 yaml，保证 `1 GPU -> 4 GPU -> 8 GPU` 只改 `nproc_per_node`、`batch`、`grad_accum`；  
四是如果正式训练周无法拿满 8 卡，就退到 4 卡，不改代码，只调 batch 和时长。  
这会比“边抢卡边改代码”稳得多。citeturn1view4turn17view1

## 实验设计

实验部分，我建议你们不要贪多，而是做成一套**非常干净、非常像论文的矩阵**：一列是标准音符指标，一列是你们的创新指标，一列是可视化案例。标准指标直接用 mir_eval 做，这样最规范。它的 note matching 默认用 **50 ms onset tolerance**，note-with-offsets 还会要求 offset 误差不超过参考音符时值的 20%，且最小不低于 50 ms；velocity 版本则在音符匹配后再额外检查力度。你们用这套指标，正文里就可以直接说自己遵循了 MIR 里通行的评估方式。citeturn11view5turn11view6

我建议主实验矩阵写成下面这样。它的好处是：**大多数实验不需要重训**，因此非常适合你们现在的时间与算力约束。

| 实验名 | 变量 | 是否需要重训 | 作用 |
|---|---|---|---|
| Baseline Decode | greedy，无约束，无 pedal cleanup | 否 | 给出原始模型能力 |
| + Constrained Decoding | 加动态合法性 mask | 否 | 证明非法事件减少、主指标提升 |
| + Pedal-aware Cleanup | 在前者基础上开启 pedal-aware offset 修正 | 否 | 证明踏板区域与五线谱可读性提升 |
| Score Mode | 在前者基础上做量化、左右手分配、MusicXML 渲染 | 否 | 证明“一站式输出五线谱”成立 |
| No Pedal Token | 删去 pedal token 训练一个轻量对照 | 选做 | 如果时间允许，证明 pedal 建模值不值 |
| Smaller Model | 编码器/解码器各减 2 层 | 选做 | 给出算力-效果折中曲线 |

你们自己的创新指标，我建议至少加两个。第一个是 **Invalid Event Rate**：无约束解码时，统计“非法 `NOTE_OFF`、重复 `NOTE_ON`、错误 `PEDAL_OFF`、非法 token state transition”的比例；加约束后这个值应该接近 0。第二个是 **Pedal-region Offset Error**：只在参考踏板激活区间内统计 offset 误差，专门证明 pedal-aware cleanup 没有白做。再配一个非常简单的人工指标：**Score Readability Rating**，让乐理同学对 20 个样例从节奏可读性、左右手分配、杂乱音符多少这三项打 1～5 分。这样你们的论文就既有“标准指标”，又有“项目创新指标”，还有“人能看懂的结果”。citeturn1view1turn11view5turn11view6

至于和外部方法的比较，我建议把策略分成两层。**正文主表只放你们自己的 baseline 与 ablation**；文献里的 Onsets and Frames、Seq2Seq Piano、MT3、YourMT3+ 放到 Introduction/Related Work 里作为背景，不要把不同数据、不同设定、不同实现的结果硬塞进主表做“伪公平比较”。如果最后时间富余，可以尝试跑公开可用的 MT3 piano checkpoint 或公开 Onsets and Frames checkpoint，在 10～20 个 demo clip 上做一个附表/附录对照；但这不是主线，更不是必须项。citeturn1view2turn4view0turn0search0turn3search1turn9view0

## 论文图表与参考文献

你们的小论文，我建议不要写成“新 SOTA 模型论文”，而要写成**“在受限算力与短周期约束下，实现一个从钢琴音频到五线谱的一体化紧凑系统，并在解码与后处理中提出三个轻量创新”**。这样论文叙事会很顺。Abstract 的核心句式可以是：问题是什么、为什么现有 full MT3 路线不适合你们、你们做了什么、系统可输出什么、在什么指标上有提升、局限是什么。Introduction 则负责讲应用动机与相关工作；Data&Model 讲 MiniMT3-Piano、事件词表、约束解码、双模式输出和 pedal-aware cleanup；Experiments 与 Results 分别负责设置和结果；Future Work & Conclusion 诚实写 meter/key inference、复杂谱面结构和多乐器扩展还没做。citeturn4view0turn0search0turn1view2turn3search1turn1view1

你们的图表清单，我建议最少准备下面这些。只要这些图表做得干净，这篇小论文的“完成度观感”会立刻上来。

| 资产 | 内容 | 关键要求 |
|---|---|---|
| Figure 1 | 整体系统架构图 | 主链清楚、文字不拥挤、输出用矢量乐谱 snippet |
| Figure 2 | 约束解码与 pedal-aware cleanup 示例图 | 用 1～2 个音高轨迹/事件序列例子说明前后差异 |
| Figure 3 | 定性案例图 | 同一片段展示音频、预测 MIDI、最终谱面 |
| Table 1 | 数据集与模型配置 | MAESTRO split、特征、模型大小、训练预算 |
| Table 2 | 主结果表 | Note F1 / Note+Offset F1 / Velocity F1 / Readability |
| Table 3 | 消融表 | 无约束、无 cleanup、有无 pedal token 等 |

你们之前遇到的图发糊、乐谱不清晰，本质上是**把不适合缩放的位图元件塞进了论文大图**。所以论文图有三条硬规则。第一，**乐谱 snippet 一律用 SVG/PDF 导出后再嵌入**，不要截图。第二，**任何公式尺寸标注放在右侧图例里，不要直接叠在模块内部**。第三，**Figure 里的样例只展示 1～2 小节**，绝不展示整页缩略图。借助 music21 生成 MusicXML，再用 MuseScore/Verovio 导出矢量图，是你们做清晰论文图最稳的路径。citeturn14view0turn14view1turn11view0turn11view1turn5search22

你们论文的 References，起始清单至少应该覆盖下面这一组。它足够支撑 Introduction、Related Work、Data&Model 和评估部分。

| 文献/资料 | 在论文中的用途 |
|---|---|
| MAESTRO Dataset | 数据集来源与踏板标注依据 |
| Onsets and Frames | 钢琴 AMT 经典基线 |
| Sequence-to-Sequence Piano Transcription with Transformers | 你们最直接的架构祖先 |
| MT3 | 统一 seq2seq AMT 的上位背景 |
| Revisiting the Onsets and Frames Model with Additive Attention | 论证规则后处理的重要性 |
| YourMT3+ | 说明 MT3 家族后续增强方向，但你们选择收敛范围 |
| mir_eval 文档/论文 | 实验指标来源 |
| PyTorch DDP/AMP、Gradio、music21、MuseScore/Verovio 官方文档 | 工程实现依据 |

这组参考文献与官方文档，已经足够覆盖你们论文里的大部分“该引用什么”的问题。citeturn1view1turn1view2turn4view0turn0search0turn10view0turn3search1turn11view5turn11view6turn1view4turn17view1turn12view0turn14view0turn11view0turn5search22

## 分工与日程

你们两个人的组合其实很适合这个项目，但前提是**分工不要按“谁做模型、谁做点杂活”来分，而要按“谁负责技术闭环、谁负责音乐与论文闭环”来分**。你负责把系统做通；另一位同学负责把系统“变得像一个音乐项目，而不是一堆代码输出”。这个分法比让没有技术基础的同学硬上工程要高效得多。

| 工作包 | 你 | 乐理同学 |
|---|---|---|
| 数据准备与脚本 | 主负责 | 协助检查样例是否可听、可用 |
| 模型实现与训练 | 主负责 | 不参与底层编码 |
| 约束解码与 pedal cleanup 规则 | 主负责实现 | 主负责给规则、阈值和可读性反馈 |
| Score mode 策略 | 主负责工程实现 | 主负责左右手分配、节奏量化、谱面习惯判断 |
| UI 与系统联调 | 主负责 | 主负责测试与交互意见 |
| 实验与指标脚本 | 主负责 | 主负责 qualitative rubric 与打分 |
| 论文写作 | Data&Model、Experiments、Results 初稿 | Abstract、Introduction、案例分析、图表说明初稿 |
| 最终 demo 选曲 | 共同 | 主负责选“适合展示”的曲段 |

下面这张排期表，我是按**26 天**去设计的；如果你们实际只剩 22～24 天，就把相邻两天合并执行，但顺序不要打乱。尤其注意：**训练周前，一定要已经有“单卡能跑通端到端”的脚本**。

| 时间 | 你要完成的事 | 乐理同学要完成的事 | 验收标准 |
|---|---|---|---|
| Day 1 | 冻结 scope，建仓库、环境、目录规范 | 整理 20 个目标 demo 片段清单 | scope 不再扩张 |
| Day 2 | 下载/索引 MAESTRO，写 metadata 读取脚本 | 听样例、标出高踏板/高密度片段 | 数据可遍历 |
| Day 3 | 写 MIDI→事件词表转换，加入 pedal token | 参与定义非法事件与谱面问题清单 | 事件序列可导出 |
| Day 4 | 写 dataloader、20s 裁剪、特征提取 | 检查 10 个样本的事件可读性 | batch 可正常出图 |
| Day 5 | 搭模型 skeleton：encoder/decoder/loss | 学会看推理样例并做记录表 | 前向/反向可跑 |
| Day 6 | 做 10 条样本 overfit sanity check | 记录失败样本与乐理问题 | 小样本可明显过拟合 |
| Day 7 | 写 infer、MIDI 导出、基础评估脚本 | 建 qualitative 评分表 | 单卡端到端可输出 MIDI |
| Day 8 | 单卡 dev run，修 shape、mask、OOM | 试听 dev 输出并写问题列表 | 主链无阻塞 |
| Day 9 | 上一版约束解码初版 | 根据错误样例给规则修订意见 | 非法 token 开始下降 |
| Day 10 | 主训练脚本 torchrun 化 | 不技术介入，整理论文大纲 | 1 卡/8 卡脚本统一 |
| Day 11–13 | 正式 8 卡主训练 | 盯验证样例、做中期质检 | 产出主 checkpoint 候选 |
| Day 14 | 选 best ckpt，跑验证集/测试集批量推理 | 听/看结果，选好与坏案例 | 有稳定主模型 |
| Day 15 | 加 pedal-aware cleanup | 共同确定 cleanup 阈值 | 踏板区样例明显改善 |
| Day 16 | 做 score mode：量化、左右手分配 | 主导节奏/指法/分手规则调整 | 可输出初版谱面 |
| Day 17 | 接 music21/MusicXML/MuseScore/Verovio 渲染 | 检查谱面清晰度与可读性 | PDF/PNG/SVG 可导出 |
| Day 18 | 做离线 UI 骨架 | 试用 UI，提交交互修改意见 | 上传音频→出结果 |
| Day 19 | UI 联调：错误处理、缓存、文件命名 | 选 demo 页面文案 | 离线 demo 稳定 |
| Day 20 | 跑 baseline / +constraint / +cleanup / score mode 四组结果 | 开始对 20 例做 qualitative 评分 | 主结果表原始数据齐全 |
| Day 21 | 整理 ablation 与可视化 | 完成 qualitative 评分 | 图表原材料齐全 |
| Day 22 | 画 Figure 1、Figure 2 | 写 figure caption 初稿 | 核心图完成 |
| Day 23 | 写 Data&Model、Experiments 初稿 | 写 Abstract、Introduction 初稿 | 论文骨架成形 |
| Day 24 | 写 Results、Future Work & Conclusion | 补案例分析与 related work 润色 | 初稿闭环 |
| Day 25 | 全文修订、统一术语、补 references | 统一语言、查错字与图表编号 | 二稿成形 |
| Day 26 | 最终演示彩排、打包代码与论文 | 演示脚本与讲稿准备 | 可以正式提交/答辩 |

你们每天最好固定做一件很小但高收益的流程管理动作：**晚上 15 分钟同步一次“今天新增了什么、明天最危险的点是什么、如果明天卡住就删掉什么”**。对这种短周期学生项目来说，这个动作比再多看两篇论文更有用。

## 风险控制与备选路线

这个项目最怕的，不是模型效果不够 SOTA，而是**范围失控**。所以我建议你们从第一天就明确“不做清单”：**不做多乐器、不做在线服务、不做 full MT3 复现、不做复杂文本指令条件控制、不做端到端 MusicXML generative modeling、不做连续踏板深度回归、不做复杂 meter/key 识别大系统**。你们真正要做的是：**把 piano-only seq2seq 主模型训稳，然后把输出做得更合法、更可读、更像一个完整产品**。citeturn0search0turn4view0turn9view0turn3search1

我建议把主要风险和回退动作预先写清楚，按下表执行。只要按触发信号及时降级，项目就不会崩。

| 风险 | 触发信号 | 立即动作 |
|---|---|---|
| 模型不收敛 | Day 6 还不能 overfit 10 条样本 | 先砍模型规模，再查词表与 target 对齐，不要继续盲训 |
| 8 卡拿不到 | 训练周开始时卡仍被占用 | 退到 4 卡 + grad accumulation，不改主代码 |
| 推理显存爆掉 | 长曲 OOM 或速度太慢 | 固定 30 s 滑窗 + overlap merge |
| 约束解码收益不明显 | 主指标不升 | 继续保留它，因为 invalid event rate 和可读性通常会改善 |
| pedal token 不稳定 | pedal 事件几乎不被预测 | 保留 pedal-aware heuristic fallback，用音符重叠与短时重复规则近似 |
| 谱面不好看 | 节拍/分手异常、乐谱拥挤 | 默认给 4/4，增加“高级选项：3/4、4/4、6/8 覆盖”，必要时退到 MIDI→MuseScore 导谱路线 |
| 论文实验不够丰满 | 只剩一个模型结果 | 重点做 decode/cleanup ablation 与 qualitative，可不勉强补新训练 |

如果严格照这套收敛方案执行，你们实际上只在赌一件事：**主模型能不能在 MAESTRO 上学到足够稳定的 note 级转录**。而一旦这件事成立，约束解码、双模式输出、pedal-aware cleanup、离线 UI、论文图表和大部分实验都能并行推进，不需要再额外烧很多训练成本。对于一个时间不到一个月、团队只有两个人、并且还要照顾现有服务器任务的学生项目来说，这就是风险最低、展示效果最强、也最容易按时交付的路线。