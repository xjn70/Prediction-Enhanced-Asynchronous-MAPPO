# 预测增强的异步 MAPPO：多 AUV 长期 VoI 数据采集研究方案

**版本：** v1.1（在 v1.0 基础上修订，改动均已标出）  
**日期：** 2026-11-13  
**基线版本：** v1.0（2026-11-13）

---

## 本版（v1.1）修改说明

**怎么读这些标记。** 正文里凡是改过的地方，都在该处写了【v1.1·类别】。类别有三种：

- **【v1.1·用户确认】**：你在本轮明确定下的三点。
- **【v1.1·技术修正】**：v1.0 里会造成估计偏差、训练不稳或无法实现的写法，必须改。
- **【v1.1·建议】**：我给的默认取值或范围取舍。不同意可以回退，回退办法写在该处。

| # | 修改内容 | 类别 | 所在章节 |
| :-: | :-- | :-: | :-: |
| 1 | 保留数据量项：包价值改为 $w_p[\omega_v+\omega_I e^{-\eta\cdot\text{age}}]$，$\omega_v=0.3,\ \omega_I=0.7$；目标、预测、指标同步修改 | 用户确认 | 1.1, 1.2, 6.1, 6.4, 7.3, 10.4, 12, 14.6 |
| 2 | 到达类型维持三种（S1 固定、S2 分段突变、S3 MMPP），不增加"缓慢变化"；MMPP 作压力测试 | 用户确认 | 3.2, 4.4, 14.1 |
| 3 | 基线新增已有论文方法的适配版：GAAP-style、SPDC-GA-style、LG-DQN-style，并写明适配规则 | 用户确认 | 2.2, 14.5, 附录 A.3 |
| 4 | 遗忘改为"向先验遗忘"，遗忘强度随时间间隔变化 | 技术修正 | 4.2 |
| 5 | 报告增加累计生成计数器 $G_j$，并规定溢出策略 | 技术修正 | 3.3 |
| 6 | 预测窗口计入航行时间；预测价值在训练循环里用闭式，仿真只用于评估 | 技术修正 | 4.3, 5, 6.4 |
| 7 | Actor 和 Critic 加入 AUV 编号 | 技术修正 | 9.1, 9.3, 10.1, 10.5 |
| 8 | 奖励归属：按交付者归属 + 势函数塑形 + 小比例团队项；v1.0 写法保留为消融 | 技术修正 + 建议 | 7.3, 10.4, 14.2 |
| 9 | 第一版动作空间收缩为 Visit + Return，Wait 与 Collect(τ) 作扩展 | 建议 | 8.3 |
| 10 | 第一版单类别数据包 | 建议 | 6.1 |
| 11 | Sink：单通道；排队可观测性；先检验 Sink 利用率 | 建议 | 6.3, 9.1, 9.2, 14.3 |
| 12 | 实验规模：每个 CH 每回合的访问次数要求；稀疏版本 | 建议 | 14.7, 15 |
| 13 | 加入强启发式基线 Belief + ratio-greedy，并明确它是 MAPPO 必须超过的对照 | 建议 | 7.3, 14.2, 14.5 |
| 14 | 相关工作新增 5 项 | 建议 | 2.2 |
| 15 | 10 个待明确事项给出默认取值 | 建议 | 8.5, 9.2, 10.3, 16 |
| 16 | 开发顺序改为 7 周排期，并列出延后项 | 建议 | 15 |
| 17 | 新增附录：玩具实验的预期与风险、未核实项、术语表 | 建议 | 附录 |

**没有改动的部分：** 总技术路线（Gamma-Poisson + Dec-POSMDP + 异步 MAPPO）、CTDE、PPO-Clip 公式、三维直线导航、事件驱动环境和事件类型，均沿用 v1.0。

**本版有几处我没能核实**（GAAP 全称、LG-DQN 内部结构、两篇只读过摘要的文献、势函数塑形在异步情形下的不变性证明），统一列在附录 A.3。

## 1. 研究方案总览

### 1.1 正式确定的技术路线

本研究 **Prediction-Enhanced Asynchronous MAPPO（预测增强的异步多智能体近端策略优化）**

| 模块 | 正式方案 |
| :--: | :-- |
| 网络 | 固定 CH、固定 Sink、多 AUV |
| 数据生成 | 混合模式：S1 固定速率 Poisson、S2 分段突变速率、S3 MMPP（压力测试）【v1.1·用户确认：先做这三种】 |
| CH 报告 | 当前缓存量、缓存数据包生成时间戳、累计生成计数器 $G_j$【v1.1·技术修正】 |
| 速率估计 | Gamma-Poisson 在线贝叶斯估计，向先验遗忘【v1.1·技术修正】 |
| 预测对象 | 数据生成量、未来缓存量、未来 Sink 交付 VoI |
| 通信 | AUV 进入 CH 通信范围后才能获得报告和采集数据 |
| AUV 间通信 | 不依赖持续直接通信，借助 Sink、各节点进行通信，所留下的信息应有时间戳从而进行学习 |
| 决策 | 异步事件触发、宏动作、Semi-MDP / Dec-POSMDP |
| 强化学习 | Asynchronous MAPPO |
| Actor | 参数共享（输入含 AUV 编号）、分散执行【v1.1·技术修正】 |
| Critic | 集中式 Critic（条件于 AUV 编号）【v1.1·技术修正】 |
| 底层导航 | 固定三维直线导航，不涉及 PID 和六自由度控制（待定，后续可升级） |
| Sink | 固定 Sink、有限上传能力、第一版 FCFS |
| 目标 | 最大化任务周期内实际送达 Sink 的累计价值；包价值 = 数据量项（$\omega_v=0.3$）+ 时效项（$\omega_I=0.7$）【v1.1·用户确认】 |

### 1.2 研究问题

在有限任务周期内，CH 持续产生具有不同时间敏感性和信息价值的数据。AUV 受水声通信范围限制，只能通过局部、延迟的 CH 报告了解网络状态。多台 AUV 需要联合决定：

1. 下一步访问哪个 CH；
2. 现在访问还是稍后访问；
3. 到达 CH 后采集多久；
4. 何时返回固定 Sink；
5. 如何减少重复服务和无效航行；
6. 如何考虑 Sink 排队对最终交付价值的影响。

**【v1.1·建议】** 第一版的动作空间只开放第 1、4 项（选 CH、何时返航），第 2、3 项（等待、采集时长）作为扩展，见 8.3。

研究目标是最大化任务周期内**实际上传到 Sink 的数据所兑现的价值**。【v1.1·用户确认】每个数据包的价值由两部分组成：

- **数据量项**（权重 $\omega_v=0.3$）：只要送达就得到，不随时间衰减，对应"多收数据"；
- **时效项**（权重 $\omega_I=0.7$）：随"生成到送达"的时间指数衰减，对应"数据要新"。

时效项权重更大，所以目标仍然偏向新鲜度，但不放弃数据量。$\omega_v=0$ 时退回 v1.0 的纯时效目标。

### 1.3 核心逻辑链

$$
\boxed{
\text{局部 CH 报告}
\rightarrow \text{数据生成预测}
\rightarrow \text{未来缓存预测}
\rightarrow \text{未来交付 VoI 预测}
\rightarrow \text{异步 MAPPO 决策}
\rightarrow \text{Sink 实际交付}
}
$$

预测模块必须改变任务选择、访问时机、采集时长或返航决策；不能只证明预测误差较小，还要证明预测改善了最终决策质量。

## 2. 与相关工作的区别

本研究的重点是：

- 预测 CH 数据生成过程、未来缓存状态和数据交付价值；
- 使用缓存量和数据包生成时间戳作为 CH 报告主体；
- AUV 根据局部、延迟报告维护预测状态；
- 使用固定 Sink，并显式建模上传排队；
- 采用长期多轮、异步事件驱动任务决策；
- 优化实际送达 Sink 的累计价值（数据量项 + 时效项，6.1）。

不将 AUV 状态预测、边界观测簇或区域间任务迁移作为核心创新。

### 2.1 创新性边界

已有研究已涉及数据生成速率、缓存紧迫度、多 AUV 任务分配、VoI 路径规划、等待决策和分层多智能体控制。因此，不应声称以下内容本身是首次提出：

- 首次将数据生成速率用于任务选择；
- 首次进行多 AUV VoI 数据采集；
- 首次将等待动作加入 AUV 决策；
- 首次采用分层任务决策；
- 首次采用异步 MAPPO。

应突出本研究的组合问题：**稀疏局部报告下的数据状态预测、面向固定 Sink 实际交付的预测 VoI，以及预测增强的异步 MAPPO 长期决策。**

### 2.2 相关研究阅读方向

- **Async-MAPPO / ACE，AAMAS 2023**：借鉴异步宏动作、独立轨迹缓存和集中训练思路。
- **Event-Driven Multi-Agent Decision Processes**：借鉴事件驱动多智能体决策、不同持续时间宏动作和优势估计处理。
- **SPDC**：作为 AUV 状态预测、区域任务调整和移动 Sink 机制的对照。
- **多 AUV 数据生成速率与缓存紧迫度研究**：用于构建 $\lambda_j Q_j$ 类预测/启发式基线。
- **多 AUV VoI 路径规划研究**：用于区分已有 VoI 路径规划与本文的 Sink 排队、数据包级交付价值建模。

**【v1.1·建议】新增 5 项。** 带 ★ 的只读过摘要或网页抽取，写相关工作前必须精读，并核对作者、年份和系统假设（见附录 A.3）：

- **Han 等 2026（IEEE TCCN，Hierarchical MARL for Networked Multi-AUV Data Collection in UWSNs）**：上层 QMIX、下层 MADDPG，AUV 之间机会式转交；VoI 用带下限的指数衰减形式；CH 一次性上报，连续数据到达被列为后续工作。与本文的区别：本文处理连续生成、多趟往返和 Sink 排队。
- ★ **Xu 等 2025（IEEE TVT，Encoder-Decoder Based DRL for Multi-AUV Assisted Data Collection in UWSNs）**：多 AUV VoI 数据采集。它的对比基准包括 GAAP、LG-DQN、SPDC-GA（见 14.5）。据网页抽取结果，它不建模数据生成速率与缓存，每个簇只有一个生成时刻。
- ★ **Wang 等 2024（IEEE IoT-J，A Multi-AUV Collaborative Ocean Data Collection Method Based on LG-DQN and Data Value）**：LG-DQN 为 local global deep Q learning，用于多 AUV 路径规划。
- ★ **Le 等 2026（arXiv:2601.10849）**：通信受限下无人机数据采集的异步 Dec-POSMDP。与"异步 + Dec-POSMDP + 数据采集"直接相关，论文里必须说清区别。
- ★ **arXiv:2410.11282**：基于离线强化学习的多 AUV IoUT 数据采集，含积压量与指数 VoI，缓存状态经基站得知。

## 3. 系统模型

### 3.1 网络组成

CH 集合：

$$
\mathcal{J}=\{1,2,\ldots,N\}
$$

AUV 集合：

$$
\mathcal{A}=\{1,2,\ldots,M\}
$$

固定 Sink 位置：

$$
\mathbf{s}=(x_s,y_s,z_s)
$$

CH $j$ 的三维位置：

$$
\mathbf{c}_j=(x_j,y_j,z_j)
$$

CH $j$ 的通信半径和缓存容量分别为 $\left(R_j,B_j\right)$。

AUV $i$ 的位置、速度、缓存容量和当前存储量分别为：

$$
\mathbf{x}_i(t),\quad v_i,\quad C_i,\quad S_i(t)
$$

任务周期：

$$
\mathcal{T}=\left[0,T_{\max}\right]
$$

第一版假设：

- CH 位置固定且已知，Sink 固定；
- 多台 AUV 执行长期多轮采集；
- CH 持续产生数据，CH 和 AUV 均有有限缓存；
- AUV 进入 CH 通信范围后才能获取报告并采集；
- AUV 不依赖持续直接通信；
- AUV 采用固定速度三维直线导航；
- 暂不将能耗作为主要优化目标，但保留存储、通信范围和任务时间约束；
- 任务周期结束时，只有成功上传到 Sink 的数据计入收益。

### 3.2 CH 数据生成模型

基础 Poisson 模型：

$$
N_j(t,t+\Delta) \sim \operatorname{Poisson}\left(\lambda_j \Delta\right)
$$

其中 $\lambda_j$ 是未知数据生成速率。

混合模式可扩展为两状态 Markov-Modulated Poisson Process（MMPP）：

$$
z_j(t) \in \{0,1\}
$$

其中 $z_j=0$ 表示正常生成，$z_j=1$ 表示突发生成：

$$
\lambda_j(t) = \lambda_j^{(z_j(t))}
$$

状态转移矩阵：

$$
P_j=
\begin{bmatrix}
p_{00} & p_{01} \\
p_{10} & p_{11}
\end{bmatrix}
$$

条件生成过程：

$$
N_j(t,t+\Delta) \mid z_j
\sim \operatorname{Poisson}\left(\lambda_j^{(z_j)} \Delta\right)
$$

实现顺序：先实现固定速率 Poisson 和分段变化速率，再实现隐状态 MMPP。若生成模式对 AUV 不可见，需要估计模式概率，不能假设 AUV 直接知道真实模式。

**【v1.1·用户确认】到达类型先做三种，不增加"缓慢变化"：**

| 场景 | 速率怎么变 | 默认设置（时间单位是任意的，需按真实量级标定） |
| :-: | :-- | :-- |
| S1 固定 | $\lambda_j$ 不变，各 CH 不同 | 各 CH 的 $\lambda_j$ 独立取自 Gamma(2, 10)，均值 0.2 |
| S2 分段突变 | 每隔一段随机时长，$\lambda_j$ 重新抽取 | 段长 = 指数分布（均值 $T_{\max}/3$）加最短 $T_{\max}/20$；新速率仍取自 Gamma(2, 10) |
| S3 MMPP | 两状态，正常与突发切换 | 突发速率约为正常的 4–5 倍；突发时间占比约 10%–20%（待标定） |

**【v1.1·建议】S3 只作压力测试。** 第一版不做隐状态滤波，估计器仍用 Gamma-Poisson（向先验遗忘），并如实报告突发期预测区间覆盖率的下降（Poisson 类估计器在突发下会过度自信）。隐状态滤波放在 4.4，作为可选扩展。回退办法：若要把 S3 当主线，需恢复 14.1 里的"MMPP + 条件 Gamma-Poisson"对比，并增加开发时间。

### 3.3 CH 报告形式与可辨识性

用户已确定的主要报告形式：

$$
\mathcal{R}_j(t_r) = \left\{ Q_j(t_r),\ \mathcal{T}_j(t_r) \right\}
$$

其中 $Q_j(t_r)$ 是报告时刻的缓存量，$\mathcal{T}_j(t_r)$ 是当前缓存中各数据包的生成时间戳集合，$t_r$ 是报告生成时间。

AUV 记录报告年龄：

$$
\delta_j(t) = t - t_r
$$

报告越旧，预测不确定性越大。

**可辨识性提醒：**

仅有当前缓存量和当前缓存中数据包的时间戳，不一定能准确恢复两次报告之间的总生成量。期间可能发生 AUV 采集和缓存溢出，离开缓存的数据包不会出现在后续时间戳集合中。

**【v1.1·技术修正】采用累计生成计数器，并规定溢出策略。**

1. **加 $G_j$。** 报告里再加一个整数：$G_j(t_r)$ 是从任务开始到 $t_r$，CH $j$ 生成的数据包总数（包括因缓存满而被丢弃的包）。两次报告之间新生成的包数就是 $n_j=G_j(t_r)-G_j(t_r^{\text{prev}})$，不受 AUV 采集和溢出影响，速率更新是精确的（见 4.2）。代价只是每个 CH 多存一个整数。
2. **规定溢出策略。** 缓存满时默认**丢弃新到的包**（tail-drop），被丢弃的包价值为 0，并计入累计溢出数 $O_j$（可选字段）。"丢旧保新"作为敏感性实验。这个选择决定了时间戳集合的含义，必须在环境里写死。
3. **采集顺序。** AUV 容量不足以取走全部缓存时，按时间戳先进先出（先取最老的包），剩下的包保留原时间戳。预测模块直接读取报告里的真实缓存，不假设"访问后缓存为空"。
4. **回退办法。** 若不想加 $G_j$：用缓存中最早时间戳 $t_1$ 作为曝露起点，更新为 $a\leftarrow a+n-1,\ b\leftarrow b+(t_r-t_1)$，其中 $n$ 是当前缓存包数。这比直接用 $n$ 和报告间隔的偏差小，但发生溢出或被其他 AUV 采集时仍有混杂，必须在仿真里评估偏差。
5. **可选字段（黑板）。** CH 还可记录"最近一次被服务的时间与 AUV 编号"，后来的 AUV 读到后能知道队友最近去过哪里。第一版不做，作为信息共享扩展。

## 4. 数据生成速率预测：Gamma-Poisson

### 4.1 Gamma 先验

采用 shape-rate 参数形式：

$$
\lambda_j \sim \operatorname{Gamma}\left(a_j, b_j\right)
$$

$$
\mathbb{E}\left[\lambda_j\right]=\frac{a_j}{b_j},
\qquad
\operatorname{Var}\left(\lambda_j\right)=\frac{a_j}{b_j^{2}}
$$

### 4.2 后验更新

若长度为 $\Delta t$ 的观测窗口内观测到 $n_j$ 个新生成包：

$$
\lambda_j \mid n_j
\sim \operatorname{Gamma}\left(a_j+n_j,\ b_j+\Delta t\right)
$$

后验均值：

$$
\hat{\lambda}_j = \frac{a_j+n_j}{b_j+\Delta t}
$$

适应非平稳速率时需要遗忘旧信息。

**【v1.1·技术修正】v1.0 的写法 $a\leftarrow\rho a+n,\ b\leftarrow\rho b+\Delta t$ 是"向 0 遗忘"。** 久未访问时 $a,b$ 同时缩向 0，后验变成几乎没有信息的宽分布，预测会对"久未访问的 CH 一个新包都没有"给出过高的概率；而且 $\rho$ 是每次更新固定的常数，访问间隔长短不同的 CH 被遗忘的程度却一样。在玩具实验（附录 A.1 (c)）中，后验取 Gamma(5,30)、窗口 100 时，真实的“缓存为空”概率约 $6\times10^{-8}$；向 0 遗忘给出的概率，随遗忘强度 $\nu=0.01,\,0.02,\,0.05$ 分别是 1.4%、11%、81%；改成向先验后分别约 0.27%、0.52%、0.81%。

改为**向先验遗忘**，遗忘强度由距上次更新的时间间隔 $w$ 决定：

$$
\rho(w)=e^{-\nu w},\qquad
a_e=\rho\,a^{\mathrm{old}}+(1-\rho)\,a_0,\qquad
b_e=\rho\,b^{\mathrm{old}}+(1-\rho)\,b_0
$$

$$
\boxed{\;a^{\mathrm{new}}=a_e+n_j,\qquad b^{\mathrm{new}}=b_e+w\;}
$$

其中 $(a_0,b_0)$ 是全体 CH 共用的先验（玩具实验用 $a_0=2,\ b_0=10$，实际用校准数据确定）；$w=t_r-t_r^{\mathrm{prev}}$ 是两次报告的间隔；$n_j$ 是这段时间新生成的包数，用 $G_j$ 之差得到（见 3.3）。$\nu=0$ 时就是不遗忘的静态估计，$\nu$ 越大遗忘越快。默认 $\nu=3/T_{\max}$，在 S2 上调参，S1 上应基本不受影响。

### 4.3 未来生成量预测

未来 $h$ 时间内的数据包数量服从 Gamma-Poisson 混合预测分布，即负二项分布：

$$
\boxed{
\hat{N}_j(h)=\frac{a_j}{b_j}h
}
$$

$$
\boxed{
\operatorname{Var}\left[N_j(h)\right]
=
\frac{a_j}{b_j}h
+
\frac{a_j}{b_j^{2}}h^{2}
}
$$

**【v1.1·技术修正】$h$ 的取法。** 预测 AUV 到达 CH $j$ 时的缓存，窗口应取

$$
h_{ij}=\delta_j(t)+T_{ij}^{\mathrm{arr}}
$$

即"报告年龄 + 航行时间"。只用报告年龄会少算 AUV 在路上这段时间新生成的数据。

若平均包长为 $\bar{b}_j$：

$$
\hat{A}_j(h)=\bar{b}_j \hat{N}_j(h)
$$

预测器同时输出均值和方差／预测区间。

### 4.4 混合模式的处理

对于 MMPP，Gamma-Poisson 可作为条件于生成模式的速率估计器。若模式对 AUV 隐藏，需要额外使用 HMM 滤波、粒子滤波或其他状态估计方法维护：

$$
P\left(z_j(t)=0 \mid \mathcal{H}_i\right),
\quad
P\left(z_j(t)=1 \mid \mathcal{H}_i\right)
$$

再将各模式条件下的速率后验加权。不能把两个模式的 Gamma 后验直接当作完整 MMPP 解。

**【v1.1·建议】第一版不实现本节的隐状态滤波**（S3 只作压力测试，见 3.2）。若时间允许再作扩展：速率是一维时，在网格上做精确贝叶斯滤波的效果与粒子滤波相当，而且没有采样抖动，可以优先于粒子滤波。

## 5. 未来缓存预测

CH 缓存容量为 $B_j$。令 $A_j$ 为新生成数据量，$X_j$ 为预测窗口内被其他 AUV 采集的数据量：

$$
Q_j(t_r+h)
=
\min\left\{
B_j,\
\left[Q_j(t_r)-X_j\right]^{+}+A_j(h)
\right\}
$$

其中 $\left[x\right]^{+}=\max(0,x)$。

溢出量：

$$
O_j(h)=
\left[Q_j(t_r)-X_j+A_j(h)-B_j\right]^{+}
$$

AUV $i$ 到达 CH $j$ 通信范围的预计时间：

$$
T_{ij}^{\mathrm{arr}}
=
\frac{\left[\left\|\mathbf{x}_i(t)-\mathbf{c}_j\right\|-R_j\right]^{+}}{v_i}
$$

预计到达缓存量：

$$
\boxed{
\hat{Q}_{ij}^{\mathrm{arr}}
=
\mathbb{E}
\left[
\min\left\{
B_j,\
\left[Q_j(t_r)-\tilde{X}_j\right]^{+}
+\tilde{A}_j\left(T_{ij}^{\mathrm{arr}}\right)
\right\}
\right]
}
$$

预测输出：

$$
\hat{Q}_{ij}^{\mathrm{arr}},
\quad
\sigma_{Q_j},
\quad
P\left(O_j>0\right)
$$

第一版可先假设没有其他 AUV 在途服务该 CH；加入预约机制后，再根据服务状态修正 $\tilde{X}_j$。

**【v1.1·技术修正】计算方式。** 截断 $\min\{B_j,\cdot\}$ 的期望和溢出概率不需要采样：$Q_j(t_r)+A_j(h)$ 服从平移后的负二项分布，对 $0,1,\dots,B_j$ 的概率质量逐项求和即可，每次最多 $B_j+1$ 项。$B_j$ 远大于预期数据量时，截断可以忽略。

## 6. 面向 Sink 实际交付的 VoI

### 6.1 数据包价值

数据包 $p$ 的生成时间、基础价值、类别和衰减系数分别为

$$
\left(\tau_p,\ w_p,\ k(p),\ \eta_{k(p)}\right)
$$

**【v1.1·用户确认】价值 = 数据量项 + 时效项。** v1.0 只有时效项，这里恢复数据量项。完成 Sink 上传时：

$$
\boxed{
V_p\left(t_p^{d}\right)
=
w_p\left[\omega_v+\omega_I\, e^{-\eta_{k(p)}\left(t_p^{d}-\tau_p\right)}\right]
\mathbf{1}\left(t_p^{d}\leq T_{\max}\right),
\qquad \omega_v+\omega_I=1
}
$$

默认 $\omega_v=0.3,\ \omega_I=0.7$（时效项权重更大）。

- $t_p^{d}$ 是数据包真正完成 Sink 接收的时刻。任务周期内未完成上传的数据不计入最终收益。
- 刚生成就送达时，价值是 $w_p$；随着"生成到送达"的时间变长，价值衰减，但最低保留 $\omega_v w_p$。
- 当 $w_p=1$ 时，总价值可以拆成两部分：$J=\omega_v\cdot N_{\mathrm{del}}+\omega_I\cdot V_{\mathrm{I}}$，其中 $N_{\mathrm{del}}$ 是送达包数，$V_{\mathrm{I}}=\sum_p e^{-\eta(t_p^{d}-\tau_p)}$ 是时效部分。两项都以"等效包数"为单位，可以直接相加，不需要再归一化。
- 特例：$\omega_v=0$ 就是 v1.0 的纯时效目标；$\omega_I=0$ 就是纯吞吐量目标。
- 包长不等时，数据量项改成 $\omega_v\,w_p\,l_p/\bar{l}$（$l_p$ 是包长，$\bar{l}$ 是平均包长）。第一版包等长，不需要。

**衰减系数的标定方法。** 设"经过 $T_d$ 时间后，包价值应降到初值的 $r$ 倍"（要求 $r>\omega_v$），则

$$
\eta=-\frac{\ln\!\big((r-\omega_v)/\omega_I\big)}{T_d}
$$

例：$\omega_v=0.3,\ T_d=300,\ r=0.5$，得 $\eta\approx0.0042$（即衰减时间常数约 239.5）。$T_d$ 和 $r$ 要按真实任务的时间尺度来取。

**【v1.1·建议】第一版单类别。** $k=1$，$w_p=1$，包等长，此时数据量项就是送达包数。引入多类别后，各类 $\eta_k$ 不同，携带数据的价值要按类别分别记录（见 6.4 的携带状态）。回退办法：若需要多类别，把 6.4 的携带状态扩成每类一对 $(V^{\mathrm{base}},V^{\mathrm{dec}})$ 即可。

**【v1.1·建议】对 $\omega_v$ 做敏感性实验。** 取 $\omega_v\in\{0,\,0.3,\,0.5,\,1\}$，启发式方法全部重跑；训练型方法只在一个设置上重训。目的是说明结论不依赖于 0.3 这个数。

### 6.2 采集、返航与上传时间

AUV 选择 CH $j$ 后：

$$
t_{i,\mathrm{arr}} = t + T_{ij}^{\mathrm{travel}}
$$

$$
t_{i,\mathrm{collect}} = t_{i,\mathrm{arr}} + \tau_{ij}
$$

$$
t_{i,\mathrm{arr}}^{S} = t_{i,\mathrm{collect}} + T_{jS}^{\mathrm{travel}}
$$

数据包交付时间还需要考虑 Sink 排队和上传服务。

### 6.3 固定 Sink 的 FCFS 排队

第一版将 Sink 建模为单服务通道，上传速率为 $R_S$。

AUV $i$ 到达时间为 $t_{i,\mathrm{arr}}^{S}$，携带数据量为 $D_i$：

$$
T_i^{\mathrm{upload}}=\frac{D_i}{R_S}
$$

令 $F_S$ 为 Sink 当前队列最后一个任务的预计完成时间：

$$
t_i^{\mathrm{start}}=\max\left(t_{i,\mathrm{arr}}^{S},\ F_S\right)
$$

$$
W_i^{\mathrm{Sink}}=t_i^{\mathrm{start}}-t_{i,\mathrm{arr}}^{S}
$$

$$
F_i=t_i^{\mathrm{start}}+T_i^{\mathrm{upload}}
$$

$$
F_S \leftarrow F_i
$$

若数据包 $p$ 前面尚有 $D_{i,\leq p}$ 字节需要上传：

$$
t_p^{d}=t_i^{\mathrm{start}}+\frac{D_{i,\leq p}}{R_S}
$$

同一批数据的不同数据包可能在不同时间完成交付，因此应使用数据包级交付时间计算 VoI。

**【v1.1·建议】先检验 Sink 利用率，否则排队消融可能是零结果。** 定义 Sink 利用率

$$
\rho_S=\frac{\text{单位时间平均送达的数据量}}{R_S}
$$

也就是上传通道平均有多少时间在忙。如果 AUV 返航间隔长、上传又快，$\rho_S$ 很低，排队等待接近 0，14.3 的消融就不会有差别。因此：在参考策略（例如 7.3 的 Belief + ratio-greedy）下，把 $R_S$、包长和 AUV 数调到 $\rho_S$ 大约在 0.3–0.8；如果怎么调都很低，就不要把"建模 Sink 排队"写成贡献，只作为环境的真实细节。

### 6.4 预测未来交付 VoI

对候选动作 $a$，其待交付数据集合为 $\mathcal{D}_i(a)$：

$$
\boxed{
\hat{V}_i^{\mathrm{Sink}}(a)=
\mathbb{E}
\left[
\sum_{p\in\mathcal{D}_i(a)}
w_p\left[\omega_v+\omega_I e^{-\eta_{k(p)}\left(t_p^{d}-\tau_p\right)}\right]
\mathbf{1}\left(t_p^{d}\leq T_{\max}\right)
\ \middle|\ \mathcal{H}_i(t)
\right]
}
$$

【v1.1·用户确认】公式里已经同时含数据量项和时效项。它直接就是候选动作的价值评分（"分配先验"），不需要再另外按 0.3/0.7 加权一次。

预测需考虑当前 AUV 缓存、到达 CH 后可采集的数据、采集时间、返航时间、Sink 排队、上传服务时间和任务截止时间。

**【v1.1·技术修正】训练循环里用闭式，不用"后验采样 + 离散事件仿真"。** v1.0 写的是后验采样加仿真。每次决策都要对每个 CH、每台 AUV 做一遍，放进强化学习的训练循环会太慢。第一版改用下面的闭式，仿真只用来检验闭式近似的误差和做评估。

**(1) 携带状态。** AUV $i$ 携带的数据只需要两个数（单类别时）：

$$
V_i^{\mathrm{base}}=\sum_{p\in\text{携带}} w_p,\qquad
V_i^{\mathrm{dec}}=\sum_{p\in\text{携带}} w_p\, e^{-\eta\cdot \mathrm{age}_p}
$$

时间过去 $\Delta$ 后，$V_i^{\mathrm{dec}}\leftarrow V_i^{\mathrm{dec}}e^{-\eta\Delta}$，$V_i^{\mathrm{base}}$ 不变。如果再过 $\Delta$ 后送达，这部分携带数据的价值是

$$
\omega_v V_i^{\mathrm{base}}+\omega_I V_i^{\mathrm{dec}}\,e^{-\eta\Delta}
$$

这个记账与逐包求和完全一致（附录 A.1 的检验）。

**(2) 新采集部分（CH $j$ 在报告之后新生成的包）。** 取窗口 $h=h_{ij}=\delta_j+T_{ij}^{\mathrm{arr}}$（4.3），采集后到送达还要 $\tau^{\mathrm{del}}_{ij}$ 时间（采集时间 + 返航 + Sink 排队 + 上传）。定义

$$
I_1(h,\tau)=\omega_v h+\omega_I e^{-\eta\tau}\,\frac{1-e^{-\eta h}}{\eta}
$$

$$
I_2(h,\tau)=\omega_v^{2}h+2\omega_v\omega_I e^{-\eta\tau}\frac{1-e^{-\eta h}}{\eta}+\omega_I^{2}e^{-2\eta\tau}\frac{1-e^{-2\eta h}}{2\eta}
$$

设速率后验为 $\mathrm{Gamma}(a_j,b_j)$，包的基础价值均值为 $\bar{w}$、二阶矩为 $\overline{w^{2}}$（第一版都是 1），则

$$
\boxed{
\mathbb{E}\left[V_{\mathrm{new}}\right]=\frac{a_j}{b_j}\,\bar{w}\,I_1(h,\tau^{\mathrm{del}}_{ij})
}
$$

$$
\boxed{
\operatorname{Var}\left[V_{\mathrm{new}}\right]=\frac{a_j}{b_j}\,\overline{w^{2}}\,I_2(h,\tau^{\mathrm{del}}_{ij})+\frac{a_j}{b_j^{2}}\left(\bar{w}\,I_1(h,\tau^{\mathrm{del}}_{ij})\right)^{2}
}
$$

此外，"这一个窗口里一个包都没有"的概率是 $\left(\dfrac{b_j}{b_j+h}\right)^{a_j}$。这些公式只用到速率后验的均值和方差，所以换成别的滤波器也能用。

**(3) 报告缓存里已有的包。** 它们有确定的时间戳，价值直接按送达时刻算。第一版假设没有队友在途采集同一个 CH（见 5 节）。

**(4) "访问 $j$ 再返航"候选的总价值：**

$$
\hat{V}_i(\mathrm{Visit}\ j\rightarrow \mathrm{Return})
=\underbrace{\omega_v V_i^{\mathrm{base}}+\omega_I V_i^{\mathrm{dec}}e^{-\eta\,\Delta_{c\to d}}}_{\text{已携带}}
+\underbrace{\text{缓存已有部分}}_{(3)}
+\underbrace{\mathbb{E}[V_{\mathrm{new}}]}_{(2)}
$$

其中 $\Delta_{c\to d}$ 是从现在到送达的总时间。多站点候选按顺序递推：每到一站就更新 $(V^{\mathrm{base}},V^{\mathrm{dec}})$。

**(5) Sink 排队和批量近似。** 第一版用对 $F_S$ 的点估计（最近一次观测到的队列状态，见 9.1）；同一批数据按"批末时刻"交付，这是保守近似，评估时仍用包级交付时间。这个近似造成的误差，在 14.1 的 VoI 预测误差里如实报告。

**(6) 容量截断。** AUV 剩余容量 $C_i-S_i$ 小于预期采集量时，把预期采集量限制在剩余容量。精确处理留到实现时再定。

**(7) 检验情况。** 公式 (2) 的均值和方差已用蒙特卡洛核对：5 组参数（含 $\omega_v=1$ 和 $\omega_I=1$ 两个退化情形）、2 种基础价值分布，最大相对偏差 1.2%（每组 12 万样本，属于采样噪声量级）。见附录 A.1。

预测模块输出：

$$
\mathbf{z}_{ij}^{\mathrm{pred}}
=
\left[
\hat{A}_j,\ \sigma_{A_j},
\ \hat{Q}_j,\ \sigma_{Q_j},\ P_{\mathrm{overflow},j},
\ \hat{V}_{ij}^{\mathrm{Sink}},\ \sigma_{V_{ij}},\ \hat{t}_{ij}^{d}
\right]
$$

预测结果作为 Actor 的局部观测特征，而不是直接替代 Actor 决策。

## 7. AUV 三维导航与采集

### 7.1 三维直线导航

AUV $i$ 的位置：

$$
\mathbf{x}_i(t)=\left(x_i(t),y_i(t),z_i(t)\right)
$$

到 CH 通信范围边界的距离：

$$
d_{ij}(t)=
\left[\left\|\mathbf{x}_i(t)-\mathbf{c}_j\right\|-R_j\right]^{+}
$$

航行时间：

$$
T_{ij}^{\mathrm{travel}}=\frac{d_{ij}(t)}{v_i}
$$

底层采用固定速度直线导航。高层连续选择目标形成任务级三维路径：

$$
\mathcal{P}_i=
\left(
\mathbf{x}_i^{0},
\mathbf{p}_{j_1},
\mathbf{p}_{j_2},
\ldots,
\mathbf{s}
\right)
$$

本文研究任务级三维路径规划，不研究连续速度控制、PID 或六自由度动力学。

### 7.2 数据采集约束

若 AUV 在 CH 停留 $\tau_{ij}$，通信速率为 $r_{ij}$，则：

$$
X_{ij}
\leq
\min\left\{
Q_j^{\mathrm{available}},\ C_i-S_i,\ r_{ij}\tau_{ij}
\right\}
$$

$$
S_i^{\mathrm{new}}=S_i+X_{ij}\leq C_i
$$

预测用于提前规划；到达 CH 后，实际新报告用于修正采集计划。

### 7.3 返航决策

不采用单一固定缓存阈值作为唯一返航规则。Actor 应综合当前缓存的预计交付价值、继续访问候选 CH 的增量价值、时间成本、Sink 预计排队、剩余任务周期和 AUV 存储余量，学习何时返航。

**【v1.1·建议】两个启发式规则（用作强基线和特征）。**

**规则 A：ratio-greedy（每趟收益 ÷ 用时，返航也是候选）。** 每次决策时，对每个候选 $a\in\{\mathrm{Visit}\ j\}_{j}\cup\{\mathrm{Return}\}$，用 6.4 的闭式算出"执行 $a$ 然后返航送达"的总价值 $\hat{V}_i(a)$，以及从现在到送达完成的总时间 $T_i(a)$，选

$$
a^{*}=\arg\max_{a}\ \frac{\hat{V}_i(a)}{T_i(a)}
$$

其中 $\mathrm{Return}$ 的价值就是已携带数据的送达价值，用时是返航 + 排队 + 上传。它不需要学习，是后面 MAPPO 必须超过的对照（14.2、14.5 的 Belief + ratio-greedy）。

**规则 B：边际采集判据（要不要再绕一站）。** 访问 $j$ 值得，当且仅当新增收益大于绕路造成的损失：

$$
\underbrace{\hat{V}_j^{\mathrm{new}}}_{\text{6.4 的新增价值}}
\ >\
\underbrace{\omega_I\, V_i^{\mathrm{dec}}\,e^{-\eta T_i^{\mathrm{ret}}}\left(1-e^{-\eta\,\Delta_j}\right)}_{\text{绕路让已携带数据多衰减的部分}}
$$

其中 $T_i^{\mathrm{ret}}$ 是现在直接返航的送达时间，$\Delta_j$ 是访问 $j$ 比直接返航多花的时间。左边已含数据量项和时效项。

注意：这条判据是我推导的，只比较了"绕一站"与"直接返航"，没有计入时间的机会成本和队友的行动。把它当作 Actor 的输入特征 $g_{ij}$（边际净收益），第一版不作硬约束，也不保证对 MAPPO 有帮助（附录 A.3）。

**势函数。** 10.4 用到的 $\Phi_i$ 就是"现在直接返航的送达价值"：

$$
\Phi_i(s)=
\begin{cases}
\omega_v V_i^{\mathrm{base}}+\omega_I V_i^{\mathrm{dec}}\,e^{-\eta T_i^{\mathrm{ret}}}, & t+T_i^{\mathrm{ret}}\le T_{\max}\\[2pt]
0, & \text{否则}
\end{cases}
$$

这里 $T_i^{\mathrm{ret}}$ 只含航行与上传时间，不含排队（排队依赖队友，不属于 AUV 自己的状态）。

## 8. 异步宏动作与 Dec-POSMDP

### 8.1 异步决策时刻

每台 AUV 有独立决策时刻：

$$
\mathcal{T}_i=\left\{t_i^{0},t_i^{1},t_i^{2},\ldots\right\}
$$

一般：

$$
t_i^{n}\neq t_j^{n},
\quad
i\neq j
$$

不强制所有 AUV 在固定时间步同时重新决策。

### 8.2 事件类型

- 到达 CH 通信范围；
- CH 采集完成；
- AUV 到达 Sink；
- Sink 上传开始；
- Sink 上传完成；
- 等待宏动作结束；
- 任务周期截止。

### 8.3 宏动作空间

**【v1.1·建议】第一版动作空间收缩为 Visit + Return。** 理由：每多一类宏动作，策略要学的东西和调试量都翻倍；而 Wait 和 Collect(τ) 的收益要等基本框架跑通后才看得出。

$$
\mathcal{A}_i=
\left\{
\mathrm{Visit}\left(CH_1\right),\ldots,\mathrm{Visit}\left(CH_N\right),
\mathrm{Return}\left(\mathrm{Sink}\right)
\right\}
$$

- **Visit($j$)：** 航行到 CH $j$ 的通信范围，申请服务锁（9.2），成功后采集，直到"缓存取空"或"AUV 存储装满"为止。采集时长由缓存量和通信速率决定，不由策略选。宏动作在采集完成时结束，AUV 再做下一次决策。
- **Return：** 航行到 Sink，排队、上传全部携带数据。宏动作在上传完成时结束。

**动作掩码（只允许可行动作）：**

- 携带量为 0 时，掩掉 Return；
- AUV 存储已满时，掩掉所有 Visit；
- 剩余时间不够"访问 $j$ → 返航 → 上传"（含最短排队）时，掩掉 Visit($j$)，因为数据来不及送达，价值为 0；
- 被服务锁拒绝过的 CH，在一个服务时长内掩掉（9.2）；
- 所有动作都被掩掉时，AUV 原地结束，不再决策。

**扩展 E1（延后）：** 恢复 v1.0 的等待与可选采集时长。

导航阶段：

$$
\mathcal{A}_i^{\mathrm{nav}}=
\left\{
\mathrm{Visit}\left(CH_1\right),\ldots,\mathrm{Visit}\left(CH_N\right),
\mathrm{Return}\left(\mathrm{Sink}\right),
\mathrm{Wait}\left(\delta\right)
\right\}
$$

服务阶段：

$$
\mathcal{A}_i^{\mathrm{service}}=
\left\{
\mathrm{Collect}\left(\tau_1\right),\ldots,\mathrm{Collect}\left(\tau_L\right),
\mathrm{Return}
\right\}
$$

可离散化采集时间，例如 $\tau\in\{30,60,120\}$ 秒，具体值需依据通信速率和缓存规模确定。

### 8.4 异步状态转移

AUV $i$ 第 $n$ 次决策的局部信念状态为 $b_i^{n}$，动作为 $a_i^{n}$，持续时间：

$$
\Delta_i^{n}=t_i^{n+1}-t_i^{n}
$$

转移：

$$
P\left(b_i^{n+1},\Delta_i^{n}\mid b_i^{n},a_i^{n}\right)
$$

持续时间不同，因此回报和优势估计必须考虑实际持续时间。

### 8.5 时间折扣

$$
\Gamma\left(\Delta_i^{n}\right)
=
\gamma^{\Delta_i^{n}/\Delta_0}
$$

其中 $\Delta_0$ 是参考时间单位。由于 VoI 已使用指数时间衰减，主实验可考虑令物理时间折扣 $\gamma=1$，或单独开展外部折扣敏感性实验，避免重复惩罚时间。

**【v1.1·建议】默认 $\gamma=1$。** 价值本身已随时间衰减，再加折扣会重复惩罚时间。周期有限时，$\gamma=1$ 的回报有界。为了让策略知道“还剩多少时间”，把 $T_{\max}-t$ 放进观测（9.1）。另做一个 $\gamma<1$ 的敏感性实验（例如每个 $\Delta_0$ 折扣 0.99）。

## 9. 局部观测、任务冲突与 CTDE

### 9.1 局部观测

$$
o_i(t)=\left\{o_i^{\mathrm{self}},o_i^{\mathrm{CH}},o_i^{\mathrm{pred}},o_i^{\mathrm{Sink}}\right\}
$$

自身状态：

$$
o_i^{\mathrm{self}}
=
\left\{
\mathbf{x}_i,\ S_i/C_i,\ m_i,\ T_{\max}-t,\ \mathbf{e}_i,\ V_i^{\mathrm{base}},\ V_i^{\mathrm{dec}}
\right\}
$$

CH 报告与历史：

$$
o_i^{\mathrm{CH}}
=
\left\{
Q_j(t_r),\ \mathcal{T}_j(t_r),\ G_j(t_r),\ \delta_j
\right\}_{j=1}^{N}
$$

预测状态：

$$
o_i^{\mathrm{pred}}
=
\left\{
\hat{\lambda}_j,\ \sigma_{\lambda_j},
\ \hat{Q}_j,\ \sigma_{Q_j},
\ P_{\mathrm{overflow},j},\ \hat{V}_{ij}^{\mathrm{Sink}}
\right\}_{j=1}^{N}
$$

Sink 局部估计：

$$
o_i^{\mathrm{Sink}}
=
\left\{
\hat{W}_i^{\mathrm{Sink}},\ \hat{R}_S,\ \text{last-observation age}
\right\}
$$

AUV 不直接获得其他 AUV 的完整实时状态，也不直接获得 CH 当前真实状态，除非通信机制实际提供该信息。

**【v1.1·技术修正】新增的观测项：**

- **AUV 编号 $\mathbf{e}_i$（one-hot）。** Actor 参数共享时，观测相同就给出相同的动作分布，三台 AUV 会扎堆去同一个 CH，CH 只有 5 个时尤其严重。加入编号后，策略才能让不同 AUV 分工。14.2 的 I 组检验这一点。
- **携带状态 $V_i^{\mathrm{base}},\,V_i^{\mathrm{dec}}$。** 6.4 定义的两个数，描述 AUV 当前携带数据的价值。
- **$G_j(t_r)$。** 累计生成计数器，用于更新速率估计（3.3、4.2），不必直接送进网络。

**【v1.1·建议】Sink 队列的可观测性。** 默认只有 AUV 抵达 Sink 附近时才能看到队列；返航前，$\hat{W}_i^{\mathrm{Sink}}$ 用最近一次观测值，并附带“上次观测的年龄”。Oracle 版本（始终知道真实队列）用于 14.3 对照。

### 9.2 CH 本地服务锁

$$
L_j(t)\in\left\{\mathrm{Free},\ \mathrm{Locked}\left(i,t_{\mathrm{exp}}\right)\right\}
$$

AUV 进入通信范围后申请服务；CH 检查锁状态；空闲时授予服务权；已锁定时拒绝；被拒绝的 AUV 重新决策；超时后释放锁。

**【v1.1·建议】被拒绝后的默认做法：** 立即重新决策，被拒绝的 CH 在一个服务时长内被动作掩码掩掉（8.3），避免反复申请。

服务锁保证同一时刻的服务互斥，但不能保证多台 AUV 不会同时在途中前往同一 CH。第一版先接受这种可能的无效航行，后续再研究预约或轻量级意图共享。

### 9.3 CTDE

训练阶段全局状态：

$$
s^{g}(t)=
\left\{
S_{\mathrm{CH}}(t),\ S_{\mathrm{AUV}}(t),\ S_{\mathrm{Sink}}(t),\ T_{\max}-t
\right\}
$$

集中式 Critic 可使用全部 CH 真实缓存和生成过程、AUV 位置和存储、Sink 真实队列、服务锁和数据包交付状态。

**【v1.1·技术修正】Critic 再加上被评估 AUV 的编号，写成 $V_{\psi}(s^{g},\mathbf{e}_i)$。** 按交付者归属（10.4）以后，各 AUV 的未来回报不同，同一个全局状态对不同 AUV 的价值也不同。

执行阶段 Actor 只使用自身局部观测和预测状态，不依赖 AUV 间持续通信。

## 10. Asynchronous MAPPO

### 10.1 Actor-Critic 结构

参数共享 Actor：

$$
\pi_{\theta}\left(a_i\mid o_i\right)
$$

集中式 Critic：

$$
V_{\psi}\left(s^{g},\mathbf{e}_i\right)
$$

所有能力相同的 AUV 共享 Actor 参数。训练时使用全局状态，执行时各 AUV 根据局部观测独立决策。

### 10.2 独立轨迹缓存

每台 AUV 维护独立缓存 $C_i$。第 $n$ 次宏动作记录：

$$
\mathcal{T}_i^{n}
=
\left(
o_i^{n},\ s_n^{g},\ a_i^{n},\
\log \pi_{\theta_{\mathrm{old}}}\left(a_i^{n}\mid o_i^{n}\right),\ 
R_i^{n},\ \Delta_i^{n},\ o_i^{n+1},\ s_{n+1}^{g}
\right)
$$

宏动作完成后形成转移并写入对应缓存，训练时汇总多台 AUV 样本。

### 10.3 PPO 旧策略一致性

一个 rollout 批次采集期间冻结旧策略参数 $\theta_{\mathrm{old}}$，确保保存的 log-prob 与采样策略一致。

宏动作跨 rollout 边界时可：

1. 等待当前宏动作结束后更新；
2. 在边界截断宏动作，保存部分累计奖励并用 Critic bootstrap。

第一版建议先采用方案 1，便于验证；后续再优化采样效率。

**【v1.1·建议】** 采用方案 1。周期结束（$T_{\max}$）时仍在执行的宏动作被截断，其 bootstrap 值取 0，因为周期结束后没有未来收益；此时尚未送达的数据价值为 0。

### 10.4 宏动作累计奖励

Sink 在数据包完成上传时产生实际价值（含数据量项）：

$$
r_e=
\sum_{p:\,t_p^{d}=t_e}
w_p\left[\omega_v+\omega_I e^{-\eta_{k(p)}\left(t_e-\tau_p\right)}\right]
$$

**【v1.1·技术修正】三层奖励：按交付者归属 + 势函数塑形 + 小比例团队项。** v1.0 的做法是把所有 AUV 的宏动作区间内的团队奖励都算给每个 AUV。由于各 AUV 的区间重叠，同一笔交付会被算进多台 AUV 的回报，信用分配很嘈杂。

**(1) 按交付者归属。** 每个数据包只由一台 AUV 送达。令 $r_e^{(i)}$ 为事件 $e$ 中由 AUV $i$ 的货物产生的价值：

$$
R_i^{n}=
\sum_{e:\,t_i^{n}\leq t_e<t_i^{n+1}}
\gamma^{\left(t_e-t_i^{n}\right)/\Delta_0}\, r_e^{(i)}
$$

Return 宏动作到上传完成才结束，所以 AUV $i$ 的所有送达事件都落在它自己的 Return 区间里。$\gamma=1$ 时，所有 AUV 的 $R_i^n$ 之和恰好等于 Sink 账本的总价值：没有重复，也没有遗漏。这一点要写成单元测试（守恒检验）。

**(2) 势函数塑形。** 采集发生在 Visit，奖励却要等 Return 才出现，中间隔了一整趟航行，奖励很延迟。用 7.3 的 $\Phi_i$（"现在直接返航的送达价值"）做塑形：

$$
\tilde{R}_i^{n}=R_i^{n}+\Gamma\!\left(\Delta_i^{n}\right)\Phi_i\!\left(s_{n+1}\right)-\Phi_i\!\left(s_{n}\right)
$$

AUV 起始货物为空、$\Phi_i=0$；上传完成后货物清空、$\Phi_i=0$；周期结束时 $\Phi_i=0$。$\gamma=1$ 时，沿一台 AUV 的轨迹把 $\Phi_i(s_{n+1})-\Phi_i(s_n)$ 逐项相加，中间项相互抵消，总和为 0，所以塑形不改变任何轨迹的总回报，也就不改变最优策略。这个证明是我对 $\gamma=1$ 做的逐轨迹推导，没有找到针对异步 SMDP 的已发表证明（附录 A.3）；退路是引用 Ng 等 1999 和 Lu & Schwartz 2011 的势函数塑形结论，并用实验验证"加不加塑形，最终策略的评估价值一致"。

**(3) 小比例团队项。** 最终用于训练的奖励是

$$
R_i^{n,\mathrm{train}}=(1-\kappa)\,\tilde{R}_i^{n}+\frac{\kappa}{M}\sum_{e:\,t_i^{n}\leq t_e<t_i^{n+1}}\gamma^{\left(t_e-t_i^{n}\right)/\Delta_0}\, r_e
$$

第二项是该区间内全队所有交付的平均份额（不论由谁送达）。默认 $\kappa=0.1$，$\kappa=0$ 作消融。它的作用是让 AUV 稍微关心队友的收益，避免互相抢同一个 CH 时各自只顾自己。后面的 TD 误差和 GAE 里，$R_i^n$ 一律指这个 $R_i^{n,\mathrm{train}}$。

**(4) v1.0 的写法保留为消融。** 即"所有交付都算给每个 AUV"，记作"团队奖励（v1.0）"，在 14.2 里与 (1)+(2)+(3) 对比。

**评估永远只看 Sink 账本。** 环境总收益只从唯一的 Sink 交付账本计算，不由各 AUV 的局部回报相加得到。

### 10.5 时间感知 TD 误差

$$
\boxed{
\delta_i^{n}
=
R_i^{n}
+
\Gamma\left(\Delta_i^{n}\right) V_{\psi}\left(s_{n+1}^{g},\mathbf{e}_i\right)
-
V_{\psi}\left(s_n^{g},\mathbf{e}_i\right)
}
$$

**【v1.1·技术修正】** 这里的 $R_i^{n}$ 是 10.4 的 $R_i^{n,\mathrm{train}}$；Critic 条件于 AUV 编号；周期结束的终止状态 $V_{\psi}=0$。

### 10.6 异步 GAE

沿每台 AUV 自己的决策序列：

$$
\boxed{
\hat{A}_i^{n}
=
\delta_i^{n}
+
\Gamma\left(\Delta_i^{n}\right)
\lambda_{\mathrm{GAE}}^{\Delta_i^{n}/\Delta_0}
\hat{A}_i^{n+1}
}
$$

### 10.7 PPO-Clip

概率比：

$$
r_i^{n}\left(\theta\right)
=
\frac{\pi_{\theta}\left(a_i^{n}\mid o_i^{n}\right)}
{\pi_{\theta_{\mathrm{old}}}\left(a_i^{n}\mid o_i^{n}\right)}
$$

Actor 损失：

$$
\boxed{
L_{\mathrm{actor}}
=
-\mathbb{E}
\left[
\min\left(
r_i^{n}\hat{A}_i^{n},
\
\operatorname{clip}\left(r_i^{n},1-\epsilon,1+\epsilon\right)\hat{A}_i^{n}
\right)
+
c_H H\left(\pi_{\theta}\right)
\right]
}
$$

Critic 损失：

$$
L_{\mathrm{critic}}
=
\mathbb{E}
\left[
\left(
V_{\psi}\left(s_n^{g},\mathbf{e}_i\right)-\hat{G}_i^{n}
\right)^{2}
\right]
$$

总损失：

$$
L=L_{\mathrm{actor}}+c_V L_{\mathrm{critic}}
$$

实现时应明确熵项符号约定，并使用有效动作掩码计算动作概率和熵。

## 11. 训练流程伪代码

```text
Initialize shared Actor π_θ
Initialize centralized Critic V_ψ
Initialize CH predictors
Initialize event-driven environment

for each training episode:
    reset environment
    reset per-AUV trajectory buffers
    initialize event queue
    freeze old policy θ_old

    while simulation_time < T_max:
        event = pop_earliest_event()
        advance_environment_to(event.time)
        update_CH_generation_and_buffers()
        update_AUV_states()
        update_Sink_FCFS_queue()

        if event triggers an AUV decision:
            i = event.auv_id
            o_i = build_local_observation(i)
            prediction_i = update_or_query_predictor(i, o_i)   # [v1.1] 用 G_j 差分更新速率；6.4 闭式预测
            action_mask = build_feasible_action_mask(i)
            a_i, log_prob = Actor_old(o_i, action_mask)
            start_macro_action(i, a_i)
            store_macro_action_start(...)

        if a macro action terminates:
            calculate_macro_duration()
            calculate_macro_interval_reward()   # [v1.1] 按交付者归属 + 势函数塑形 + κ 团队项（10.4）
            append_transition_to_AUV_buffer()

    compute_actual_sink_delivered_VoI()
    compute_duration_aware_TD_errors()
    compute_asynchronous_GAE()
    update_shared_Actor_with_PPO()
    update_centralized_Critic()
    evaluate_and_save_checkpoints()
```

正式实现时还需明确事件优先级和同时事件处理。【v1.1】宏动作跨 rollout 边界处理见 10.3，奖励归属见 10.4，均已给出默认方案。

## 12. 优化问题

核心目标：

$$
\boxed{
\max_{\pi}
\mathbb{E}_{\pi}
\left[
\sum_{p\in\mathcal{D}_{\mathrm{del}}\left(T_{\max}\right)}
w_p\left[\omega_v+\omega_I e^{-\eta_{k(p)}\left(t_p^{d}-\tau_p\right)}\right]
\right]
}
$$

其中 $\mathcal{D}_{\mathrm{del}}\left(T_{\max}\right)$ 是任务周期内真正完成 Sink 上传的数据包集合。

【v1.1·用户确认】目标含数据量项：$\omega_v=0.3$，$\omega_I=0.7$（6.1）。$w_p=1$ 时等价于 $J=\omega_v N_{\mathrm{del}}+\omega_I V_{\mathrm{I}}$。

约束：

**CH 缓存：**

$$
0\leq Q_j(t)\leq B_j
$$

【v1.1·技术修正】缓存满时丢弃新到的包，被丢弃的包价值为 0（3.3）。

**AUV 存储：**

$$
0\leq S_i(t)\leq C_i
$$

**通信范围：**

$$
\left\|\mathbf{x}_i(t)-\mathbf{c}_j\right\|\leq R_j
$$

**CH 服务互斥：**

$$
\sum_{i=1}^{M} y_{ij}(t)\leq 1
$$

**AUV 速度：**

$$
\left\|\dot{\mathbf{x}}_i(t)\right\|\leq v_i
$$

**Sink 服务：** 单服务通道下，同一时刻最多一台 AUV 占用上传服务。

**有限周期：**

$$
0\leq t_p^{d}\leq T_{\max}
$$

不额外设置很大的最短路径奖励，避免策略偏向近距离但低价值 CH。航行、等待和上传时间已经通过数据实际交付时间影响 VoI。

## 13. 创新点与论文贡献

### 创新点 1：稀疏局部报告下的 CH 数据状态预测

- 根据 CH 缓存报告和数据包时间戳维护历史状态；
- 在线估计未知数据生成速率；
- 预测 AUV 到达时的缓存状态；
- 估计缓存溢出风险和预测不确定性；
- 针对混合生成模式扩展条件速率估计。

### 创新点 2：面向 Sink 实际交付的预测 VoI

- 将数据生成时间戳与价值衰减结合；
- 纳入采集、返航和上传时间；
- 建模固定 Sink 的 FCFS 排队；
- 使用数据包级上传完成时间计算价值；
- 预测候选任务最终能够兑现的 Sink 端 VoI。

### 创新点 3：预测增强的异步 MAPPO 长期决策框架

- 将长期多轮采集建模为局部观测下的异步决策问题；
- 使用宏动作处理不同 AUV 的任务持续时间；
- 使用参数共享 Actor 和集中式 Critic；
- 联合学习 CH 选择、访问时机、采集时长和返航；
- 通过真实 Sink 交付 VoI 形成团队优化目标。

**注意：异步 MAPPO 是采用的算法基础，不应单独宣称为本文原创算法。** 贡献应落在其面向水下预测、交付 VoI 和长期异步采集问题的建模与机制设计上。

**【v1.1·建议】贡献的写法要与证据强度匹配。** 创新点 1、2 的价值取决于 14.2 的消融：预测输入（B、C、D 组）是否显著优于 A 组，D 是否超过 H1。玩具实验里预测的增益只有 3–5 个百分点，而且依赖访问次数（附录 A.1），所以建议把第一贡献写成“新的问题设定”（连续随机到达、多趟往返、显式返航、送达时按包结算且含数据量项），把预测写成“在这一设定下，信息何时有用的系统研究”。创新点 3 只在消融支持时才写成方法贡献；第一版动作空间只含 Visit 与 Return（8.3），“访问时机”和“采集时长”属于扩展。强度评估见附录 A.2。

## 14. 实验方案

### 14.1 预测模型性能

【v1.1·用户确认】三种到达类型，默认设置见 3.2：

| 场景 | 特征 |
| :--: | :-- |
| S1 | 固定 Poisson 速率 |
| S2 | 分段突变速率 |
| S3 | 正常／突发模式切换（MMPP，压力测试） |

对比方法：

- Last Observation；
- Moving Average；
- Gamma-Poisson（静态，$\nu=0$）；
- Gamma-Poisson + 向先验遗忘（4.2，默认 $\nu=3/T_{\max}$）；
- Oracle（使用真实未来生成速率，仅作上界）；
- 【可选扩展】网格贝叶斯滤波；以及 MMPP + 条件 Gamma-Poisson（v1.0 的对比项，随 4.4 的隐状态滤波一起延后）。

指标：

$$
\mathrm{MAE}_{\lambda},
\quad
\mathrm{RMSE}_{\lambda},
\quad
\mathrm{MAE}_{Q},
\quad
\mathrm{RMSE}_{Q}
$$

以及未来生成量误差、缓存预测误差、溢出概率校准、VoI 预测误差和预测区间覆盖率（80% 区间的实际覆盖率）。

**【v1.1·建议】按"每个 CH 每回合被访问次数"分组报告**（取 2、5、10、30 次）。预测的价值取决于访问频率：我的玩具实验里，每个节点约 2 次访问时，信念相对"只看距上次服务的时间"基本没有增益，5 次以上才有 3–5 个百分点（附录 A.1）。这张图本身就是"信息在什么条件下有用"的实证结果。

**【v1.1·建议】S2、S3 如实报告覆盖率下降。** Poisson 类估计器在速率突变时会过度自信，这不是实现问题，是模型假设的局限。我的玩具实验（不是 MMPP，是我自己写的突变和"混合"场景）里，80% 预测区间的实际覆盖率：静态估计器在突变场景约 0.66、"混合"场景约 0.56；向先验遗忘后约 0.86 和 0.76；速率固定时约 0.90–0.92（附录 A.1）。MMPP 我没有单独测过，S3 的覆盖率要在环境里实测。

### 14.2 预测是否改善 MAPPO 决策

保持环境、Actor-Critic 结构、训练预算和随机种子一致，仅改变 Actor 可获得的预测信息。所有组都含：自身状态（含 AUV 编号、携带状态 $(V^{\mathrm{base}},V^{\mathrm{dec}})$、剩余时间）和报告年龄。

| 组别 | Actor 输入 |
| :--: | :-- |
| A | 当前缓存和时间戳 |
| B | 当前缓存 + 生成速率估计 |
| C | 当前缓存 + 未来缓存预测 |
| D | 当前缓存 + 未来缓存 + 未来交付 VoI |
| E | Oracle 未来状态 |

观察 Sink 累计价值、数据交付延迟、溢出率和返航策略变化。

**【v1.1·建议】关键对照：D 组必须超过不学习的强启发式。** 同一环境下加两条参考线：

| 组别 | 方法 | 说明 |
| :--: | :-- | :-- |
| H1 | Belief + ratio-greedy（7.3 规则 A） | 用 6.4 的闭式预测，不训练 |
| H2 | Oracle-rate ratio-greedy | 同样的规则，但知道真实速率 |

我的玩具实验里（单 AUV，$\mu=0.005$，四种速率场景，本轮重跑确认），"网格信念 + ratio-greedy"达到"能看到真实缓冲区的参考策略"的约 97%–99%；知道真实速率的版本约 99%–100%；每趟固定访问 3 个节点的规则约 91%–95%（附录 A.1）。这个玩具比本文环境简单得多，只能说明天花板很高。另外要注意，玩具里用的是网格信念，而 H1 用的是 Gamma-Poisson（向先验遗忘）；玩具里不遗忘的 Gamma-Poisson 在非平稳场景只有约 91%–96%，所以 H1 的实际水平要在本文环境里测出来。因此 MAPPO 的优势必须在 H1 之上、超过多随机种子的波动范围才算数。如果 D 组赢不了 H1，论文重心要转向"问题设定 + 信息何时有用的系统研究"，不能强行宣称 MAPPO 更好。这个风险要在第 5 周前就知道（见 15 节）。

**【v1.1·建议】其他消融：**

| 组别 | 内容 |
| :--: | :-- |
| F | 奖励归属：(1)+(2)+(3) 与 v1.0 团队奖励（10.4 (4)）对比；$\kappa\in\{0,0.1\}$；有无势函数塑形 |
| G | $\omega_v$ 敏感性（6.1） |
| I | 无 AUV 编号的共享 Actor（说明编号的必要性） |

### 14.3 Sink 排队实验

比较不考虑 Sink 排队、历史排队估计、预测排队状态和 Oracle 真实队列状态。

指标：

- Sink 平均等待时间；
- 数据包平均交付延迟；
- Sink 累计价值；
- AUV 平均存储占用；
- 返航次数；
- 多 AUV 同时抵达 Sink 的比例。

**【v1.1·建议】前置条件：** 先按 6.3 检验 $\rho_S$ 落在 0.3–0.8；否则本实验不会有差异，不应作为创新点。

### 14.4 同步与异步实验

比较：

- Synchronous MAPPO；
- Asynchronous MAPPO；
- Prediction-Enhanced Asynchronous MAPPO。

尽量保持策略网络规模、训练预算和环境条件一致，验证异步决策是否减少无效决策、改善长期价值，以及预测是否进一步改善异步任务选择。

**【v1.1·建议】优先级较低。** 时间紧时，本实验是第一个被砍的（见 15 节）。

### 14.5 基线体系

| 类型 | 方法 | 用途 |
| :-- | :-- | :-- |
| 传统方法 | Nearest Neighbor | 最近节点访问 |
| 传统方法 | Greedy VoI | 当前报告价值贪心 |
| 预测启发式 | $\left(\lambda_j Q_j/B\right)$ Urgency | 速率与缓存紧迫度 |
| 预测启发式 | Bayesian Prediction + Greedy | 检验预测器独立作用（选 $\hat V$ 最大者） |
| 预测启发式 | **Belief + ratio-greedy**（7.3 规则 A）【v1.1·建议】 | 强基线，MAPPO 必须超过 |
| **已有论文方法（适配版）**【v1.1·用户确认】 | GAAP-style | 见下表 |
| **已有论文方法（适配版）**【v1.1·用户确认】 | SPDC-GA-style | 见下表 |
| **已有论文方法（适配版）**【v1.1·用户确认】 | LG-DQN-style | 见下表 |
| 多智能体 RL | MAPPO（同步） | 同步策略基线 |
| 多智能体 RL | Async-MAPPO（无预测） | 异步策略基线（即 14.2 的 A 组） |
| 完整方法 | Prediction-Enhanced Async-MAPPO | 本文方案 |
| 上界 | Oracle Async-MAPPO；Oracle-rate ratio-greedy | 预测上界 |

Nearest Neighbor、Greedy VoI 和 Urgency 都需要一个返航规则：存储占比达到 $\theta_{\mathrm{ret}}$ 或没有可行的 Visit 时返航，$\theta_{\mathrm{ret}}$ 在验证种子上调参。

**【v1.1·用户确认】已有论文方法的适配版。** 下面三种方法原本都是单轮采集（每个簇只有一个数据生成时刻，没有数据生成速率、缓存和持续生成），不能直接搬到本文的长期多趟设定。适配规则如下。"原论文机制"一列来自 Xu 等 2025 对它们的转述（我没有读过 GAAP 原文，也没有读过 LG-DQN 全文），所以是二手信息，见附录 A.3。

| 方法 | 原论文机制（据 Xu 等 2025 转述） | 适配到本文设定的规则 | 与原论文的主要差异 | 优先级与预计工作量 |
| :-- | :-- | :-- | :-- | :-- |
| **GAAP-style** | 贪心地为每台 AUV 选下一个簇，依据 VoI 的衰减；之后去簇内满足约束的最近节点 | 下一目标选"到达时刻衰减后价值最大"的 CH：$\mathrm{score}_j=\sum_{p\in\text{报告缓存}}w_p[\omega_v+\omega_I e^{-\eta(t+T^{\mathrm{arr}}_{ij}-\tau_p)}]$，只用报告里的缓存，不预测新生成；返航用 $\theta_{\mathrm{ret}}$ 规则；"簇内最近节点"一步不需要，CH 固定 | 原为单轮、每簇一个生成时刻；本文多趟、连续生成。"按衰减贪心"的具体打分是我对一句话描述的解读，另一种解读（按衰减速度排序）作为变体 | **必做**，约 1 天 |
| **SPDC-GA-style** | 给每台 AUV 分配一个区域，用遗传算法联合优化路径规划和簇头选择 | 用 k-means 按位置把 CH 分成 $M$ 个区域，每台 AUV 一个区域；每趟出发前，用 GA 在本区域内选 CH 子集和访问顺序，目标是"以最近报告缓存估计的送达价值 ÷ 趟用时"（不预测），趟中不重规划；"簇头选择"省略，CH 固定 | 不含 He 2022（SPDC 原文）的队友状态预测、观察簇和定时卸载机制，那部分在 2 节已声明不作核心创新；我读过的 He 2022 用的是几何插入式路径规划而不是 GA，所以"SPDC-GA"是 Xu 等的改造版，这里按其转述实现。要对标完整 SPDC，另作扩展 SPDC-full | **必做**，约 3–4 天（GA、区域划分、评估） |
| **LG-DQN-style** | Wang 等 2024 的"local global deep Q learning"，用于多 AUV 路径规划、降低网络总能耗；Xu 等转述为：取离簇中心最近的节点作簇头，用 DQN 规划各 AUV 到簇头的路径，回报为总 VoI | 简化为共享参数的独立 DQN：每台 AUV 在自己宏动作结束时用 ε-greedy 选 Visit/Return；输入与 14.2 的 A 组相同（无预测）；奖励用 10.4 (1) 的按交付者归属价值，保证与 MAPPO 同一奖励；无集中式 Critic，不复现"local-global"结构 | 不能称为 LG-DQN 的复现。原版目标是能耗，这里是价值；簇头选择规则不适用；原文内部结构未核实 | **次优先**，约 2–3 天（复用环境和观测构造） |

工作量是我的粗估，没有实测。

**公平性规则：**

- 所有方法使用同一环境、同一信息（只用 CH 报告）、同一评估随机种子；
- 每个方法的超参数在同一预算的验证种子上调；
- 论文里一律写成"X-style (adapted)"，列出与原论文的假设差异，不写"复现"；
- 如果某个适配基线在本环境里异常差，先检查是不是单轮假设导致（例如原方法没有返航规则），再检查实现，并在附录说明。

### 14.6 核心指标

【v1.1·用户确认】总价值同时报告三个数：

$$
J=\omega_v N_{\mathrm{del}}+\omega_I V_{\mathrm{I}},\qquad
N_{\mathrm{del}},\qquad V_{\mathrm{I}}
$$

（$w_p=1$ 时；一般形式见 6.1。）报告 $N_{\mathrm{del}}$ 和 $V_{\mathrm{I}}$，是为了与只追求吞吐量或只追求新鲜度的已有方法公平比较。

$$
\mathrm{AoI}_{\mathrm{Sink}}^{\mathrm{average}},
\quad
D_{\mathrm{delivery}}^{\mathrm{average}}
$$

$$
P_{\mathrm{overflow}},
\quad
W_{\mathrm{Sink}}^{\mathrm{average}},
\quad
P_{\mathrm{conflict}}
$$

还可报告数据交付率、AUV 平均航行距离、存储利用率、CH 服务公平性和多随机种子训练方差。

### 14.7 实验规模与统计【v1.1·建议】

- **调试规模：** 3 AUV、5 CH、1 Sink。只用于验证接口和账本，不用于出结论。
- **主实验规模：** 至少 10 个 CH，3 台 AUV；调整任务周期和速率，使每个 CH 每回合被访问约 5–30 次。
- **稀疏版本：** 每个 CH 每回合约 2 次访问，展示预测信息失效的边界（与 14.1 对应）。
- **统计：** 每个训练配置至少 5 个训练随机种子；每个种子在固定的一批评估回合上测试（回合数先取 50，视方差调整）；结果报告均值与 95% 置信区间，不只报最好种子。
- **账本检验：** 每个实验开始前跑守恒单元测试：Sink 账本总价值 = 各 AUV 按交付者归属的奖励之和（10.4 (1)）；携带状态的记账 = 逐包求和（6.4 (1)）。

## 15. 项目开发顺序

### 阶段一：事件驱动环境

1. CH 数据生成（S1、S2、S3）和有限缓存（缓存满时丢新包）；
2. 数据包生成时间戳和累计生成计数器 $G_j$【v1.1】；
3. AUV 三维直线导航；
4. CH 通信范围和数据采集（按时间戳先进先出）；
5. 固定 Sink 的 FCFS 排队；
6. 数据包级上传完成时间和价值账本（含数据量项）；
7. 全局事件队列；
8. **守恒单元测试**【v1.1】：账本总价值 = 各 AUV 按交付者归属的奖励之和；携带状态记账 = 逐包求和；$\omega_v=1$ 与 $\omega_I=1$ 两个退化情形的总价值等于送达包数 / 纯时效价值。

### 阶段二：预测模块

1. Gamma-Poisson 在线更新（向先验遗忘）；
2. 验证速率估计；
3. 未来生成量预测（窗口 = 报告年龄 + 航行时间）；
4. 到达时缓存量和溢出概率预测（负二项逐项求和）；
5. 候选任务的 Sink 交付价值预测（6.4 闭式）；
6. 评估预测误差与置信区间（按访问次数分组）。

### 阶段三：异步 MAPPO

1. 单 AUV 宏动作和事件触发（Visit + Return）；
2. 多 AUV 与 CH 本地服务锁；
3. 每台 AUV 独立轨迹缓存；
4. 参数共享 Actor（含 AUV 编号）；
5. 集中式 Critic（条件于 AUV 编号）；
6. 持续时间感知 TD 与 GAE；
7. PPO-Clip 更新；
8. 验证旧策略 log-prob 与 rollout 参数一致。

### 阶段四：完整实验

1. 预测精度；
2. 预测输入消融（含 H1、H2 参考线）；
3. 已有论文方法的适配基线；
4. Sink 排队消融（$\rho_S$ 合格才做）；
5. 长期多轮价值对比；
6. 不同 AUV 数量、CH 数量和数据生成模式的扩展实验；
7. 同步／异步 MAPPO 对比（优先级最低）。

初始仿真从 **3 AUV、5 CH、1 个固定 Sink** 开始，验证接口和交付逻辑后再扩展到 14.7 的主实验规模。

### 排期（共 7 周）【v1.1·建议】

按 2026-11-11 起算七周，周界是粗略的。工作量是我的估计，没有实测，进度落后时按"决策门"调整。

| 周 | 日期（约） | 内容 |
| :-: | :-- | :-- |
| 1 | 11/11–11/17 | 阶段一：环境骨架、账本、守恒测试；本方案 v1.1 定稿 |
| 2 | 11/18–11/24 | 阶段二：预测模块与 14.1 评估；启发式基线（NN、Greedy VoI、Urgency、Belief + ratio-greedy、GAAP-style）；调 $\rho_S$ |
| 3 | 11/25–12/01 | SPDC-GA-style；单 AUV 异步 MAPPO 跑通；测出 H1、H2 的成绩，也就是 MAPPO 要超过的线 |
| 4 | 12/02–12/08 | 多 AUV、服务锁、共享 Actor 与集中式 Critic、三层奖励 |
| 5 | 12/09–12/15 | 14.2 消融小规模版；LG-DQN-style；**决策门**（见下） |
| 6 | 12/16–12/22 | 主实验（≥10 CH、≥5 种子）、稀疏版本、$\omega_v$ 敏感性、Sink 排队（若 $\rho_S$ 合格） |
| 7 | 12/23–12/31 | 补实验、整理图表、写作；有余力再做 14.4 |

**决策门（第 5 周末）：** 比较 14.2 的 D 组与 H1。如果 D 没有超过 H1（差距小于多种子波动），第 6 周不再继续调 MAPPO，而是把论文重心放在"问题设定 + 强启发式 + 信息何时有用的系统研究"，RL 作为对照而不是主角。越早知道越好。

**延后项：** MMPP 隐状态滤波；真实数据验证；Wait 与 Collect(τ)（E1）；预约机制与意图共享、CH 黑板；多类别数据包；能耗约束；完整 SPDC（SPDC-full）；同步／异步对比（时间紧时第一个砍）。

## 16. 已确定事项与待确认事项

### 已确定

- [x] 长期多轮数据采集，不是单轮访问；
- [x] CH 持续生成数据；
- [x] 固定 Sink；
- [x] 数据只有在 Sink 成功上传后才兑现价值；
- [x] AUV 进入 CH 通信范围才能获取报告；
- [x] CH 报告主体为当前缓存量和数据包生成时间戳；
- [x] 数据生成模式采用混合模式；
- [x] 预测数据来源为仿真训练、真实数据辅助验证；
- [x] 预测对象包括生成量、缓存量和未来 VoI；
- [x] 预测结果参与目标选择、访问时机和返航决策；
- [x] AUV 不依赖持续直接通信；
- [x] 高层离散任务决策，底层固定三维导航；
- [x] 不考虑 PID 和六自由度控制；
- [x] 主算法采用 Asynchronous MAPPO；
- [x] 参数共享 Actor、集中式 Critic、CTDE；
- [x] 使用异步宏动作和事件驱动决策；
- [x] 【v1.1·用户确认】包价值保留数据量项：$\omega_v=0.3$，$\omega_I=0.7$；
- [x] 【v1.1·用户确认】到达类型先做 S1 固定、S2 分段突变、S3 MMPP 三种，不增加"缓慢变化"；
- [x] 【v1.1·用户确认】基线加入已有论文方法的适配版：GAAP-style、SPDC-GA-style、LG-DQN-style。

### 实现前待明确：v1.1 默认取值

【v1.1·建议】v1.0 列的 10 个事项，各给一个默认取值。不同意的可以按"回退"一列改。

| # | 事项 | v1.1 默认取值 | 回退或备注 |
| :-: | :-- | :-- | :-- |
| 1 | 累计生成包计数器 | 加 $G_j$；缓存满时丢新包；AUV 按时间戳先进先出采集（3.3） | 不加：用最早时间戳法，有混杂，需在仿真里评估偏差 |
| 2 | 混合生成模式的状态数、转移矩阵、速率范围 | S3 两状态；突发速率约为正常的 4–5 倍，突发时间占比约 10%–20%；只作压力测试，不做隐状态滤波 | 若作主线：恢复 14.1 的 MMPP 对比并加开发时间 |
| 3 | 数据包类别、基础价值、衰减系数 | 单类别，$w_p=1$，$\omega_v=0.3,\ \omega_I=0.7$，$\eta$ 用 6.1 的方法标定 | 多类别：携带状态按类别分别记录 |
| 4 | Sink 是否单服务通道 | 是，FCFS | 多通道作扩展 |
| 5 | Sink 排队状态向 AUV 提供的信息 | AUV 到 Sink 附近才看到队列；返航前只有"上次观测值 + 其年龄"；另做 Oracle 对照 | 若 $\rho_S$ 太低，这项消融不做 |
| 6 | 被服务锁拒绝后的重规划 | 立即重新决策；被拒绝的 CH 在一个服务时长内被掩码 | — |
| 7 | 宏动作跨 rollout 边界 | 方案 1：等宏动作结束再更新；周期末终止值取 0，不 bootstrap | 方案 2 作后续优化 |
| 8 | 异步团队奖励的时间归属 | 按交付者归属 + 势函数塑形 + $\kappa=0.1$ 团队项（10.4） | v1.0 的团队奖励作消融 |
| 9 | 时间折扣与 VoI 衰减 | $\gamma=1$，剩余时间放进观测；另做 $\gamma<1$ 的敏感性 | — |
| 10 | 真实水下数据的用途 | 只用于标定参数量级（速率、包长、衰减时间）和合理性检查；第一版不做真实数据验证实验 | 有现成数据再扩展 |

## 17. 最终研究定位摘要

本文研究一种面向长期多轮数据采集的预测增强异步多 AUV 数据收集方法。在有限任务周期内，水下 CH 持续产生具有不同信息价值和时间敏感性的监测数据，而 AUV 受限于水声通信范围，只能通过局部、延迟的 CH 报告获取网络信息。为此，建立在线数据生成速率估计、未来缓存状态预测和 Sink 交付 VoI 预测机制，并将预测结果输入异步 MAPPO 策略，使多 AUV 在局部观测条件下联合优化目标选择、访问时机、采集时长和返航决策（第一版先做目标选择与返航，访问时机与采集时长作为扩展）。进一步考虑固定 Sink 的有限上传服务能力和 FCFS 排队过程，以任务周期内实际送达 Sink 的累计价值（数据量项与时效项加权）最大化作为优化目标。

最终技术路线：

$$
\boxed{
\begin{array}{c}
\text{混合数据生成}
+
\text{Gamma-Poisson 预测}
+
\text{未来交付 VoI}\\
+
\text{Dec-POSMDP}
+
\text{Asynchronous MAPPO}
\end{array}
}
$$

本文件作为当前研究方案的正式基线版本。后续模型、代码和实验应以此为依据；若修改 CH 报告字段、Sink 服务机制、动作空间或算法结构，应同步更新系统模型和实验设计。


---

## 附录 A（v1.1 新增）

### A.1 玩具实验：结果与预期

**这是什么。** 我写的一个很小的仿真：单台 AUV、10 个节点、一个 Sink，用来判断"预测信息到底值多少"。它不是本方案的仿真器（没有多 AUV、服务锁、Sink 排队、缓存溢出），只能说明量级，不能当论文证据。标"本轮重跑"的数字是写 v1.1 时重新跑过的，其余来自此前的运行，没有重跑。

**(a) 预测的价值（此前运行）。**

| 比较 | 结果 |
| :-- | :-- |
| 只看"距上次服务的时间"的规则 vs 知道真实速率的贪心 | 前者达到后者的 91%–98% |
| 用信念（Gamma-Poisson 或网格）相对"只看距上次服务的时间"的增益 | 每个节点被访问 ≥5 次时约 +3–5 个百分点（$\mu=0.005$）；$\mu=0.001$ 时约 +1；$\mu=0.02$ 时约 +3–5；每个节点约 2 次访问时约 0 |
| 静态估计器（不遗忘） | 速率非平稳且访问次数少时，比不用历史还差 |
| 知道真实速率的贪心 vs 能看到真实缓冲区的参考策略 | 约 98%–99.7% |

**(b) 收还是返航：整趟策略对比（本轮重跑）。** 60×60 区域，Sink 在中心，10 个节点，单 AUV，权重 (0.3, 0.7)，$\mu=0.005$，每格 40 个回合，数值是"相对能看到真实缓冲区的参考策略的得分比例"。T=3000 和 9000 分别对应每个节点约 11 次和 34 次服务。

| 场景 | T | 知道真实速率 | 网格信念 | 静态 Gamma-Poisson | 总体均值自适应 | 固定每趟 1 个 | 固定每趟 2 个 | 固定每趟 3 个 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| fixed | 3000 | 0.992 | 0.983 | 0.987 | 0.946 | 0.839 | 0.913 | 0.935 |
| fixed | 9000 | 0.995 | 0.988 | 0.993 | 0.946 | 0.856 | 0.915 | 0.937 |
| slow | 3000 | 0.999 | 0.969 | 0.919 | 0.921 | 0.828 | 0.899 | 0.921 |
| slow | 9000 | 0.996 | 0.983 | 0.943 | 0.924 | 0.849 | 0.902 | 0.921 |
| abrupt | 3000 | 0.993 | 0.970 | 0.932 | 0.945 | 0.848 | 0.917 | 0.934 |
| abrupt | 9000 | 0.995 | 0.984 | 0.956 | 0.954 | 0.865 | 0.926 | 0.945 |
| mixed | 3000 | 0.990 | 0.968 | 0.913 | 0.940 | 0.839 | 0.905 | 0.927 |
| mixed | 9000 | 0.994 | 0.982 | 0.933 | 0.936 | 0.863 | 0.922 | 0.937 |

读法：网格信念加 ratio-greedy 规则大约在 97%–99%；每趟固定访问 3 个节点约 91%–95%（以前在测试种子上调趟长，3–7 个里最好的达 90.4%–94.5%，偏乐观）；每趟只访问 1 个节点最差（83%–87%）；"总体均值自适应"规则平均每趟访问约 4.5–5.4 个节点，而参考策略约 3 个，说明它过度延长。这里的"slow、mixed"是我自己写的场景（"mixed"是突变基线乘以正弦），不是本方案的 S1–S3。这些数字也说明：天花板很高，MAPPO 的提升空间可能很小。

**(c) 预测质量（本轮重跑）。**

- 80% 预测区间的实际覆盖率（每格 40 个回合）：

| 场景 | 静态估计器 | 向先验遗忘（$\nu=0.01$，玩具实验的时间尺度） |
| :-- | :-: | :-: |
| fixed | 0.900 | 0.919 |
| slow | 0.683 | 0.787 |
| abrupt | 0.660 | 0.862 |
| mixed | 0.557 | 0.760 |

- 久未访问后"缓存为空"的概率（后验 Gamma(5,30)，窗口 100，真实值约 $5.8\times10^{-8}$）：

| 遗忘强度 $\nu$ | 向 0 遗忘（v1.0） | 向先验遗忘（v1.1） |
| :-: | :-: | :-: |
| 0.01 | 1.4% | 0.27% |
| 0.02 | 11.1% | 0.52% |
| 0.05 | 81.1% | 0.81% |

不遗忘（静态）时是 $6.5\times10^{-4}$。

**(d) 闭式公式检验（本轮重跑）。** 6.4 的采集部分均值与方差，用蒙特卡洛核对：5 组参数、2 种基础价值分布，最大相对偏差 1.2%；空窗口概率闭式 0.0204 对蒙特卡洛 0.0206；携带状态记账与逐包求和完全一致。脚本：`sim/check_floor_closed_form.py`。

**(e) 滤波器选择（此前运行）。** 速率是一维时，网格滤波、500 粒子的粒子滤波和"BOCPD + 遗忘"三者的预测对数似然基本相当；粒子滤波有采样抖动（预测均值的变异系数在 100、500、2000 粒子时约 7.5%、3.5%、1.7%），网格滤波没有。

### A.2 主要风险（按重要性）

| # | 风险 | 应对 |
| :-: | :-- | :-- |
| 1 | MAPPO 可能赢不了 Belief + ratio-greedy（玩具里天花板 97%–99%） | 第 3 周就测出 H1、H2；第 5 周末的决策门（15 节） |
| 2 | Sink 排队消融可能无差异 | 先按 6.3 调 $\rho_S$；调不出来就不作为贡献 |
| 3 | 范围对 7 周偏大 | 动作空间收缩、延后项清单、同步/异步对比最先砍 |
| 4 | 访问次数太少时预测信息失效 | 主实验保证每个 CH 约 5–30 次访问，并做稀疏版本说明边界 |
| 5 | S2、S3 下预测区间过度自信 | 如实报告覆盖率；隐状态滤波作扩展 |
| 6 | 适配基线被质疑"不是原方法" | 论文里写 X-style (adapted)，列出假设差异，用同一信息和同一奖励比较 |
| 7 | 创新性不足 | 见下表；贡献重心放在问题设定和系统研究，不夸大算法 |

**创新点的诚实评估（我的判断，不是客观结论）：**

| 创新点 | 强度 | 依据 |
| :-- | :-: | :-- |
| ① 新问题设定：连续随机到达、多趟往返、显式返航、送达时按包结算（含数据量项） | 中到强 | 已读文献里，Han 2026 是一次性上报，Xu 2025 每簇单一生成时刻 |
| ② 隐藏缓冲区的信念预测 | 中到弱 | 玩具里增益 3–5 个百分点，且依赖访问次数 |
| ③ 结构化的异步 MAPPO 设计 | 弱到中 | 取决于 14.2 消融；异步 MAPPO 本身不是原创 |
| ④ 信息共享（黑板） | 弱 | 第一版不做 |
| ⑤ "信息何时有用"的系统实证 | 中 | 14.1 的访问次数分组和 14.2 的信息阶梯 |

### A.3 未核实项

1. **GAAP。** 我没有读过 GAAP 原文，也不知道它的全称；Essex 的页面没有定义 GAAP。它的机制只来自 Xu 等 2025 第 V-B 节的一句话转述（经网页抽取得到）。14.5 的"按衰减贪心"打分是我的解读。
2. **LG-DQN。** Wang 等 2024 我只有摘要级信息；一个站点被防火墙拦截，AMiner 只返回主页元数据。"local global"的具体含义、状态／动作／奖励都没有核实。"簇头取离簇中心最近的节点、回报为总 VoI"来自 Xu 等的转述。
3. **SPDC-GA。** He 2022（SPDC 原文）我读过全文，它用几何插入式路径规划，不用 GA；"SPDC-GA"来自 Xu 等的转述，是他们的改造版。
4. **Xu 等 2025。** 我只拿到摘要和第 V-B 节的抽取结果，没有精读全文；"不建模生成速率与缓存、每簇一个生成时刻"来自这次抽取，写相关工作前必须核对原文。
5. **Le 等 2026（arXiv:2601.10849）和 arXiv:2410.11282。** 只读过摘要级信息。
6. **Han 2026。** 我读到全文提取文本的约 1300 行，参考文献部分没读。
7. **势函数塑形。** 异步 SMDP、$\gamma=1$ 情形下"不改变最优策略"是我自己的逐轨迹推导，没有找到已发表证明；退路是 Ng 等 1999 和 Lu & Schwartz 2011，外加实验检验。
8. **规则 B（边际采集判据）。** 我自己推导，近视，没有在 MARL 里检验过。
9. **玩具实验。** 自己的代码与参数，不是本方案的环境；A.1 标"此前运行"的数字本轮没有重跑。
10. **默认参数与工作量。** S2、S3 的默认设置、$\nu$、$\kappa$、$\rho_S$ 目标区间、排期工作量都是粗估，需要在实现中标定。

### A.4 术语表

| 术语 | 一句话解释 |
| :-- | :-- |
| CH（簇头） | 汇集附近传感器数据、等 AUV 来取的水下节点 |
| Sink | 固定的汇聚点，AUV 把数据送到这里上传 |
| VoI | 信息价值；这里是"数据越新越值钱，但送达了至少保底" |
| 数据量项 / 时效项 | 价值里不随时间衰减的部分 / 随时间指数衰减的部分 |
| 宏动作 | 要持续一段时间才结束的动作，比如"去某个 CH 并取完数据" |
| 异步 | 每台 AUV 在自己动作结束时才决策，不要求大家同时决策 |
| Dec-POSMDP | 多智能体、局部观测、动作时长不固定的决策模型 |
| CTDE | 训练时用全局信息，执行时只用本地信息 |
| Gamma-Poisson | 用 Gamma 分布表示"数据生成速率不确定"，每次观测后更新 |
| 向先验遗忘 | 很久没更新时，估计慢慢退回到出厂设定，而不是退到"什么都不知道" |
| 势函数塑形 | 给奖励加一个辅助项，让延迟的奖励提前出现，又不改变最优策略 |
| ratio-greedy | 每步选"价值÷用时"最大的动作，返航也是候选 |
| Oracle | 假装知道真实情况的上界对照 |
| 利用率 $\rho_S$ | Sink 上传通道平均有多少时间在忙 |
| 适配基线（X-style） | 把别人的方法改到我们的设定后的版本，不等于原方法 |
| 决策门 | 到某个时间点检查一个关键条件，决定后面往哪走 |

> （注：v1.0 部分内容由豆包 AI 生成；v1.1 的修订由 Claude 协助完成，未核实项见 A.3。）

