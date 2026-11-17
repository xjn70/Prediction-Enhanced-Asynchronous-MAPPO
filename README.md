# Prediction-Enhanced Asynchronous MAPPO UWSN

面向多 AUV 水下传感器网络数据采集的 Python 研究项目。当前仿真覆盖 CH 生成、AUV 异步航行与采集、返航及 Sink 包级交付。

| 模块 | 已实现内容 |
| --- | --- |
| `uwsn/entities/` | 不可变数据包、CH FIFO 缓存与服务锁、AUV 存储、Sink 单通道 FCFS 上传及包级 VoI |
| `uwsn/data_generation/` | 固定与分段常数速率 Poisson、两状态 MMPP、生成事件与缓存统计 |
| `uwsn/environment/` | 事件队列、Simulator、CH 报告、异步 AUV 决策就绪、Sink 排队与交付事件、S1–S3 场景构造 |
| `uwsn/navigation/` | 三维直线航行和 CH 通信范围的最近进入点 |
| `uwsn/baselines/` | 固定循环巡航、最近邻、基于历史报告的 VoI 贪心 |
| `uwsn/evaluation/` | 异步策略运行、配对种子比较和交付指标 |
| `tests/` | 实体、随机过程、事件时序、可复现性和包守恒测试 |

AUV 携带数据抵达 Sink 后以实际到达顺序等待单通道 FCFS 服务。同刻到达按事件序号排队。每个包完成上传时才从 cargo 移除并写入交付账本。全部上传完成后，AUV 在任务剩余时间内重新进入决策就绪状态。

## Python 与依赖

要求 **Python 3.10 或更高版本**。当前验证环境为 Python 3.12.7、pytest 7.4.4。
核心运行代码仅使用 Python 标准库，测试使用 pytest 7 或更高版本。

项目使用 `int | None` 类型注解和 `bisect.insort_right(..., key=...)` 等 Python 3.10 功能。

在仓库根目录执行以下命令可安装项目及测试依赖。可编辑安装会直接使用工作目录中的源代码。

```powershell
python -m pip install -e ".[test]"
```

## 运行测试

在仓库根目录使用任一命令运行完整测试。

```powershell
pytest -v
python -m pytest -v
```

`pyproject.toml` 的 `pythonpath = ["."]` 将项目根目录加入测试导入路径。
已有 pytest 的环境也可以直接在源代码目录运行测试。
在 VS Code 中，请选择包含 pytest 的 Python 解释器，并在项目根目录打开终端。

## 运行演示

```powershell
python scripts/run_generation_demo.py
python scripts/run_mmpp_demo.py
python scripts/run_event_demo.py
python scripts/run_auv_collection_demo.py
python scripts/run_sink_delivery_demo.py
```

也可以使用模块入口。

```powershell
python -m scripts.run_generation_demo
```

Poisson 生成演示在 `[0, 120)` 秒内生成 3 个 CH 的数据，包含固定速率、分段速率和零速率区间。
输出每个 CH 的生成包数、累计成功入缓存包数、溢出丢包数、超大包拒绝数、生成总字节数和最终缓存量。

每个 CH 使用自己的 `random.Random(seed)`。在相同运行环境中，相同参数和种子产生相同事件序列。
跨速率区间时保留剩余累计强度，时间段之间的空隙按零速率处理。

Sink 交付演示沿用 75 秒采集场景，以 100 B/s 的上传速率产生排队。输出每个 AUV 的等待时间、包级交付时间、累计 VoI 和任务结束时的包守恒检查。

## Task 07 规则基线实验

```powershell
python scripts/run_baseline_comparison.py
python scripts/run_baseline_comparison.py --scenario S3 --seeds 41 42 --horizon-s 180 --output-dir results/task07-smoke
```

默认使用 S1、3 台 AUV、5 个 CH、500 B/s 的 Sink、900 秒时限和种子 41、42、43。每个种子下三种策略重置同一初始场景，运行器逐包核对 CH 编号、生成时刻和大小组成的生成序列。用 `--seeds 41` 可运行单个配对种子。

| 策略 | 可行访问中的选择方式 |
| --- | --- |
| Cyclic Patrol | 按 CH 编号循环，初始偏移由 AUV 编号确定，返航后继续原巡航序号 |
| Nearest Neighbor | 选距通信球最近的 CH，距离相同时选编号较小者 |
| Observed-VoI Greedy | 对历史报告的可交付 FIFO 前缀估计单位时间 VoI，考虑剩余容量、采集、往返和上传时间。用 `1 / (1 + 报告年龄 / 120 s)` 衰减历史评分。无正分时优先访问最近未知 CH，再回退到最近可行 CH |

局部观测包含本 AUV 状态和 cargo、公开地图与通信速率、该 AUV 上一次宏动作结果和亲自收到的最后一份 CH 报告。`report=None` 代表未知。缓存值保留报告时刻的快照，服务锁冲突时仍能收到现场报告。报告、cargo 和动作选择均无需全局包 ID。

`configs/default_config.py` 配置三种策略共用的返航阈值（75%）、截止预留（30 秒）和 CH 重访冷却（20 秒）。空载时屏蔽返航。访问可行性使用公开几何和至少一个包的传输时间，同时预留现有 cargo 的上传时间。返航至少要来得及上传第一个包。冷却期内无动作时安排未来唤醒，整段剩余时限都不可行时让环境推进至任务截止。仿真器另有第 101 次同刻宏动作提交报错的保护。

默认输出到 `results/task07/`。`episodes.csv` 包含每回合指标和逐 CH 到达次数，`decisions.csv` 包含逐 AUV 的访问、所见报告时刻、采集、返航、上传和未决策原因，`summary.csv` 包含均值、样本标准差和有效样本数。未交付总数含丢弃包，CH 和 AUV 中仍存储的包独立列出。成功访问指至少采完一包，拒绝访问指服务锁冲突。平均 Sink 等待仅统计已开始服务的请求，忙碌时间含截止前的未完成上传。无交付或无服务时的对应平均值留空。

## 复现场景和训练回合

```python
from uwsn.environment import make_scenario

sim = make_scenario("S2", master_seed=41, record_log=False)
```

同一主种子可重建相同回合，下一回合用新的主种子重新构造。`sim.reset()` 重放已有场景的初始随机流。构造器为每个 CH 分别派生速率参数与到包过程的种子，通用管理器保留原有显式种子规则，便于构造同刻事件测试。

| 场景 | 生成模型 |
| --- | --- |
| S1 | 每个 CH 的固定速率独立采样，Gamma 形状参数 2、率参数 10 |
| S2 | 每段重新采样 Gamma 速率，段长为 `T/20` 加均值为 `T/3` 的指数等待时间 |
| S3 | 两状态 MMPP，突发速率为正常速率的 4.5 倍，稳态突发占比为 15% |

默认调试场景为 5 个 CH、3 台 AUV、600 秒。构造器默认开启本 CH 的累计生成计数，消融实验可传 `include_generation_counter=False`。`record_log=False` 停止积累 `event_log`，`step_event()` 和 `run_until()` 继续返回本次处理的事件。逐事件消费可避免一次保留整段返回值。

多个 AUV 同刻就绪时，用 `sorted(sim.decision_ready)` 按 ID 提交决策。新计划会使同一 AUV 的旧事件失效，开始 `Visit` 或 `Return` 时即时移出该 AUV 的就绪标志。

## 模型与统计约定

- 时间单位为秒，位置单位为米，数据大小单位为 bytes，生成速率单位为 packets/s。
- 等长包在 Sink 完成上传时兑现 VoI。`Sink.volume_weight` 默认是 0.3，时效项权重为 `1 - volume_weight`。超过任务截止时间的交付价值为零。
- 恰好在截止时刻完成的上传包计入交付和 VoI。未完成的包留在 AUV cargo，Sink 的 `interrupted_uploads` 保留这些请求的诊断记录。队列、交付账本和中断记录属于仿真内部信息。
- 数据生成采用左闭右开时间区间，每个 CH 的包大小、基础价值和衰减系数由 `PacketParameters` 设置。
- 生成速率由 Poisson 的 `RateSchedule` 或 `MMPPConfig` 定义。每个 MMPP CH 必须显式配置不同的 `random_seed`。
- CH 默认采用 `DROP_NEWEST`，缓存满时优先保留已有包。对比实验可显式使用 `DROP_OLDEST`。
- 累计成功入缓存数包含后来被淘汰的数据包，超大包拒绝与普通缓存溢出分别统计。
- `GenerationManager` 管理的所有 CH 共用递增包 ID。续接已有任务时，`first_packet_id` 应高于历史已使用的 ID。
- AUV 通过 `sim.get_auv_ch_report(auv_id, ch_id)` 读取它最近实际收到的报告。`sim.create_ch_report(ch_id)` 可用于当前时钟下的仿真诊断。
- `sim.generation_manager.true_rate_at(ch_id)` 为 Oracle 和校准提供当前真实速率。MMPP 的显式时刻查询限定在当前缓存区间内。AUV 报告只提供局部缓存信息。

## 许可证

本项目采用 [MIT License](LICENSE)。
