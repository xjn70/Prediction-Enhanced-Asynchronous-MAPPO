# Prediction-Enhanced Asynchronous MAPPO UWSN

面向多 AUV 水下传感器网络数据采集的 Python 研究项目，当前已实现核心实体层和 CH 数据生成机制。

| 模块 | 已实现内容 |
| --- | --- |
| `uwsn/entities/` | 不可变数据包、CH FIFO 缓存与服务锁、AUV 存储、Sink 单通道 FCFS 上传及包级 VoI |
| `uwsn/data_generation/` | 固定与分段常数速率 Poisson 过程、多 CH 独立随机流、生成事件处理与缓存统计 |
| `tests/` | 实体行为、随机过程、可复现性和缓存统计测试 |

仿真环境、预测模块和异步 MAPPO 属于后续开发阶段。

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

## 运行数据生成演示

```powershell
python scripts/run_generation_demo.py
```

也可以使用模块入口。

```powershell
python -m scripts.run_generation_demo
```

演示在 `[0, 120)` 秒内生成 3 个 CH 的数据，包含固定速率、分段速率和零速率区间。
输出每个 CH 的生成包数、累计成功入缓存包数、溢出丢包数、超大包拒绝数、生成总字节数和最终缓存量。

每个 CH 使用自己的 `random.Random(seed)`。在相同运行环境中，相同参数和种子产生相同事件序列。
跨速率区间时保留剩余累计强度，时间段之间的空隙按零速率处理。

## 模型与统计约定

- 时间单位为秒，位置单位为米，数据大小单位为 bytes，生成速率单位为 packets/s。
- 等长包在 Sink 完成上传时兑现 VoI。`Sink.volume_weight` 默认是 0.3，时效项权重为 `1 - volume_weight`。超过任务截止时间的交付价值为零。
- 数据生成采用左闭右开时间区间，每个 CH 的包大小、基础价值和衰减系数由 `PacketParameters` 设置。
- 生成速率由 Poisson 的 `RateSchedule` 或 `MMPPConfig` 定义。每个 MMPP CH 必须显式配置不同的 `random_seed`。
- 累计成功入缓存数包含后来被淘汰的数据包，超大包拒绝与普通缓存溢出分别统计。
- `GenerationManager` 管理的所有 CH 共用递增包 ID。续接已有任务时，`first_packet_id` 应高于历史已使用的 ID。
- 要读取 50 秒的报告，先调用 `sim.run_until(50)`，再调用 `sim.create_ch_report(ch_id)`。`packet_timestamps` 是缓存中各包的生成时间序列。需要本 CH 的累计生成包数时，显式传入 `include_generation_counter=True`。

## 许可证

本项目采用 [MIT License](LICENSE)。
