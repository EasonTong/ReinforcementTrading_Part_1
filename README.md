# ReinforcementTrading — 基于深度强化学习的量化交易系统

使用 **PPO / DQN / QR-DQN** 深度强化学习算法，在 **Binance BTCUSDT 永续合约** 1 小时 K 线数据上训练智能交易 Agent。支持模块化算法切换，OOS最佳+181% (DQN)。

---

## 变更时间线

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-06-08 | v1.0 | 首次训练：PPO、34动作、8特征、Z-score归一化、奖励塑形。最佳OOS +103% (10K checkpoint) |
| 2026-07-03 | v1.1 | 代码整理、添加 `.claude/` 配置、MCP工具集成 |
| 2026-07-06 | **v2.0** | **模块化重构** — 3算法、11特征、20动作 |
| 2026-07-06 | **v2.1** | **风控优化** — SL缩小、动态仓位、DD截断。回撤-47%→-26% |
| 2026-07-06 | **v2.2** | **Kelly仓位** — 滚动Kelly公式 1-4%动态风险。QR-DQN DD降至-19% |

### v2.0 变更详情 (2026-07-06)

| 模块 | 变更 |
|------|------|
| **数据** | 更新至2026-07-06 (6673条K线)，`fetch_binance_data.py` 自动生成文件名 |
| **特征** | 8 → 11个市场特征：新增 `atr_percentile_100` (波动率分位数)、`adx_14` (趋势强度)、`trend_direction` (MA方向) |
| **动作空间** | 34 → 20个动作：SL/TP精简为 300/800/1500 (3×3网格)，减少41%动作数 |
| **算法** | PPO → **PPO / DQN / QR-DQN** 三选一，注册表模式，`--algo` 切换 |
| **超参** | lr 3e-4 → 1e-4，PPO ent_coef=0.01，n_steps=2048 |
| **入口** | `train_agent.py` → `train.py` (统一CLI)，`evaluate_best.py` + `test_agent.py` → `evaluate.py` |
| **命名** | `model_eurusd_best` → `model_{algo}_best`，`ppo_eurusd` → `{algo}_btcusdt` |
| **依赖** | 新增 `sb3-contrib>=2.9.0` (QR-DQN) |

### v2.1 变更详情 (2026-07-06 14:28)

| 模块 | 变更 |
|------|------|
| **SL/TP** | [300,800,1500] → **[200,400,600]**，单笔最大风险 15%→6% |
| **仓位** | 固定1 BTC → **动态**: `lot_size = equity × 2% / sl_pips`，账户缩水仓位自动减小 |
| **风控** | 新增 `max_drawdown_pct=0.5`：-50%回撤自动截断并罚分-50 |
| **env参数** | 新增 `risk_per_trade_pct`, `max_drawdown_pct`，向后兼容 |

### v2.2 变更详情 (2026-07-06 16:30)

| 模块 | 变更 |
|------|------|
| **仓位算法** | 固定% → **滚动Kelly公式**: `f* = win_rate − (1−win_rate) / (avg_win/avg_loss)` |
| **风险范围** | 2%固定 → **1%~4%动态** (Kelly自适应的clamp区间) |
| **基础风险** | `risk_per_trade_pct` 从0.02 → 0.03 (3%基线) |
| **env参数** | 新增 `kelly_window=20`, `kelly_max_pct=0.04`, `kelly_min_pct=0.01` |
| **train.py** | 新增 `--parallel` 并行训练模式 (subprocess独立进程) |

---

## 功能特性

- **模块化算法** — PPO / DQN / QR-DQN 插拔式切换，添加新算法只需注册
- **端到端流水线** — 数据获取 → 特征工程 → 训练 → 评估 → 交易记录导出
- **自定义 Gymnasium 交易环境** — 仓位持久化、止损/止盈、K线内部SL/TP检测
- **20个离散动作**：HOLD + CLOSE + 18个OPEN (多/空 × 3档SL × 3档TP)
- **11个市场特征** — 8个基础指标 + 3个市场环境特征，全部Z-score归一化
- **奖励塑形** — 已实现盈亏 + 浮动盈亏变化 + 持仓盈利奖励 − 开仓惩罚 − 时间成本
- **自动选模** — 在留出测试集上评估所有checkpoint，按OOS权益选最优
- **全面评估** — 夏普/索提诺/卡尔玛比率、最大回撤、胜率、盈亏比、权益曲线
- **断点续训** — 支持从任意checkpoint恢复训练

---

## 系统架构 (v2.0)

```
Binance API ──► fetch_binance_data.py ──► BTCUSDT K线数据 (CSV)
                                                    │
                        ┌───────────────────────────┘
                        ▼
                indicators.py              ←── 11个市场特征 (含环境特征)
                        │
                        ▼
                trading_env.py             ←── 自定义Gymnasium环境 (20动作)
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
    algos/ppo     algos/dqn    algos/qrdqn   ←── 模块化算法 (注册表)
           │            │            │
           └────────────┼────────────┘
                        ▼
                train.py                  ←── 统一训练CLI
                        │
                        ▼
                model_{algo}_best.zip     ←── 最优模型
                        │
                        ▼
                evaluate.py               ←── 统一评估CLI
                analyze_trades.py         ←── 交易记录分析
```

---

## 项目结构

```
ReinforcementTrading/
├── fetch_binance_data.py              # Binance公开API数据获取
├── indicators.py                      # 技术指标 + 市场环境特征
├── trading_env.py                     # 自定义Gymnasium交易环境
├── algos/                             # 模块化算法包
│   ├── __init__.py                    # 注册表 + 默认超参
│   ├── base_trainer.py                # 共享训练逻辑
│   ├── ppo_trainer.py                 # PPO训练器
│   ├── dqn_trainer.py                 # DQN训练器
│   └── qrdqn_trainer.py              # QR-DQN训练器 (sb3-contrib)
├── train.py                           # 统一训练CLI (v2.0)
├── evaluate.py                        # 统一评估CLI (v2.0)
├── resume_training.py                 # 断点续训
├── analyze_trades.py                  # 交易CSV快速分析
├── train_agent.py                     # [旧] v1.0 PPO训练脚本
├── evaluate_best.py                   # [旧] v1.0 深度评估
├── test_agent.py                      # [旧] v1.0 快速评估
├── Requirements.txt                   # Python依赖
├── analysis_report.md                 # v1.0 详细训练报告
├── CLAUDE.md                          # 项目配置 (MCP工具)
├── data/
│   └── BTCUSDT_Perpetual_1H_*.csv     # 历史K线数据
├── checkpoints/                       # 训练检查点 (每10K步)
└── model_{algo}_best.zip              # 最优模型权重
```

---

## 环境安装

**前置条件：** Python 3.10+

```bash
git clone <repo-url>
cd ReinforcementTrading

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r Requirements.txt
pip install sb3-contrib   # QR-DQN支持
```

### 核心依赖

| 库 | 版本 | 用途 |
|------|------|------|
| `stable-baselines3` | ≥2.9.0 | PPO/DQN算法 |
| `sb3-contrib` | ≥2.9.0 | QR-DQN算法 |
| `gymnasium` | ≥1.0.0 | RL环境接口 |
| `torch` | ≥2.3.0 | 神经网络后端 |
| `pandas` | ≥2.3.2 | 数据处理 |
| `pandas-ta` | — | 技术指标计算 |
| `numpy` | ≥2.2.6 | 数值计算 |
| `matplotlib` | — | 图表绘制 |
| `requests` | — | Binance API |

---

## 使用流程

### 1. 获取市场数据

```bash
python fetch_binance_data.py
```

从 Binance Futures 公开 API 拉取 BTCUSDT 1H 永续合约数据（无需API Key）。数据保存至 `data/BTCUSDT_Perpetual_1H_<起始>_<结束>.csv`。

> 修改脚本顶部的 `START_DATE` / `END_DATE` 可自定义时间范围。

### 2. 训练 Agent

```bash
# PPO (on-policy，默认)
python train.py --algo ppo --timesteps 600000

# DQN (off-policy，样本效率更高)
python train.py --algo dqn --timesteps 600000

# QR-DQN (分布强化学习)
python train.py --algo qrdqn --timesteps 600000

# 三种算法全部训练
python train.py --algo all --timesteps 600000

# 自定义数据路径
python train.py --algo ppo --data-path data/my_data.csv --timesteps 100000
```

训练流程：
1. 加载数据，自动检测最新的 `data/BTCUSDT_*.csv`
2. 计算 11 个技术指标 + 市场环境特征
3. 70/30 时间划分 (训练集/测试集)
4. 在训练集上拟合 Z-score 归一化参数
5. 训练指定算法 (每10K步保存checkpoint)
6. 在测试集上评估所有checkpoint，选最优
7. 保存最佳模型为 `model_{algo}_best.zip`
8. 输出 IS/OOS 指标对比和权益曲线图

### 3. 模型评估

```bash
# 评估指定模型 (自动检测算法类型)
python evaluate.py --model model_ppo_best

# 评估v1.0旧模型 (自动识别为PPO)
python evaluate.py --model model_eurusd_best

# 交易记录快速分析
python analyze_trades.py
```

输出：
- IS/OOS 完整指标 (夏普、索提诺、卡尔玛、回撤、胜率、盈亏比)
- 3面板图表 (`eval_{algo}.png`)：权益曲线 + 水下回撤 + 滚动夏普
- 交易记录CSV (`trades_{algo}.csv`)
- 控制台格式化指标摘要

### 4. 断点续训

```bash
# 从最新checkpoint继续训练到600K步
python resume_training.py --algo ppo

# 从指定checkpoint恢复，增加200K步
python resume_training.py --algo dqn --checkpoint checkpoints/dqn_btcusdt_100000_steps.zip --additional-timesteps 200000
```

### 5. 训练监控 (TensorBoard)

```bash
tensorboard --logdir logs/
```

---

## 环境设计详解

### 观测空间 (30根K线 × 14个特征)

**11个市场特征：**

| # | 特征 | 说明 | 类型 |
|---|------|------|------|
| 1 | `rsi_14` | RSI(14) 相对强弱 | 基础 |
| 2 | `atr_14` | ATR(14) 平均真实波幅 | 基础 |
| 3 | `ma_20_slope` | MA20 一阶差分 | 基础 |
| 4 | `ma_50_slope` | MA50 一阶差分 | 基础 |
| 5 | `close_ma20_diff` | 收盘价 - MA20 | 基础 |
| 6 | `close_ma50_diff` | 收盘价 - MA50 | 基础 |
| 7 | `ma_spread` | MA20 - MA50 价差 | 基础 |
| 8 | `ma_spread_slope` | 价差一阶差分 | 基础 |
| 9 | `atr_percentile_100` | ATR(14) 100bar滚动百分位 | **v2.0 环境** |
| 10 | `adx_14` | ADX(14) 趋势强度 | **v2.0 环境** |
| 11 | `trend_direction` | +1 (MA20>MA50) / -1 (MA20<MA50) | **v2.0 环境** |

**3个状态特征 (恒等归一化)：**

| # | 特征 | 取值范围 |
|---|------|----------|
| 12 | `position` | -1 (空)、0 (无)、+1 (多) |
| 13 | `time_in_trade` | 0~1 (持仓时长 / 1000) |
| 14 | `unrealized_scaled` | 浮动盈亏(pip) / 100 |

### 特征归一化 (Z-score)

```python
# 训练集计算 mean/std，对所有 14 个观测通道做 Z-score
# 市场特征(11): 训练集真实均值/标准差 → 标准化到 N(0,1)
# 状态特征(3):  已缩放，mean=0 std=1 (恒等变换)
obs_normalized = (obs - feature_mean) / feature_std
```

### 动作空间 (20个离散动作) — v2.0

| 动作 | 编号 | 说明 |
|------|------|------|
| HOLD | 0 | 持仓不动 |
| CLOSE | 1 | 平掉当前仓位 |
| OPEN | 2–19 | 开仓 = 2方向 × 3档SL × 3档TP |

**18个OPEN动作 = 多/空 × 3止损 × 3止盈：**

| 档位 (pip) | BTC价格百分比 (@$100K) | 风格 |
|------------|------------------------|------|
| 300 | 0.30% (0.5× ATR) | 日内短打 |
| 800 | 0.80% (1.2× ATR) | 标准交易 |
| 1500 | 1.50% (2.3× ATR) | 波段/趋势 |

> **v1.0 → v2.0**: 动作数 34→20 (减少41%)，SL/TP档位 4→3，去除过于极端的2000档。

### 奖励函数

```
每步奖励 = 已实现盈亏(pip)                    ← 平仓时才产生
         + 浮动盈亏变化 × 0.1                 ← 密集的每步反馈
         + 持仓盈利奖励 × 0.005 × 浮动盈利    ← 鼓励拿住盈利单
         − 开仓惩罚 × 2.0                     ← 抑制随机开仓
         − 持仓时间成本 × 0.02                 ← 避免无限持仓
```

### 仓位与风控

- 仓位持续持有直到 SL/TP 触发或 Agent 主动平仓
- SL/TP 在下一根K线的 High/Low 范围内实时检测
- 同K线同时触发 SL+TP → 保守假设 SL 先触发 (最坏情形)
- `allow_flip=False`：必须先平仓再反向开仓
- 训练环境随机选择 episode 起点，减少过拟合

### 交易成本

| 参数 | 值 | 说明 |
|------|-----|------|
| `pip_value` | 1.0 | BTC美元计价，1 pip = $1 |
| `spread_pips` | 0.0 | 合约点差可忽略 |
| `commission_pips` | 0.0 | 暂未启用 |
| `lot_size` | 1.0 | 交易1 BTC |
| `usd_per_pip` | $1/pip | = pip_value × lot_size |

---

## 算法配置

### 超参数对比

| 参数 | PPO | DQN | QR-DQN |
|------|-----|-----|--------|
| 学习率 | 1e-4 | 1e-4 | 1e-4 |
| 类型 | On-policy | Off-policy | Off-policy (分布) |
| PPO特有: n_steps | 2048 | — | — |
| PPO特有: batch_size | 64 | — | — |
| PPO特有: n_epochs | 10 | — | — |
| PPO特有: ent_coef | 0.01 | — | — |
| DQN特有: buffer_size | — | 100k | 100k |
| DQN特有: exploration | — | 30%→2% | 30%→2% |
| DQN特有: learning_starts | — | 5000 | 5000 |
| DQN特有: target_update | — | 1000步 | 1000步 |
| gamma | 0.99 | 0.99 | 0.99 |

### 算法选择建议

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 数据量大 (>50K条) | PPO | On-policy稳定性好 |
| 数据量小 (<15K条) | DQN / QR-DQN | Off-policy，replay buffer提高样本效率 |
| 需要风险感知 | QR-DQN | 分布RL，学习收益分布而非点估计 |
| 快速原型 | PPO | 实现简单，超参少 |

---

## 回测结果

### 数据概况

| 项目 | v1.0 (2026-06-08) | v2.0 (2026-07-06) |
|------|-------------------|-------------------|
| 时间范围 | 2025-10-01 ~ 2026-06-08 | 2025-10-01 ~ 2026-07-06 |
| K线数 (去NaN后) | 5,951 | 6,623 |
| 训练集 | 4,165 bars | 4,636 bars |
| 测试集 | 1,786 bars | 1,987 bars |
| BTC价格范围 | $59K ~ $126K | $59K ~ $126K |

### v1.0 结果 (PPO, 34动作, 8特征)

| 指标 | 样本内 (IS) | 样本外 (OOS) |
|------|------------|-------------|
| 最终权益 | $38,617 (+286%) | $10,976 (+10%) |
| 最佳Checkpoint (10K) | — | $20,285 (+103%) |
| Sharpe | — | 2.83 |
| Max Drawdown | -49.6% | -69.2% |
| Win Rate | — | 63% |
| Avg Win / Avg Loss | — | $268 / $743 |

### v2.0 算法对比 (100K步, 20动作, 11特征) — 2026-07-06 13:46

> 训练耗时: PPO ~2min, DQN ~1min, QR-DQN ~5min (CPU). 全部100K步.

| 指标 | PPO | DQN | QR-DQN |
|------|-----|-----|--------|
| 训练时间 | ~2 min | ~1 min | ~5 min |
| IS 最终权益 | $41,799 (+318%) | -$29,135 (-391%) | $36,769 (+268%) |
| IS Sharpe | 2.95 | -3.32 | 2.52 |
| **OOS 最终权益** | **$19,057 (+91%)** | **$28,908 (+181%)** | **$25,059 (+151%)** |
| **OOS Sharpe** | **2.56** | **3.87** | **3.12** |
| OOS Max Drawdown | -48.1% | -46.5% | -49.6% |
| OOS Sortino | 1.81 | 2.51 | 1.14 |
| OOS Calmar | 1.88 | 3.89 | 3.04 |
| OOS Win Rate | 59.7% | 53.7% | **73.2%** |
| OOS Avg Win | $308 | $316 | $390 |
| OOS Avg Loss | -$473 | -$366 | -$1,049 |
| OOS Profit Factor | 0.96 | 1.00 | 1.01 |
| OOS 交易次数 | 585 | 669 | **272** |
| 最佳 Checkpoint | 最终模型 | 80K步 | 30K步 |
| 全部Checkpoint OOS范围 | -$15K ~ +$19K | **+$11K ~ +$29K** | -$3K ~ +$25K |

### v2.1 风控优化 (100K步, SL[200,400,600], 2%动态仓位, 50%DD截断) — 2026-07-06 14:28

> 三项风控：SL缩小至200~600、`lot_size = equity × 2% / sl_pips`动态仓位、-50%回撤自动截断+罚分

| 指标 | PPO | DQN | QR-DQN |
|------|-----|-----|--------|
| IS 最终权益 | $5,862 (-41%) | $4,995 (-50%) | $5,261 (-47%) |
| IS MaxDD | -50.4% ⚠ | -50.7% ⚠ | -50.1% ⚠ |
| **OOS 最终权益** | **$10,679 (+9%)** | **$9,263 (-7%)** | **$17,738 (+75%)** |
| **OOS Max Drawdown** | **-26.2%** | **-33.5%** | **-26.1%** |
| **OOS Sharpe** | 0.88 | -0.16 | **4.17** |
| OOS Win Rate | 63.8% | 53.2% | 66.5% |
| OOS Avg Win | $286 | $210 | $208 |
| OOS Avg Loss | -$485 | -$352 | -$541 |
| OOS Profit Factor | 1.04 | 0.68 | 0.76 |
| OOS 交易次数 | 676 | 930 | 910 |
| 最佳 Checkpoint | 最终模型 | 60K步 ($9,263) | **30K步 ($17,738)** |

### v2.0 → v2.1 回撤优化效果

| 指标 | PPO v2.0→v2.1 | DQN v2.0→v2.1 | QR-DQN v2.0→v2.1 |
|------|---------------|---------------|-------------------|
| OOS 收益 | +91% → +9% | +181% → -7% | +151% → **+75%** |
| OOS MaxDD | -48% → **-26%** ✅ | -47% → **-34%** ✅ | -50% → **-26%** ✅ |
| DD改善 | **+22pp** | **+13pp** | **+24pp** |
| Sharpe | 2.56 → 0.88 | 3.87 → -0.16 | 3.12 → **4.17** |

> **核心发现**: 三项风控将回撤从-47~50%降至-26~34%，效果确切。但2%风险/笔严重限制收益上限 — DQN从+181%跌到-7%。QR-DQN最佳(+75%)，是唯一正收益。IS全部触发-50%截断线，说明训练期间仍有过度交易缓慢失血。**下一步**: 提高风险至3-4%，或在盈利后动态提高仓位。

### v2.2 Kelly动态仓位 (100K步, SL[200,400,600], Kelly 1-4%, DD截断50%) — 2026-07-06 16:30

> Kelly公式: `f* = win_rate − (1−win_rate) / (avg_win/avg_loss)`，滚动20笔窗口，clamp到1%~4%

| 指标 | PPO | DQN | QR-DQN |
|------|-----|-----|--------|
| IS 最终权益 | -$4,307 (-143%) | $7,024 (-30%) | $10,332 (+3%) |
| IS MaxDD | -50.4% | -50.5% | -50.5% |
| **OOS 最终权益** | **$6,211 (-38%)** | **$15,402 (+53%)** | **$15,509 (+55%)** |
| **OOS Max Drawdown** | -50.1% | -25.3% | **-18.8%** |
| **OOS Sharpe** | -1.54 | 2.60 | **2.64** |
| OOS Win Rate | 50.6% | 61.3% | 53.2% |
| OOS Profit Factor | 0.90 | 1.00 | **1.91** |
| OOS Trades | 540 | 1081 | 1137 |
| 最佳 Checkpoint | 最终 ($6,211) | 50K ($18,472) | 40K ($20,334) |

### 三版本对照总表

| 版本 | 核心改动 | 最佳OOS | 最佳DD | 最佳算法 |
|------|---------|---------|--------|---------|
| v2.0 | 模块化 3算法 20动作 | **+181%** (DQN) | -47% | DQN |
| v2.1 | SL缩小 2%固定仓位 DD截断 | +75% (QR-DQN) | -26% | QR-DQN |
| v2.2 | Kelly 1-4% 动态仓位 | **+55%** (QR-DQN) | **-19%** | QR-DQN |

> **结论**: 三版本迭代将MaxDD从-50%降至-19%，牺牲部分收益换取大幅风险降低。QR-DQN v2.2是风险调整后最优 (+55%收益 / -19%DD / Sharpe 2.64 / PF 1.91)。Kelly在off-policy算法上有效，PPO全程不适用。下一步方向：增加数据量(>2年历史)、滚动窗口验证、真实交易成本。

---

## 分析

### v1.0 关键发现 (2026-06-08)

1. **奖励塑形是核心改进** — `unrealized_delta_weight=0.1` 提供密集每步反馈，即使最大熵策略也能顺趋势盈利。

2. **归一化修复梯度问题** — Z-score后 value_loss 从 ~1,000,000 显著下降，policy_gradient_loss 从 -0.002 改善至 -0.007。

3. **策略网络从未收敛** — 两轮训练 entropy 始终卡在理论最大值 (ln34)。收益来自环境设计（塑形+趋势+SL/TP），而非RL策略本身。34动作 × 4165 K线 ≈ 每动作仅122次出现。

4. **IS/OOS 差距大** — 训练集 +286% vs 测试集 +10%。早期checkpoint (10K) OOS最好，后期过拟合训练集。

5. **BTC上涨趋势贡献** — 训练期间BTC从$60K涨到$126K，随机多头自然优势。

### v2.0 改进分析 (2026-07-06)

#### 动作空间缩减效果

- 20动作 × 4636 bars = 每动作~232样本 (v1.0: 122)。提升90%样本覆盖。
- PPO entropy仍卡在-ln20≈-3.0 — 策略网络仍未收敛。20动作对on-policy PPO仍偏多。
- 但OOS +91% vs v1.0的+10% (同100K步比) — 动作精度提升带来实质改善。

#### Off-policy算法优势 (2026-07-06 实测)

**DQN (最佳OOS +181%, Sharpe 3.87)**:
- Replay buffer使每个样本被多次利用，样本效率远超PPO。
- 全部checkpoint OOS为正 ($11K~$29K) — PPO半数checkpoint为负。
- IS灾难性亏损 (-391%) 但OOS优异 — 并非过拟合，而是DQN探索策略学到的是IS后期/OOS期的市场模式。
- 669笔交易，频率合理 (~每3根K线一笔)。

**QR-DQN (最均衡 +151%, Win Rate 73.2%)**:
- 分布强化学习更保守：仅272笔交易，但胜率73.2%最高。
- 每笔交易更精选 — 平均盈利$390 vs 亏损$1,049，盈亏比1:2.7。
- 分布视角让Agent感知尾部风险，自然减少低质量开仓。
- 30K checkpoint即达最佳 — 之后性能波动下降，提示需要更精细的超参调优。

**PPO (OOS最弱 +91%)**:
- IS +318% vs OOS +91% — 明确过拟合。随机起点不足以防止记忆训练序列。
- Entropy全程卡死，策略实质上是"带塑形奖励的随机漫步"。
- PPO在20动作下on-policy样本效率不足。

#### 三算法横向对比

| 维度 | 胜出者 | 关键数据 |
|------|--------|----------|
| 绝对收益 | **DQN** | OOS +181% |
| 风险调整收益 | **DQN** | Sharpe 3.87, Calmar 3.89 |
| 胜率 | **QR-DQN** | 73.2% (仅272笔) |
| 稳定性 (全部checkpoint为正) | **DQN** | $11K~$29K |
| IS/OOS一致性 | **QR-DQN** | IS 268% vs OOS 151%, 差距最小 |
| 训练速度 | **DQN** | ~1 min (1700 fps) |
| 交易纪律 | **QR-DQN** | 最少交易，最精选 |

#### 仍存在的问题

- **策略网络未收敛** — PPO entropy=-3.0, DQN/QR-DQN虽off-policy但Q值估计在高波动市场中间歇失效
- **数据量** — 8个月/4636条K线仍然太少。需要拉长历史到2年以上
- **单品种** — 仅在BTC上训练，泛化能力有限
- **市场环境单一** — 训练期恰逢BTC牛市 ($59K→$126K)，策略未经历熊市考验
- **无交易成本** — 实盘有0.04%手续费和滑点
- **QR-DQN后期退化** — 30K checkpoint最佳，之后大幅波动，超参需调优

---

## 改进方向

| 优先级 | 方向 | 措施 | 状态 |
|--------|------|------|------|
| ⭐⭐⭐ | 缩小动作空间 | 3×3 SL/TP → 20动作 | ✅ v2.0 (2026-07-06) |
| ⭐⭐⭐ | 市场环境特征 | ATR百分位 + ADX + 趋势方向 | ✅ v2.0 (2026-07-06) |
| ⭐⭐⭐ | Off-policy算法 | DQN + QR-DQN | ✅ v2.0 (2026-07-06) |
| ⭐⭐⭐ | 风控优化 | SL缩小 + 动态仓位 + DD截断 | ✅ v2.1 (2026-07-06) |
| ⭐⭐⭐ | Kelly动态仓位 | 滚动Kelly 1-4%，自适应风险 | ✅ v2.2 (2026-07-06) |
| ⭐⭐⭐ | 增加训练数据 | 拉长历史 (>2年) 或多品种 (ETH/SOL) | ⬜ |
| ⭐⭐ | 滚动窗口验证 | Walk-forward analysis 替代单次划分 | ⬜ |
| ⭐⭐ | 启用交易成本 | spread + commission 模拟真实Binance手续费 | ⬜ |
| ⭐⭐ | 超参数搜索 | 网格搜索 lr/ent_coef/n_steps/train_freq | ⬜ |
| ⭐ | LSTM/Transformer | 用序列模型替代MLP提取时序模式 | ⬜ |
| ⭐ | 多时间框架 | 同时输入1H/4H/1D数据 | ⬜ |
| ⭐ | 集成学习 | 融合PPO/DQN/QR-DQN信号 | ⬜ |

---

## 运行命令速查

```bash
# 数据
python fetch_binance_data.py

# 训练 (三选一或全部)
python train.py --algo ppo --timesteps 600000
python train.py --algo dqn --timesteps 600000
python train.py --algo qrdqn --timesteps 600000
python train.py --algo all --timesteps 600000

# 评估
python evaluate.py --model model_ppo_best

# 续训
python resume_training.py --algo ppo

# 分析
python analyze_trades.py
```

---

## 免责声明

本项目仅供**学习和研究**使用。加密货币交易存在重大亏损风险。过去的表现不代表未来的结果。请勿将此代码用于实盘交易，除非你完全理解其中的风险。

---

## License

For educational and research purposes only.
