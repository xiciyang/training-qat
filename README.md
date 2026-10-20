# @xiciyang/training-qat

量化感知训练（QAT）：在**真实 GPU** 上按可选 scheme 训练，产出一个可被量化流程直接消费的 checkpoint，并且**拒绝**把「看起来没掉点」当成验收通过。

DSH（DeepSeek Harness）插件包，同时可作为普通 Node 库使用。

## 它解决的问题

QAT 最常见的两个假象：

1. **模拟精度 ≠ 部署精度**——训练时用的伪量化粒度和真正部署时的格式不一致，训练 loss 好看但落地产物不可用。本包在产物侧断言方案与粒度的对应关系，而不是只看训练曲线。
2. **空操作**——和 SFT/预训练同族：权重指纹缺失或前后相同会被判失败，不记成功。

## 安装

```bash
dsh plugin add github:xiciyang/training-qat
```

要求：

- Node.js >= 20，`engines.dsh >= 0.2.0 < 0.3.0`
- 真实训练需要 NVIDIA GPU + 容器运行时 + 训练镜像；`training_qat_health` 会逐项报告
- Python 框架在 `optionalDependencies` 中显式声明

## 工具面（DSH）

| 工具 | 作用 |
|---|---|
| `training_qat_prepare` | 量化前准备：校准集与测试集**不得相交**，切分与血缘落盘 |
| `training_qat_train` | 训练：scheme/粒度、预算闸门、权重指纹、checkpoint 保留、审计链 |
| `training_qat_validate` | 验收：精度与性能双侧门禁，先判测量有效性再判数值 |
| `training_qat_rollback` | 回退到指定 run；purge 只允许删本插件 run 目录 |
| `training_qat_get_report` | 取人类可读报告 |
| `training_qat_health` | 环境自检：GPU / 容器 / 镜像 / 基座 / 依赖 |

## 配置（不把本机布局写死）

| 环境变量 | 用途 |
|---|---|
| `DSH_BASE_MODEL` | 基座模型本地目录 |
| `TRAINING_QAT_ROOT` | 运行目录根（默认 `~/.ai-infra`） |

量化器镜像等重型依赖缺失时，工具会返回 `E_METHOD_UNAVAILABLE` 并说明缺哪一层，而不是静默降级成一次「成功的训练」。

## 测试

```bash
node tests/run.mjs
```

纯逻辑与契约测试应全绿；依赖本机历史 payload 的判据自检在产物缺失时显式 `SKIP` 并写明原因，不伪装成通过。

## 安全边界

- 训练需显式确认令牌；预算超限直接拒绝
- 回退只作用于本插件自己的 run 目录
- 审计链可第三方复算

## 许可

BSD-3-Clause。见 `LICENSE`。
