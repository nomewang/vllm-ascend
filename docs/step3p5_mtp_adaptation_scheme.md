# vLLM-Ascend Step3.5 Multi-Layer MTP 适配方案

## 1. 核心架构分析 (Step3.5 MTP)
参考 `stepfun-ai/vllm` 的实现 (`vllm/model_executor/models/step3p5_mtp.py`)：
- **Step3p5AMultiTokenPredictorLayer**: 包含 `enorm`, `hnorm`, `eh_proj`, `shared_head` (需重命名为 `lm_head`), 以及 `mtp_block` (为 `Step3p5DecoderLayer`)。
- **Step3p5AMultiTokenPredictor**: 容器类，包含 `num_mtp_layers`，通过 `spec_step_idx % self.num_mtp_layers` 管理多层 MTP 层，每层独立处理输入。
- **Step3p5MTP**: 顶层封装，提供 `embed_input_ids`, `forward`, `compute_logits`，并处理权重加载 (替换 `shared_head` 等)。

## 2. Multi-Layer MTP 配置支持
在 vLLM 的 `SpeculativeConfig` 中：
- 需要添加或支持 `enable_multi_layers_mtp` 布尔配置。
- 配合 `method="mtp"` 使用，当 `enable_multi_layers_mtp` 为 True 时，根据模型 config (`num_nextn_predict_layers`) 确定多层数量。
- 在 `__post_init__` 中自动调整 `num_speculative_tokens`（例如不超过 `n_predict`）。

## 3. MultiLayerEagleProposer 实现
在 `vllm-ascend/spec_decode/` 目录下（如 `multi_layer_eagle.py`）：
- 继承现有的 `EagleProposer`，实现 `MultiLayerEagleProposer`。
- **propose 逻辑**：循环 `self.layer_num` 次，逐层调用 `model.forward` 和 `compute_logits`，生成 draft tokens。
- **状态传递**：在层与层之间，将前一层的 `hidden_states` 作为下一层的输入。
- **Metadata**：需要扩展 `MultiLayerEagleMetadata` 记录每一层的 `token_ids`, `hidden_states` 等。

## 4. KV Cache 管理适配
在 `vllm-ascend/patch/platform/patch_kv_cache_interface.py` 或相关接口中：
- 针对 multi-layer MTP，需要调整 KV Cache 分组策略。
- 传统 MTP 可能采用间隔采样 `layers[i::num_groups]`，但 multi-layer MTP 所有 MTP 层应在同一组：`layers[i : i + group_size]`。

## 5. InputBatch 扩展
在 `vllm-ascend/vllm_ascend/worker/v2/input_batch.py` 或对应结构中：
- 为支持多层状态，添加 `cached_len`, `cached_token_ids`, `cached_hidden_states`, `cached_slot_mappings`, `cached_positions` 字段。
- 维度需扩展为 `[max_num_reqs, multi_layer_eagle_num, ...]`，在请求状态管理中增加对应的缓存清理和更新操作。

## 6. GPU Model Runner 集成
在 `vllm-ascend/vllm_ascend/worker/v2/model_runner.py` 或 `v1/model_runner_v1.py`：
- `_init_multi_layer_eagle_cache`: 初始化上述 InputBatch 的多层缓存。
- `_prepare_inputs`: 构建并传递 `MultiLayerEagleMetadata` 到 proposer。
- 根据 `enable_multi_layers_mtp` 选择初始化 `MultiLayerEagleProposer`，并确保 draft tokens 的多层提议逻辑无缝接入 Rejection Sampler。

## 7. vLLM-Ascend 适配实现指南
### 7.1 需要验证的昇腾算子
- **GemmaRMSNorm**: 验证 `vllm_ascend/ops/layernorm.py` 精度和性能。
- **Attention**: 确保 `vllm_ascend/attention/attention_v1.py` 支持 Step3p5Attention。
- **Linear / Fused MoE**: 确保 `vllm_ascend/ops/fused_moe/` 对 Step3.5 的 MoE 结构兼容。

### 7.2 投机解码框架适配指南
1. 创建 `vllm_ascend/patch/worker/patch_step3p5_mtp.py` 进行 Monkey Patch。
2. 实现 `AscendStep3p5MultiTokenPredictorLayer` 覆盖原层，支持昇腾特定算子（如 `rot` 等）。
3. 实现 `MultiLayerEagleProposer` 并注册到 `vllm_ascend/spec_decode/` 体系。

### 7.3 KV Cache 管理适配指南
- 修改 `vllm_ascend/patch/platform/patch_kv_cache_interface.py`，根据 `enable_multi_layers_mtp` 配置，覆盖 `_get_kv_cache_groups_uniform_page_size` 的分组逻辑，确保 MTP 层获得连续的 KV Cache blocks。

### 7.4 测试验证方案
- **单元测试**：针对 `MultiLayerEagleProposer` 编写 UT，验证逐层 hidden states 传递的正确性。
- **集成测试**：使用 Step3.5 真实权重，验证端到端投机解码的 acceptance rate。
- **精度对齐**：使用 `vllm_ascend/tools/send_request.py` 对比单层与多层 MTP 在昇腾上的输出 Logits 精度。
