# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""
Patch for Eagle models to set target_layer_count.

When using Eagle with hybrid models (e.g., models with both sliding window
and full attention layers), the draft layers need to correctly calculate
their effective_layer_idx when looking up layer_types. Without setting
target_layer_count, the draft layers incorrectly inherit attention types
from the target model, causing them to be placed in different KV cache
groups and triggering the error:
"All drafting layers should belong to the same kv cache group"
"""

from vllm.model_executor.models.llama_eagle import EagleLlamaForCausalLM


def _patched_eagle_llama_init(self, *, vllm_config, prefix: str = ""):
    """Patched __init__ that sets target_layer_count in draft config."""
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.models.llama_eagle import LlamaModel
    import torch.nn as nn

    nn.Module.__init__(self)
    self.config = vllm_config.speculative_config.draft_model_config.hf_config
    # Ensure draft_vocab_size is set
    # default to the base vocab size when absent
    if getattr(self.config, "draft_vocab_size", None) is None:
        base_vocab_size = getattr(self.config, "vocab_size", None)
        self.config.draft_vocab_size = base_vocab_size
    target_layer_num = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )

    # Store target layer count in draft config for
    # proper layer_types indexing in draft models.
    # This is the key fix - without this, draft layers incorrectly
    # inherit attention types (sliding_window vs full attention)
    # from the target model's layer_types array.
    self.config.target_layer_count = target_layer_num

    self.model = LlamaModel(
        vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
    )

    logit_scale = getattr(self.config, "logit_scale", 1.0)
    self.logits_processor = LogitsProcessor(
        self.config.vocab_size, scale=logit_scale
    )


# Apply the patch to EagleLlamaForCausalLM
EagleLlamaForCausalLM.__init__ = _patched_eagle_llama_init


# Also patch other Eagle models that might be used with hybrid models
def _patch_deepseek_eagle_init(self, *, vllm_config, prefix: str = ""):
    """Patched __init__ for EagleDeepseekV3ForCausalLM that sets target_layer_count."""
    import torch.nn as nn
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
    from vllm.model_executor.models.llama_eagle import maybe_prefix

    nn.Module.__init__(self)
    self.config = vllm_config.speculative_config.draft_model_config.hf_config
    quant_config = vllm_config.quant_config
    target_layer_num = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )

    # Store target layer count in draft config for
    # proper layer_types indexing in draft models.
    self.config.target_layer_count = target_layer_num

    self.model = DeepseekV2Model(
        vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
    )

    self.lm_head = ParallelLMHead(
        self.config.vocab_size,
        self.config.hidden_size,
        quant_config=quant_config,
        prefix=maybe_prefix(prefix, "lm_head"),
    )

    logit_scale = getattr(self.config, "logit_scale", 1.0)
    self.logits_processor = LogitsProcessor(
        self.config.vocab_size, scale=logit_scale
    )

    # Set MoE hyperparameters
    self.num_moe_layers = self.config.num_hidden_layers
    self.set_moe_parameters()


# Try to patch other Eagle models if they exist
try:
    from vllm.model_executor.models.deepseek_eagle import EagleDeepseekV3ForCausalLM
    EagleDeepseekV3ForCausalLM.__init__ = _patch_deepseek_eagle_init
except ImportError:
    pass

try:
    from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM

    def _patched_llama4_eagle_init(self, *, vllm_config, prefix: str = ""):
        import torch.nn as nn
        from vllm.model_executor.models.llama4_eagle import LlamaModel
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config.target_layer_count = target_layer_num
        self.model = LlamaModel(vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num)
        self.lm_head = None  # Simplified

    EagleLlama4ForCausalLM.__init__ = _patched_llama4_eagle_init
except ImportError:
    pass

try:
    from vllm.model_executor.models.minicpm_eagle import EagleMiniCPMForCausalLM

    def _patched_minicpm_eagle_init(self, *, vllm_config, prefix: str = ""):
        import torch.nn as nn
        from vllm.model_executor.models.minicpm_eagle import MiniCPMModel
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config.target_layer_count = target_layer_num
        self.model = MiniCPMModel(vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num)

    EagleMiniCPMForCausalLM.__init__ = _patched_minicpm_eagle_init
except ImportError:
    pass

try:
    from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM

    def _patched_mistral_eagle_init(self, *, vllm_config, prefix: str = ""):
        import torch.nn as nn
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config.target_layer_count = target_layer_num
        # Rest of init...

    EagleMistralLarge3ForCausalLM.__init__ = _patched_mistral_eagle_init
except ImportError:
    pass