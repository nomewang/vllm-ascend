#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import torch

_original_empty_cache = None


def _npu_empty_cache():
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def patch_torch_accelerator_empty_cache():
    global _original_empty_cache

    if _original_empty_cache is not None:
        return

    if hasattr(torch, 'accelerator') and hasattr(torch.accelerator, 'empty_cache'):
        _original_empty_cache = torch.accelerator.empty_cache
        torch.accelerator.empty_cache = _npu_empty_cache


patch_torch_accelerator_empty_cache()
