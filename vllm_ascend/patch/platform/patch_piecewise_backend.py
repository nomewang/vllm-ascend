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
import torch.fx as fx
from typing import Any


def create_concrete_args_with_fake_tensors(graph: fx.GraphModule, size: int) -> list[Any]:
    """Create example inputs with symbolic dims replaced by a concrete size.
    
    This is a patched version that creates fake tensors instead of real tensors
    to avoid FakeTensorMode mismatch issues during compilation on Ascend NPU.
    
    Used for single-size eager compilation where we need concrete-shaped
    inputs but don't have real runtime tensors yet.
    """
    from torch._prims_common import compute_required_storage_length
    from torch.fx.experimental.symbolic_shapes import is_symbolic
    from torch._guards import detect_fake_mode
    
    def concretize(sym_val: Any) -> int:
        """Replace all symbolic variables in a SymInt expression with size."""
        if not is_symbolic(sym_val):
            return int(sym_val)
        expr = sym_val.node.expr
        return int(expr.subs({s: size for s in expr.free_symbols}))
    
    fake_mode = detect_fake_mode()
    
    args: list[Any] = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        val = node.meta["example_value"]
        if isinstance(val, torch.SymInt):
            args.append(concretize(val))
        elif isinstance(val, torch.Tensor):
            new_shape = tuple(concretize(d) for d in val.shape)
            new_strides = tuple(concretize(s) for s in val.stride())
            new_storage_offset = concretize(val.storage_offset())
            needed_size = compute_required_storage_length(
                new_shape, new_strides, new_storage_offset
            )
            
            if fake_mode is not None:
                with fake_mode:
                    t = torch.empty(needed_size, dtype=val.dtype, device=val.device)
                    t = t.as_strided(new_shape, new_strides, new_storage_offset)
            else:
                t = torch.empty(needed_size, dtype=val.dtype, device=val.device)
                t = t.as_strided(new_shape, new_strides, new_storage_offset)
            args.append(t)
        else:
            args.append(val)
    return args


_is_patched = False


def patch_piecewise_backend():
    """Patch the piecewise_backend to use fake tensors for concrete args."""
    global _is_patched
    if not _is_patched:
        import vllm.compilation.piecewise_backend as piecewise_backend
        piecewise_backend.create_concrete_args = create_concrete_args_with_fake_tensors
        _is_patched = True


patch_piecewise_backend()
