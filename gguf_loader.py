# GGUF loader for Kandinsky models
# Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0

import warnings
import torch
import gguf

from .ops import GGMLTensor

def get_orig_shape(reader, tensor_name):
    metadata_keys = [
        f"comfy.gguf.orig_shape.{tensor_name}",
        f"{tensor_name}.shape",
        f"orig_shape.{tensor_name}",
    ]

    for field_key in metadata_keys:
        field = reader.get_field(field_key)
        if field is not None:
            if len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] == gguf.GGUFValueType.INT32:
                return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

    return None

def infer_shape_from_quant(tensor_shape, qtype, tensor_data=None):
    if qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                 gguf.GGMLQuantizationType.BF16}:
        return torch.Size(tuple(int(v) for v in reversed(tensor_shape)))

    block_size, type_size = gguf.GGML_QUANT_SIZES.get(qtype, (0, 0))

    if block_size > 0 and type_size > 0:
        if tensor_data is not None and len(tensor_shape) == 2:

            actual_bytes = tensor_data.size
            total_blocks = actual_bytes // type_size
            total_elements = total_blocks * block_size

            dim0 = int(tensor_shape[0])

            dim1 = total_elements // dim0

            return torch.Size([dim1, dim0])

        elif len(tensor_shape) >= 2:
            shape_list = list(tensor_shape)
            compressed_bytes = int(shape_list[-1])

            if compressed_bytes < type_size:
                return torch.Size(tuple(int(v) for v in reversed(tensor_shape)))

            num_blocks = compressed_bytes // type_size
            num_elements = num_blocks * block_size
            shape_list[-1] = num_elements

            return torch.Size(tuple(int(v) for v in reversed(shape_list)))

    return torch.Size(tuple(int(v) for v in reversed(tensor_shape)))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")

def detect_model_dims(path):
    reader = gguf.GGUFReader(path)

    dims = {}
    max_text_block_idx = -1
    max_visual_block_idx = -1

    for tensor in reader.tensors:
        tensor_name = tensor.name

        if tensor_name == "time_embeddings.in_layer.weight":
            shape = get_orig_shape(reader, tensor_name)
            has_metadata = shape is not None

            if shape is None:
                shape = infer_shape_from_quant(tensor.shape, tensor.tensor_type, tensor.data)

            dims['time_in_features'] = shape[1] if len(shape) > 1 else None  # model_dim
            dims['time_out_features'] = shape[0] if len(shape) > 0 else None  # time_dim
            dims['has_metadata'] = has_metadata

        if "text_transformer_blocks." in tensor_name:
            parts = tensor_name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_idx = int(parts[1])
                max_text_block_idx = max(max_text_block_idx, block_idx)

        if "visual_transformer_blocks." in tensor_name:
            parts = tensor_name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_idx = int(parts[1])
                max_visual_block_idx = max(max_visual_block_idx, block_idx)

        if "feed_forward.in_proj.weight" in tensor_name and dims.get('ff_dim') is None:
            shape = get_orig_shape(reader, tensor_name)
            if shape is None:
                shape = infer_shape_from_quant(tensor.shape, tensor.tensor_type, tensor.data)
            if len(shape) > 0:
                dims['ff_dim'] = shape[0]

        if tensor_name == "visual_rope_embeddings.freqs_t" and dims.get('head_dim_t') is None:
            shape = get_orig_shape(reader, tensor_name)
            if shape is None:
                shape = infer_shape_from_quant(tensor.shape, tensor.tensor_type, tensor.data)
            if len(shape) > 0:
                dims['head_dim_t'] = shape[0]

        if tensor_name == "out_layer.out_layer.weight":
            shape = get_orig_shape(reader, tensor_name)

            if shape is None:
                shape = infer_shape_from_quant(tensor.shape, tensor.tensor_type, tensor.data)
            if len(shape) > 1:
                out_features = shape[0]
                in_features = shape[1]

                if dims.get('time_in_features') != in_features:
                    dims['time_in_features'] = in_features

                dims['out_layer_out_features'] = out_features

    if max_text_block_idx >= 0:
        dims['num_text_blocks'] = max_text_block_idx + 1
    if max_visual_block_idx >= 0:
        dims['num_visual_blocks'] = max_visual_block_idx + 1

    return dims

def infer_variant_from_dims(dims):
    time_in = dims.get('time_in_features')

    if time_in is None:
        return None

    if time_in <= 2304:
        return "2B variant (sft_5s, sft_10s, i2v_5s, etc.)"
    elif time_in >= 4096:
        return "20B variant (t2v_pro_20b, i2v_pro_20b)"
    else:
        return f"Unknown variant (time_in={time_in})"

def load_gguf_state_dict(path):
    reader = gguf.GGUFReader(path)

    arch_str = get_field(reader, "general.architecture", str)
    if arch_str != "wan":
        raise ValueError(f"Expected 'wan' architecture for Kandinsky GGUF, got '{arch_str}'")

    state_dict = {}
    qtype_dict = {}

    for tensor in reader.tensors:
        tensor_name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = infer_shape_from_quant(tensor.shape, tensor.tensor_type, tensor.data)

            shape_list = list(shape)
            if all(dim > 0 for dim in shape_list):
                while len(shape_list) > 1 and shape_list[0] == 1:
                    shape_list = shape_list[1:]
                while len(shape_list) > 1 and shape_list[-1] == 1:
                    shape_list = shape_list[:-1]
            shape = torch.Size(shape_list)

            if any(dim == 0 for dim in shape):
                shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[tensor_name] = torch_tensor
        else:
            ggml_tensor = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

            state_dict[tensor_name] = ggml_tensor

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    return state_dict
