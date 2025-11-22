import os
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_compatible_fp8_dtype(device):
    if device.type != 'cuda':
        return None

    try:
        major, minor = torch.cuda.get_device_capability(device)
        if (major > 8) or (major == 8 and minor >= 9):
            try:
                test_tensor = torch.ones((2, 2), device=device, dtype=torch.float32)
                test_fp8 = test_tensor.to(torch.float8_e4m3fn)
                _ = (test_fp8.to(torch.float32) * 2.0).sum()
                return torch.float8_e4m3fn
            except (ValueError, RuntimeError):
                pass
        if major >= 8:
            try:
                test_tensor = torch.ones((2, 2), device=device, dtype=torch.float32)
                test_fp8 = test_tensor.to(torch.float8_e5m2)
                _ = (test_fp8.to(torch.float32) * 2.0).sum()
                return torch.float8_e5m2
            except (ValueError, RuntimeError):
                pass
    except Exception:
        pass
    return None


def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval


def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    M = min(max(int(mantissa_bit), 1), bits - sign_bits)
    E = bits - sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = -maxval if sign_bits == 1 else 0.0
    input_clamp = torch.clamp(x, min=minval, max=maxval)
    log_scales = torch.clamp(
        torch.floor(torch.log2(torch.abs(input_clamp) + 1e-12) + bias).detach(),
        1.0
    )
    log_scales = torch.pow(2.0, log_scales - M - bias)
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def fp8_tensor_quant(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    for i in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits)
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(qdq_out, scale, dtype):
    qdq_out = qdq_out.type(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


def fp8_linear_forward(cls, original_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                if scale_weight.device != input.device:
                    scale_weight = scale_weight.to(input.device, non_blocking=True)
                scale_weight = scale_weight.squeeze()
            if not hasattr(cls, '_fp8_input_dtype'):
                compatible_dtype = get_compatible_fp8_dtype(input.device)
                if compatible_dtype is not None:
                    cls._fp8_input_dtype = compatible_dtype
                    try:
                        major, minor = torch.cuda.get_device_capability(input.device)
                        cls._fp8_use_scaled_mm = (major > 8) or (major == 8 and minor >= 9)
                    except:
                        cls._fp8_use_scaled_mm = False
                else:
                    cls._fp8_input_dtype = None
                    cls._fp8_use_scaled_mm = False

            if cls._fp8_use_scaled_mm and cls._fp8_input_dtype is not None:
                try:
                    input_shape = input.shape
                    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
                    input = torch.clamp(input, min=-448, max=448, out=input)
                    inn = input.reshape(-1, input_shape[2]).to(cls._fp8_input_dtype).contiguous()
                    bias = cls.bias.to(original_dtype) if cls.bias is not None else None
                    o = torch._scaled_mm(inn, cls.weight.t(),
                                       out_dtype=original_dtype,
                                       bias=bias,
                                       scale_a=scale_input,
                                       scale_b=scale_weight)
                    return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
                except (ValueError, RuntimeError) as e:
                    cls._fp8_use_scaled_mm = False
            weight_fp32 = cls.weight.to(original_dtype) * scale_weight.to(original_dtype)
            bias = cls.bias.to(original_dtype) if cls.bias is not None else None
            return F.linear(input.to(original_dtype), weight_fp32, bias)
        else:
            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                if scale_weight.device != input.device:
                    scale_weight = scale_weight.to(input.device, non_blocking=True)
                scale_weight = scale_weight.squeeze()

            weight_fp32 = cls.weight.to(original_dtype) * scale_weight.to(original_dtype)
            bias = cls.bias.to(original_dtype) if cls.bias is not None else None
            return F.linear(input.to(original_dtype), weight_fp32, bias)
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, original_dtype, params_to_keep=None, scale_weight_keys=None, device=None):
    if params_to_keep is None:
        params_to_keep = set()

    setattr(module, "fp8_matmul_enabled", True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fp8_dtype = get_compatible_fp8_dtype(device)
    if fp8_dtype is None:
        print("Warning: No compatible FP8 dtype found, skipping FP8 conversion")
        return

    dtype_name = "float8_e4m3fn" if fp8_dtype == torch.float8_e4m3fn else "float8_e5m2"

    fp8_layers = []
    already_fp8 = 0
    kept_layers = 0


    for name, submodule in module.named_modules():
        if any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                kept_layers += 1
            continue

        if isinstance(submodule, nn.Linear):
            fp8_layers.append(name)

            if scale_weight_keys is not None:
                scale_key = f"{name}.scale_weight"
                if scale_key in scale_weight_keys:
                    scale_tensor = scale_weight_keys[scale_key].float()
                    submodule.register_buffer("scale_weight", scale_tensor, persistent=False)

            if submodule.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                already_fp8 += 1
            elif submodule.weight.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] and scale_weight_keys is not None:
                scale_key = f"{name}.scale_weight"
                if scale_key in scale_weight_keys:
                    weight_device = submodule.weight.device
                    submodule.weight = torch.nn.Parameter(
                        submodule.weight.cpu().to(fp8_dtype).to(weight_device)
                    )

            original_forward = submodule.forward
            setattr(submodule, "original_forward", original_forward)
            setattr(submodule, "forward",
                   lambda input, m=submodule: fp8_linear_forward(m, original_dtype, input))


def convert_fp8_linear_on_the_fly(module, original_dtype, params_to_keep=None, use_gpu=True, target_device=None):
    if params_to_keep is None:
        params_to_keep = set()

    setattr(module, "fp8_matmul_enabled", True)

    gpu_available = torch.cuda.is_available() and use_gpu
    if gpu_available:
        quant_device = torch.device('cuda')
    else:
        quant_device = torch.device('cpu')

    if target_device is None or target_device.type != 'cuda':
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fp8_dtype = get_compatible_fp8_dtype(target_device)
    if fp8_dtype is None:
        print("Warning: No compatible FP8 dtype found, skipping FP8 conversion")
        return

    dtype_name = "float8_e4m3fn" if fp8_dtype == torch.float8_e4m3fn else "float8_e5m2"
    fp8_layers = []
    kept_layers = []
    total_params_before = 0
    total_params_after = 0

    total_layers = sum(1 for name, layer in module.named_modules()
                       if isinstance(layer, nn.Linear) and not any(keyword in name for keyword in params_to_keep))

    layer_count = 0
    for name, layer in module.named_modules():
        if any(keyword in name for keyword in params_to_keep):
            if isinstance(layer, nn.Linear):
                kept_layers.append(name)
            continue

        if isinstance(layer, nn.Linear):
            layer_count += 1
            fp8_layers.append(name)
            original_forward = layer.forward
            total_params_before += layer.weight.numel() * layer.weight.element_size()
            if layer_count % 50 == 0:
                print(f"  Quantizing layer {layer_count}/{total_layers}...")
            weight_device = layer.weight.device
            scale = torch.max(torch.abs(layer.weight.flatten()))
            if scale == 0 or not torch.isfinite(scale):
                scale = torch.tensor(1.0, dtype=torch.float32, device=layer.weight.device)
            weight_normalized = layer.weight / scale
            fp8_weight = weight_normalized.to(fp8_dtype)
            layer.weight = torch.nn.Parameter(fp8_weight)
            scale = scale.cpu().to(dtype=torch.float32)
            total_params_after += layer.weight.numel() * layer.weight.element_size()
            if scale.numel() == 1:
                scale = scale.squeeze()
            layer.register_buffer("scale_weight", scale.to(dtype=torch.float32), persistent=False)

            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda input, m=layer: fp8_linear_forward(m, original_dtype, input))

    if len(fp8_layers) > 0:
        memory_saved_mb = (total_params_before - total_params_after) / (1024 * 1024)
        quant_method = "GPU-accelerated" if gpu_available else "CPU"

        if gpu_available and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    else:
        print("Warning: No layers were converted to FP8. Check if the model architecture matches expected patterns.")
