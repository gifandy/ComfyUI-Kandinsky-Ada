import torch
import gc
import time
import comfy.model_patcher
import comfy.model_management as model_management
import comfy.utils

from .src.kandinsky.models.dit import DiffusionTransformer3D
from .src.kandinsky.models import attention
from .src.kandinsky.fp8_utils import convert_fp8_linear, convert_fp8_linear_on_the_fly
from .ops import GGMLOps
from .dequant import is_quantized

KANDINSKY_CONFIGS = {
    "sft_5s": {"config": "config_5s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_5s.safetensors"},
    "sft_10s": {"config": "config_10s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "i2v_5s": {"config": "config_5s_i2v.yaml", "ckpt": "kandinsky/kandinsky5lite_i2v_5s.safetensors"},
    "i2v_pro_20b": {"config": "config_5s_i2v_pro_20b.yaml", "ckpt": "kandinsky/kandinsky5_i2v_pro_sft_5s_20b.safetensors"},
    "t2v_pro_20b": {"config": "config_5s_t2v_pro_20b.yaml", "ckpt": "kandinsky/kandinsky5Pro_t2v_sft_5s.safetensors"},
    "pretrain_5s": {"config": "config_5s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_5s.safetensors"},
    "pretrain_10s": {"config": "config_10s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_10s.safetensors"},
    "nocfg_5s": {"config": "config_5s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_nocfg_5s.safetensors"},
    "nocfg_10s": {"config": "config_10s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "distil_5s": {"config": "config_5s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_5s.safetensors"},
    "distil_10s": {"config": "config_10s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_10s.safetensors"},
}

class KandinskyModelHandler(torch.nn.Module):
    """
    A lightweight placeholder for the Kandinsky DiT model.
    """
    def __init__(self, conf, ckpt_path):
        super().__init__()
        self.conf = conf
        self.ckpt_path = ckpt_path
        self.diffusion_model = None
        self.size = int(conf.model.dit_params.model_dim * 12 * 24 * 1.5)
        self.block_swap_protected = False

    def _apply(self, fn, recurse=True):
        if self.block_swap_protected and self.diffusion_model is not None:
            return self
        else:
            return super()._apply(fn, recurse=recurse)

    def to(self, *args, **kwargs):
        if self.block_swap_protected and self.diffusion_model is not None:
            device_arg = args[0] if args else kwargs.get('device', 'unknown')
            return self
        else:
            return super().to(*args, **kwargs)

    def cuda(self, device=None):
        if self.block_swap_protected and self.diffusion_model is not None:
            return self
        else:
            return super().cuda(device=device)

class KandinskyPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model, *args, **kwargs):
        self._cached_model_size = None
        super().__init__(model, *args, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and self.model.diffusion_model is not None

    def _is_block_swap_enabled(self):
        if not hasattr(self.model, 'conf'):
            return False

        blocks_in_memory = getattr(self.model.conf, 'blocks_in_memory', 0)
        if blocks_in_memory <= 0:
            return False

        if hasattr(self.model.conf, 'model') and hasattr(self.model.conf.model, 'dit_params'):
            num_visual_blocks = self.model.conf.model.dit_params.get('num_visual_blocks', 60)
            return blocks_in_memory < num_visual_blocks

        return False

    def model_size(self):
        block_swap_enabled = self._is_block_swap_enabled()

        cached_size = getattr(self, '_cached_model_size', None)

        if block_swap_enabled and cached_size is not None:
            return cached_size

        size = super().model_size()

        if block_swap_enabled:
            self._cached_model_size = size

        return size

    def load(self, *args, **kwargs):
        if self._is_block_swap_enabled() and self.is_loaded:
            return
        else:
            return super().load(*args, **kwargs)

    def patch_model(self, device_to=None, *args, **kwargs):
        block_swap_enabled = self._is_block_swap_enabled()

        if self.is_loaded:
            if block_swap_enabled:
                return
            else:
                self.model.diffusion_model.to(self.load_device)
            return

        print("\n" + "="*60)
        print("KANDINSKY MODEL LOADING")
        print("="*60)

        model_dtype = model_management.unet_dtype()

        use_fp8 = hasattr(self.model.conf, 'use_fp8') and self.model.conf.use_fp8
        use_gguf = hasattr(self.model.conf, 'use_gguf') and self.model.conf.use_gguf

        if use_fp8 and use_gguf:
            raise ValueError("Cannot use both FP8 and GGUF formats simultaneously. Please choose one.")

        dit_params = dict(self.model.conf.model.dit_params)

        num_blocks = dit_params.get('num_visual_blocks', 32)
        if use_fp8:
            print(f"FP8: Enabled")
        if use_gguf:
            print(f"GGUF: Enabled")

        if block_swap_enabled:
            blocks_in_memory = getattr(self.model.conf, 'blocks_in_memory', 6)
            dit_params['block_swap_enabled'] = True
            dit_params['blocks_in_memory'] = blocks_in_memory
            dit_params['pin_first_n_blocks'] = 2
            dit_params['pin_last_n_blocks'] = 2

        load_dict_start = time.time()
        if use_gguf:
            print(f"Loading GGUF model from disk: {self.model.ckpt_path}")
            from .gguf_loader import load_gguf_state_dict, detect_model_dims, infer_variant_from_dims

            dims = detect_model_dims(self.model.ckpt_path)
            inferred_variant = infer_variant_from_dims(dims)

            model_dim_detected = dims.get('time_in_features')
            if model_dim_detected is not None:
                print(f"  model_dim: {model_dim_detected}")
                dit_params['model_dim'] = model_dim_detected

                if model_dim_detected >= 3584:
                    dit_params['axes_dims'] = [32, 48, 48]
                else:
                    dit_params['axes_dims'] = [16, 24, 24]

            if dims.get('time_out_features') is not None:
                dit_params['time_dim'] = dims['time_out_features']
            if dims.get('ff_dim') is not None:
                dit_params['ff_dim'] = dims['ff_dim']
            elif model_dim_detected is not None:
                inferred_ff_dim = model_dim_detected * 4
                dit_params['ff_dim'] = inferred_ff_dim

            if dims.get('num_text_blocks') is not None:
                dit_params['num_text_blocks'] = dims['num_text_blocks']
            if dims.get('num_visual_blocks') is not None:
                dit_params['num_visual_blocks'] = dims['num_visual_blocks']

            sd = load_gguf_state_dict(self.model.ckpt_path)
        else:
            sd = comfy.utils.load_torch_file(self.model.ckpt_path)
        load_dict_time = time.time() - load_dict_start

        create_start = time.time()
        def _skip_init(m):
            pass

        original_inits = {}
        for module_class in [torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Conv2d, torch.nn.Embedding]:
            if hasattr(module_class, 'reset_parameters'):
                original_inits[module_class] = module_class.reset_parameters
                module_class.reset_parameters = _skip_init

        try:
            model = DiffusionTransformer3D(**dit_params)
        finally:
            for module_class, original_init in original_inits.items():
                module_class.reset_parameters = original_init
        if block_swap_enabled:
            pinned_blocks = set()
            num_visual_blocks = dit_params.get('num_visual_blocks', 32)
            pin_first_n = dit_params.get('pin_first_n_blocks', 2)
            pin_last_n = dit_params.get('pin_last_n_blocks', 2)

            for i in range(min(pin_first_n, num_visual_blocks)):
                pinned_blocks.add(i)
            for i in range(max(0, num_visual_blocks - pin_last_n), num_visual_blocks):
                pinned_blocks.add(i)

            model.pinned_blocks = pinned_blocks
        load_weights_start = time.time()
        try:
            if use_gguf:
                with torch.no_grad():
                    m, u = model.load_state_dict(sd, strict=False, assign=True)
            else:
                m, u = model.load_state_dict(sd, strict=False)

            load_weights_time = time.time() - load_weights_start
            if len(m) > 0:
                print(f"  Warning: {len(m)} missing keys")
            if len(u) > 0:
                print(f"  Warning: {len(u)} unexpected keys")
        except RuntimeError as e:
            if use_gguf and "size of tensor" in str(e):
                from .gguf_loader import detect_model_dims, infer_variant_from_dims
                dims = detect_model_dims(self.model.ckpt_path)
                inferred = infer_variant_from_dims(dims)
                raise RuntimeError(error_msg) from e
            else:
                raise

        del sd
        gc.collect()

        if use_gguf:
            replaced_count = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(p is not None and is_quantized(p) for p in module.parameters()):
                        ggml_linear = GGMLOps.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                        ggml_linear.weight = module.weight
                        ggml_linear.bias = module.bias

                        if '.' in name:
                            parent_name, child_name = name.rsplit('.', 1)
                            parent_module = model.get_submodule(parent_name)
                            setattr(parent_module, child_name, ggml_linear)
                            replaced_count += 1

        model.eval()

        # Disable torch.compile for GGUF models to avoid Dynamo guard failures
        # GGUF quantized weights have dynamic shapes that break compiled graphs
        if use_gguf:
            if hasattr(torch, '_dynamo'):
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.force_parameter_static_shapes = False
                # Mark model as not compilable
                model._is_gguf = True
            
            # Disable compile globally for this run
            attention.DISABLE_COMPILE = True

        if use_fp8 and not use_gguf:
            fp8_start = time.time()
            print(f"Applying FP8 quantization...")
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(self.load_device)
                print(f"GPU Compute Capability: {major}.{minor}")
                if (major > 8) or (major == 8 and minor >= 9):
                    print("  -> torch._scaled_mm supported (fast FP8 computation)")
                else:
                    print("  -> torch._scaled_mm not supported (will use dequant fallback)")
                    print("     FP8 will still save memory but computation will be slower")

            params_to_keep = {"norm", "bias", "time_in", "patch_embedding", "time_", "img_emb",
                            "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj"}

            try:
                use_gpu_quant = hasattr(self.model.conf, 'fp8_use_gpu') and self.model.conf.fp8_use_gpu
                convert_fp8_linear_on_the_fly(model, model_dtype, params_to_keep=params_to_keep,
                                             use_gpu=use_gpu_quant, target_device=self.load_device)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Failed to apply FP8 quantization: {e}")

        if block_swap_enabled:
            load_start = time.time()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if use_fp8 or use_gguf:
                model.time_embeddings.to(device=self.load_device, non_blocking=True)
                model.text_embeddings.to(device=self.load_device, non_blocking=True)
                model.pooled_text_embeddings.to(device=self.load_device, non_blocking=True)
            else:
                model.time_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)
                model.text_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)
                model.pooled_text_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            if use_fp8 or use_gguf:
                model.visual_embeddings.to(device=self.load_device, non_blocking=True)
                model.text_rope_embeddings.to(device=self.load_device, non_blocking=True)
                model.visual_rope_embeddings.to(device=self.load_device, non_blocking=True)
                model.out_layer.to(device=self.load_device, non_blocking=True)
            else:
                model.visual_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)
                model.text_rope_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)
                model.visual_rope_embeddings.to(device=self.load_device, dtype=model_dtype, non_blocking=True)
                model.out_layer.to(device=self.load_device, dtype=model_dtype, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            for block in model.text_transformer_blocks:
                if use_fp8 or use_gguf:
                    block.to(device=self.load_device, non_blocking=True)
                else:
                    block.to(device=self.load_device, dtype=model_dtype, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            for i, block in enumerate(model.visual_transformer_blocks):
                if next(block.parameters()).device.type != 'cpu':
                    block.to(self.offload_device)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            num_pinned = len(pinned_blocks)
            if num_pinned > 0:
                for idx, i in enumerate(sorted(pinned_blocks)):
                    if use_fp8 or use_gguf:
                        model.visual_transformer_blocks[i].to(device=self.load_device, non_blocking=True)
                    else:
                        model.visual_transformer_blocks[i].to(device=self.load_device, dtype=model_dtype, non_blocking=True)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        mem_after = torch.cuda.memory_allocated(self.load_device) / 1024**2

            original_apply = model._apply

            def block_swap_aware_apply(fn, recurse=True):
                if not recurse:
                    return original_apply(fn, recurse=False)

                for name, module in model.named_children():
                    if name != 'visual_transformer_blocks':
                        module._apply(fn, recurse=True)

                for i, block in enumerate(model.visual_transformer_blocks):
                    if i in pinned_blocks:
                        block._apply(fn, recurse=True)

                for param in model._parameters.values():
                    if param is not None:
                        fn(param)

                for buffer in model._buffers.values():
                    if buffer is not None:
                        fn(buffer)

                return model

            model._apply = block_swap_aware_apply
            self.model.block_swap_protected = True

        else:
            if use_fp8 or use_gguf:
                model.to(device=self.load_device)
            else:
                model.to(device=self.load_device, dtype=model_dtype)

            if model_management.force_channels_last():
                model.to(memory_format=torch.channels_last)

        if use_gguf:
            print("GGUF model loaded successfully.")

        self.model.diffusion_model = model
        return

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if self.is_loaded:
            if not self._is_block_swap_enabled():
                self.model.diffusion_model.to(self.offload_device)
            else:
                self.model.block_swap_protected = False

        if unpatch_weights:
             if self.is_loaded:
                del self.model.diffusion_model
                self.model.diffusion_model = None
                self.model.block_swap_protected = False
             gc.collect()
             model_management.soft_empty_cache()
        return
