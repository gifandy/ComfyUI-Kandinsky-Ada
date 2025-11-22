import torch
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

USE_SAGE_ATTENTION = False
DISABLE_COMPILE = False  # Disable torch.compile when using sage attention

try:
    from flash_attn import flash_attn_func as flash_attention_2
    print("FlashAttention 2 is found")
except:
    flash_attention_2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attention_3
    print("FlashAttention 3 is found")
except:
    flash_attention_3 = None

try:
    import sageattention
    SAGE_AVAILABLE = True
    print(f"Sage Attention is found")
except:
    sageattention = None
    SAGE_AVAILABLE = False

def safe_compile_attention(mode=None, dynamic=False):
    def decorator(fn):
        compiled_fn = fn
        try:
            if mode:
                compiled_fn = torch.compile(mode=mode, dynamic=dynamic)(fn)
            else:
                compiled_fn = torch.compile()(fn)
        except (AttributeError, RuntimeError, TypeError):
            compiled_fn = fn

        def runtime_wrapper(*args, **kwargs):
            global DISABLE_COMPILE
            if DISABLE_COMPILE:
                return fn(*args, **kwargs)
            else:
                return compiled_fn(*args, **kwargs)

        return runtime_wrapper
    return decorator

@safe_compile_attention(mode="max-autotune-no-cudagraphs", dynamic=True)
def sdpa(q, k, v, attn_mask=None):
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask
        )
        .transpose(1, 2)
        .contiguous()
    )
    return out

def sage_attn(q, k, v):
    out = (
        sageattention.sageattn(
            q, k, v,
            tensor_layout="NHD",
            is_causal=False
        )
    )
    return out

def set_sage_attention(enabled: bool):
    global USE_SAGE_ATTENTION, DISABLE_COMPILE
    if enabled and not SAGE_AVAILABLE:
        USE_SAGE_ATTENTION = False
        DISABLE_COMPILE = False
    else:
        USE_SAGE_ATTENTION = enabled
        DISABLE_COMPILE = enabled

class SelfAttentionEngine():
    def __init__(self, engine="auto"):
        assert engine in ["auto", "flash_attention_2", "flash_attention_3", "sage", "sdpa"]
        self.attention_fn = None

        if engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_2

        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_3

        if engine == "sage":
            if sageattention is None:
                raise RuntimeError("sage engine selected, but it can't be imported.")
            self.attention_fn = sage_attn

        if engine == "sdpa":
            self.attention_fn = sdpa

        if engine == "auto":
            self.attention_fn = sdpa
            global USE_SAGE_ATTENTION
            if USE_SAGE_ATTENTION and sageattention is not None:
                self.attention_fn = sage_attn
            if flash_attention_2 is not None:
                self.attention_fn = flash_attention_2
            if flash_attention_3 is not None:
                self.attention_fn = flash_attention_3

    def get_attention(self):
        return self.attention_fn
