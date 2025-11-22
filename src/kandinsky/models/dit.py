import torch
from torch import nn

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
)
from .utils import fractal_flatten, fractal_unflatten

from . import attention

def safe_compile():
    def decorator(fn):
        compiled_fn = fn
        try:
            compiled_fn = torch.compile()(fn)
        except (AttributeError, RuntimeError):
            pass

        def runtime_wrapper(*args, **kwargs):
            if hasattr(attention, 'DISABLE_COMPILE') and attention.DISABLE_COMPILE:
                return fn(*args, **kwargs)
            else:
                return compiled_fn(*args, **kwargs)
        return runtime_wrapper
    return decorator


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        operations = nn
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim)

        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope, attention_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope, attention_mask)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto"):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, attention_engine)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params, attention_mask=None):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        visual_out = self.self_attention(visual_out, rope, sparse_params)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        visual_out = self.cross_attention(visual_out, text_embed, attention_mask)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_engine="auto",
        block_swap_enabled=False,
        blocks_in_memory=6,
        pin_first_n_blocks=2,
        pin_last_n_blocks=2,
    ):
        super().__init__()

        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        self.block_swap_enabled = block_swap_enabled
        self.blocks_in_memory = blocks_in_memory
        self.pin_first_n_blocks = pin_first_n_blocks
        self.pin_last_n_blocks = pin_last_n_blocks
        self.num_visual_blocks = num_visual_blocks

        self.swap_stream = None
        self.block_events = {}
        if block_swap_enabled and torch.cuda.is_available():
            self.swap_stream = torch.cuda.Stream()

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_text_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine)
                for _ in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    def setup_block_swapping(self, device, offload_device):
        if not self.block_swap_enabled:
            return

        # Initialize persistent state for block swapping
        self.currently_loaded_blocks = []
        self.prefetch_started = set()

        self.pinned_blocks = set()

        for i in range(min(self.pin_first_n_blocks, self.num_visual_blocks)):
            self.pinned_blocks.add(i)

        for i in range(max(0, self.num_visual_blocks - self.pin_last_n_blocks), self.num_visual_blocks):
            self.pinned_blocks.add(i)

        for i, block in enumerate(self.visual_transformer_blocks):
            if i not in self.pinned_blocks:
                block.to(offload_device)


    def clear_loaded_blocks(self):
        """Offload all currently loaded blocks to CPU. Call this when generation is complete."""
        if not self.block_swap_enabled or not hasattr(self, 'currently_loaded_blocks'):
            return

        offload_device = torch.device('cpu')
        if self.swap_stream is not None:
            with torch.cuda.stream(self.swap_stream):
                for block_idx in self.currently_loaded_blocks:
                    self.visual_transformer_blocks[block_idx].to(offload_device, non_blocking=True)
            self.swap_stream.synchronize()
        else:
            for block_idx in self.currently_loaded_blocks:
                self.visual_transformer_blocks[block_idx].to(offload_device)

        self.currently_loaded_blocks = []
        self.prefetch_started = set()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def swap_block_to_device(self, block_idx, device):
        if block_idx < len(self.visual_transformer_blocks):
            self.visual_transformer_blocks[block_idx].to(device)

    @safe_compile()
    def before_text_transformer_blocks(self, text_embed, time, pooled_text_embed, x,
                                       text_rope_pos):
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    @safe_compile()
    def before_visual_transformer_blocks(self, visual_embed, visual_rope_pos, scale_factor,
                                         sparse_params):
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape,
                                                    block_mask=to_fractal)
        return visual_embed, visual_shape, to_fractal, visual_rope

    def after_blocks(self, visual_embed, visual_shape, to_fractal, text_embed, time_embed):
        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)
        return x

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None,
        x_uncond=None,
        text_embed_uncond=None,
        pooled_text_embed_uncond=None,
        time_uncond=None,
    ):
        text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos)

        if x_uncond is not None:
            text_embed_uncond, time_embed_uncond, _, visual_embed_uncond = self.before_text_transformer_blocks(
                text_embed_uncond, time_uncond, pooled_text_embed_uncond, x_uncond, text_rope_pos)

        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
            if x_uncond is not None:
                text_embed_uncond = text_transformer_block(text_embed_uncond, time_embed_uncond, text_rope, attention_mask)

        visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params)

        if x_uncond is not None:
            visual_embed_uncond, _, _, _ = self.before_visual_transformer_blocks(
                visual_embed_uncond, visual_rope_pos, scale_factor, sparse_params)

        if self.block_swap_enabled:
            device = visual_embed.device
            offload_device = torch.device('cpu')
            main_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
            cuda_available = torch.cuda.is_available()

            # Use persistent state to keep blocks loaded across inference steps
            if not hasattr(self, 'currently_loaded_blocks'):
                self.currently_loaded_blocks = []
                self.prefetch_started = set()
            currently_loaded_blocks = self.currently_loaded_blocks
            prefetch_started = self.prefetch_started

            if cuda_available:
                start_mem = torch.cuda.memory_allocated(device) / 1024**2
                max_mem_seen = start_mem

            for i, visual_transformer_block in enumerate(self.visual_transformer_blocks):
                if i in self.pinned_blocks:
                    visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                            visual_rope, sparse_params, attention_mask)
                    if x_uncond is not None:
                        visual_embed_uncond = visual_transformer_block(visual_embed_uncond, text_embed_uncond, time_embed_uncond,
                                                                    visual_rope, sparse_params, attention_mask)
                    continue

                while len(currently_loaded_blocks) >= self.blocks_in_memory:
                    block_to_offload = currently_loaded_blocks.pop(0)
                    prefetch_started.discard(block_to_offload)

                    if cuda_available:
                        mem_before_offload = torch.cuda.memory_allocated(device) / 1024**2

                    if self.swap_stream is not None:
                        with torch.cuda.stream(self.swap_stream):
                            self.visual_transformer_blocks[block_to_offload].to(offload_device, non_blocking=True)
                        self.swap_stream.synchronize()
                    else:
                        self.visual_transformer_blocks[block_to_offload].to(offload_device)

                    if cuda_available:
                        main_stream.synchronize()
                        torch.cuda.empty_cache()
                        mem_after_offload = torch.cuda.memory_allocated(device) / 1024**2
                        freed = mem_before_offload - mem_after_offload

                if i not in currently_loaded_blocks and i not in prefetch_started:
                    if cuda_available:
                        mem_before_load = torch.cuda.memory_allocated(device) / 1024**2

                    visual_transformer_block.to(device)
                    currently_loaded_blocks.append(i)

                    if cuda_available:
                        torch.cuda.synchronize()
                        mem_after_load = torch.cuda.memory_allocated(device) / 1024**2
                        loaded = mem_after_load - mem_before_load
                        max_mem_seen = max(max_mem_seen, mem_after_load)

                elif i in prefetch_started:
                    if cuda_available and i in self.block_events:
                        event = self.block_events[i]
                        if not event.query():
                            event.synchronize()
                    currently_loaded_blocks.append(i)
                    prefetch_started.discard(i)

                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)
                if x_uncond is not None:
                    visual_embed_uncond = visual_transformer_block(visual_embed_uncond, text_embed_uncond, time_embed_uncond,
                                                                visual_rope, sparse_params, attention_mask)

                next_idx = i + 1
                if (next_idx < self.num_visual_blocks and
                    next_idx not in self.pinned_blocks and
                    next_idx not in currently_loaded_blocks and
                    next_idx not in prefetch_started and
                    len(currently_loaded_blocks) < self.blocks_in_memory):

                    if self.swap_stream is not None:
                        with torch.cuda.stream(self.swap_stream):
                            self.visual_transformer_blocks[next_idx].to(device, non_blocking=True)
                            if cuda_available:
                                if next_idx not in self.block_events:
                                    self.block_events[next_idx] = torch.cuda.Event()
                                self.block_events[next_idx].record(self.swap_stream)
                        prefetch_started.add(next_idx)

            # Blocks remain loaded across inference steps for efficiency
            # Use clear_loaded_blocks() method to explicitly offload when generation is complete
        else:
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)
                if x_uncond is not None:
                    visual_embed_uncond = visual_transformer_block(visual_embed_uncond, text_embed_uncond, time_embed_uncond,
                                                                visual_rope, sparse_params, attention_mask)

        x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
        
        if x_uncond is not None:
            x_uncond = self.after_blocks(visual_embed_uncond, visual_shape, to_fractal, text_embed_uncond, time_embed_uncond)
            return x, x_uncond
            
        return x


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit
