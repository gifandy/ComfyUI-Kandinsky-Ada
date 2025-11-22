# This is an adaptation of Magcache from https://github.com/Zehong-Ma/MagCache/
import numpy as np
import torch
from types import MethodType


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def set_magcache_params(dit, mag_ratios, num_steps, no_cfg, threshold=0.12, max_skip_steps=2, start_percent=0.2, end_percent=1.0):
    dit.cnt = 0
    dit.num_steps = num_steps * 2
    dit.magcache_thresh = threshold
    dit.K = max_skip_steps
    dit.accumulated_err = [0.0, 0.0]
    dit.accumulated_steps = [0, 0]
    dit.accumulated_ratio = [1.0, 1.0]
    dit.magcache_start_percent = start_percent
    dit.magcache_end_percent = end_percent
    dit.residual_cache = [None, None]
    dit.mag_ratios = np.array([1.0]*2 + mag_ratios)
    dit.no_cfg = no_cfg
    dit._magcache_enabled = True
    dit._magcache_skipped_count = 0
    dit._magcache_full_count = 0

    use_window = end_percent < 1.0

    print(f'MagCache configured:')
    print(f'  - Threshold: {threshold} (lower=better quality, fewer skips)')
    print(f'  - Max skip steps: {max_skip_steps}')
    magcache_start = int(dit.num_steps * start_percent)
    magcache_end = int(dit.num_steps * end_percent)
    if use_window:
        print(f'  - Window mode: {start_percent*100:.0f}%-{end_percent*100:.0f}% (steps {magcache_start}-{magcache_end} of {dit.num_steps})')
    else:
        print(f'  - Run-to-end mode: starts at {start_percent*100:.0f}% (full compute for first {magcache_start} steps)')
    print(f'  - Total steps: {num_steps}, CFG steps: {dit.num_steps}')
    print(f'  - MagCache eligible steps: {magcache_start}-{magcache_end} (~{((magcache_end - magcache_start)/dit.num_steps)*100:.0f}%)')

    if not hasattr(dit, '_original_forward'):
        dit._original_forward = dit.forward

    dit.forward = MethodType(magcache_forward, dit)

    if len(dit.mag_ratios) != num_steps * 2:
        print(f'interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}')
        mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], num_steps)
        mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], num_steps)
        interpolated_mag_ratios = np.concatenate(
            [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        dit.mag_ratios = interpolated_mag_ratios


def disable_magcache(dit):
    if hasattr(dit, '_original_forward'):
        dit.forward = dit._original_forward
        delattr(dit, '_original_forward')

    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()

    magcache_attrs = [
        'cnt', 'num_steps', 'magcache_thresh', 'K',
        'accumulated_err', 'accumulated_steps', 'accumulated_ratio',
        'magcache_start_percent', 'magcache_end_percent',
        'residual_cache', 'mag_ratios',
        'no_cfg', '_magcache_enabled', 'pinned_blocks',
        '_magcache_skipped_count', '_magcache_full_count'
    ]

    for attr in magcache_attrs:
        if hasattr(dit, attr):
            delattr(dit, attr)


def magcache_forward(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
    attention_mask=None
):
    if not hasattr(self, 'cnt') or not hasattr(self, '_magcache_enabled'):
        if hasattr(self, '_original_forward'):
            return self._original_forward(x, text_embed, pooled_text_embed,
                                         time, visual_rope_pos, text_rope_pos,
                                         scale_factor, sparse_params, attention_mask)
        else:
            # Fallback: run without magcache optimization
            text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
                text_embed, time, pooled_text_embed, x, text_rope_pos)
            for text_transformer_block in self.text_transformer_blocks:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
            visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
                visual_embed, visual_rope_pos, scale_factor, sparse_params)
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)
            x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
            return x

    if self.cnt == 0:
        self.residual_cache = [None, None]

    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos)

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params)

    skip_forward = False
    ori_visual_embed = visual_embed

    magcache_start = int(self.num_steps * self.magcache_start_percent)
    magcache_end = int(self.num_steps * self.magcache_end_percent)
    in_magcache_range = self.cnt >= magcache_start and self.cnt < magcache_end

    if in_magcache_range:
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2]*cur_mag_ratio
        self.accumulated_steps[self.cnt%2] += 1
        cur_skip_err = np.abs(1-self.accumulated_ratio[self.cnt%2])
        self.accumulated_err[self.cnt%2] += cur_skip_err

        if self.accumulated_err[self.cnt%2]<self.magcache_thresh and self.accumulated_steps[self.cnt%2]<=self.K:
            if self.residual_cache[self.cnt%2] is not None:
                skip_forward = True
                # Move cached residual to same device as visual_embed to avoid device mismatch
                residual_visual_embed = self.residual_cache[self.cnt%2].to(visual_embed.device)
        else:
            self.accumulated_err[self.cnt%2] = 0
            self.accumulated_steps[self.cnt%2] = 0
            self.accumulated_ratio[self.cnt%2] = 1.0
    else:
        self.accumulated_err[self.cnt%2] = 0
        self.accumulated_steps[self.cnt%2] = 0
        self.accumulated_ratio[self.cnt%2] = 1.0

    if skip_forward:
        visual_embed =  visual_embed + residual_visual_embed
        self._magcache_skipped_count += 1
    else:
        # Run full forward pass with block swapping support
        if hasattr(self, 'block_swap_enabled') and self.block_swap_enabled:
            device = visual_embed.device
            offload_device = torch.device('cpu')
            cuda_available = torch.cuda.is_available()

            # Use persistent state to keep blocks loaded across inference steps
            if not hasattr(self, 'currently_loaded_blocks'):
                self.currently_loaded_blocks = []
                self.prefetch_started = set()
            currently_loaded_blocks = self.currently_loaded_blocks
            prefetch_started = self.prefetch_started

            # Ensure required attributes exist
            if not hasattr(self, 'pinned_blocks'):
                self.pinned_blocks = set()
            if not hasattr(self, 'block_events'):
                self.block_events = {}
            if not hasattr(self, 'swap_stream'):
                self.swap_stream = None

            for i, visual_transformer_block in enumerate(self.visual_transformer_blocks):
                if i in self.pinned_blocks:
                    visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                            visual_rope, sparse_params, attention_mask)
                    continue

                while len(currently_loaded_blocks) >= self.blocks_in_memory:
                    block_to_offload = currently_loaded_blocks.pop(0)
                    prefetch_started.discard(block_to_offload)

                    if self.swap_stream is not None:
                        with torch.cuda.stream(self.swap_stream):
                            self.visual_transformer_blocks[block_to_offload].to(offload_device, non_blocking=True)
                        self.swap_stream.synchronize()
                    else:
                        self.visual_transformer_blocks[block_to_offload].to(offload_device)

                    if cuda_available:
                        torch.cuda.empty_cache()

                if i not in currently_loaded_blocks and i not in prefetch_started:
                    visual_transformer_block.to(device)
                    currently_loaded_blocks.append(i)

                elif i in prefetch_started:
                    if cuda_available and i in self.block_events:
                        event = self.block_events[i]
                        if not event.query():
                            event.synchronize()
                    currently_loaded_blocks.append(i)
                    prefetch_started.discard(i)

                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)

                # Prefetch next block
                next_idx = i + 1
                num_blocks = getattr(self, 'num_visual_blocks', len(self.visual_transformer_blocks))
                if (next_idx < num_blocks and
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
        else:
            # No block swapping, simple loop
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)

        residual_visual_embed = visual_embed - ori_visual_embed
        self._magcache_full_count += 1

    # Cache residual with detach to avoid keeping computation graphs
    self.residual_cache[self.cnt%2] = residual_visual_embed.detach() 

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    if self.no_cfg:
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps:
        total_steps = self._magcache_skipped_count + self._magcache_full_count
        if total_steps > 0:
            skip_percent = (self._magcache_skipped_count / total_steps) * 100
            print(f'MagCache stats: {self._magcache_skipped_count} skipped, {self._magcache_full_count} full ({skip_percent:.1f}% cached)')

        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self.residual_cache = [None, None]  # Clear cached residuals between generations
        self._magcache_skipped_count = 0
        self._magcache_full_count = 0
    return x
