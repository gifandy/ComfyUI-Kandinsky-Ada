import torch
from tqdm import trange
from typing_extensions import override
from comfy_api.latest import io
import comfy.utils
import comfy.model_management
from comfy.cli_args import args
from PIL import Image
import torch.nn.functional as Fmod
import server
from threading import Thread
import io as pyio
import time
import struct
from importlib.util import find_spec

from .src.kandinsky.magcache_utils import set_magcache_params, disable_magcache
from .src.kandinsky.models.utils import fast_sta_nabla
from .src.kandinsky.models.attention import set_sage_attention

serv = server.PromptServer.instance
MAX_PREVIEW_RESOLUTION = args.preview_size

HYVIDEO_LATENT_RGB_FACTORS = [
    [-0.41, -0.25, -0.26],
    [-0.26, -0.49, -0.24],
    [-0.37, -0.54, -0.3],
    [-0.04, -0.29, -0.29],
    [-0.52, -0.59, -0.39],
    [-0.56, -0.6, -0.02],
    [-0.53, -0.06, -0.48],
    [-0.51, -0.28, -0.18],
    [-0.59, -0.1, -0.33],
    [-0.56, -0.54, -0.41],
    [-0.61, -0.19, -0.5],
    [-0.05, -0.25, -0.17],
    [-0.23, -0.04, -0.22],
    [-0.51, -0.56, -0.43],
    [-0.13, -0.4, -0.05],
    [-0.01, -0.01, -0.48],
]
HYVIDEO_LATENT_RGB_BIAS = [0.0, 0.0, 0.0]


class Latent2RGBPreviewer:
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        latent_image = Fmod.linear(
            x0.movedim(1, -1),
            self.latent_rgb_factors,
            bias=self.latent_rgb_factors_bias,
        )

        img_min = latent_image.min()
        img_max = latent_image.max()
        if (img_max - img_min) > 1e-6:
            latent_image = (latent_image - img_min) / (img_max - img_min)
        else:
            latent_image = torch.zeros_like(latent_image)

        return latent_image


class WrappedPreviewer:

    def __init__(self, previewer, rate=16):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.swarmui_env = find_spec("SwarmComfyCommon") is not None
        if self.swarmui_env:
            print("previewer: SwarmUI output enabled")

        if hasattr(previewer, "latent_rgb_factors"):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
        else:
            raise Exception("Unsupported preview type for VHS animated previews")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(
                dtype=x0.dtype, device=x0.device
            )

        latent_image = Fmod.linear(
            x0.movedim(1, -1),
            self.latent_rgb_factors,
            bias=self.latent_rgb_factors_bias,
        )

        img_min = latent_image.min()
        img_max = latent_image.max()
        if (img_max - img_min) > 1e-6:
            latent_image = (latent_image - img_min) / (img_max - img_min)
        else:
            latent_image = torch.zeros_like(latent_image)

        return latent_image

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])

        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate

        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None

        if self.first_preview:
            self.first_preview = False
            serv.send_sync(
                "VHS_latentpreview",
                {"length": num_images, "rate": self.rate, "id": serv.last_node_id},
            )
            self.last_time = new_time + 1 / self.rate

        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index : self.c_index + num_previews]

        Thread(
            target=self.process_previews,
            args=(x0, self.c_index, num_images, serv.last_node_id),
        ).start()

        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def process_previews(self, image_tensor, ind, leng, node_id):
        if node_id is None:
            return

        max_size = 256

        image_tensor = self.decode_latent_to_preview(image_tensor)

        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = Fmod.interpolate(image_tensor, (height, max_size), mode="bilinear")
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = Fmod.interpolate(image_tensor, (max_size, width), mode="bilinear")
            image_tensor = image_tensor.movedim(0, -1)

        previews_ubyte = (
            image_tensor.clamp(0, 1)
            .mul(0xFF)
            .to(device="cpu", dtype=torch.uint8)
        )

        for preview in previews_ubyte:
            img = Image.fromarray(preview.numpy())
            message = pyio.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(ind.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", node_id.encode("ascii")))
            img.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), serv.client_id)
            ind = (ind + 1) % leng

        if self.swarmui_env:
            images = [Image.fromarray(preview.numpy()) for preview in previews_ubyte]
            message = pyio.BytesIO()
            header = struct.pack(">I", 3)
            message.write(header)
            images[0].save(
                message,
                save_all=True,
                duration=int(1000.0 / self.rate),
                append_images=images[1:],
                lossless=False,
                quality=80,
                method=0,
                format="WEBP",
            )
            message.seek(0)
            preview_bytes = message.getvalue()
            serv.send_sync(1, preview_bytes, sid=serv.client_id)


def get_kandinsky_video_previewer():
    previewer_core = Latent2RGBPreviewer(HYVIDEO_LATENT_RGB_FACTORS, HYVIDEO_LATENT_RGB_BIAS)
    return WrappedPreviewer(previewer_core, rate=16)


@torch.no_grad()
def get_sparse_params(conf, batch_shape, device, attention_mode):
    F_cond, H_cond, W_cond, C_cond = batch_shape
    patch_size = conf.model.dit_params.patch_size

    T = F_cond // patch_size[0]
    H = H_cond // patch_size[1]
    W = W_cond // patch_size[2]

    if attention_mode == "nabla":
        # Use defaults if not in config
        wT = getattr(conf.model.attention, "wT", 11) if hasattr(conf.model, "attention") else 11
        wH = getattr(conf.model.attention, "wH", 3) if hasattr(conf.model, "attention") else 3
        wW = getattr(conf.model.attention, "wW", 3) if hasattr(conf.model, "attention") else 3
        P = getattr(conf.model.attention, "P", 0.8) if hasattr(conf.model, "attention") else 0.8
        add_sta = getattr(conf.model.attention, "add_sta", True) if hasattr(conf.model, "attention") else True
        method = getattr(conf.model.attention, "method", "topcdf") if hasattr(conf.model, "attention") else "topcdf"

        sta_mask = fast_sta_nabla(T, H // 8, W // 8, wT, wH, wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": "nabla",
            "to_fractal": True,
            "P": P,
            "wT": wT,
            "wW": wW,
            "wH": wH,
            "add_sta": add_sta,
            "visual_shape": (T, H, W),
            "method": method,
        }
    else:
        sparse_params = None

    return sparse_params

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
):
    with torch._dynamo.utils.disable_cache_limit():
        if abs(guidance_weight - 1.0) > 1e-6:
            pred_velocity, uncond_pred_velocity = dit(
                x,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=attention_mask,
                x_uncond=x,
                text_embed_uncond=null_text_embeds["text_embeds"],
                pooled_text_embed_uncond=null_text_embeds["pooled_embed"],
                time_uncond=t * 1000,
            )
            pred_velocity = uncond_pred_velocity + guidance_weight * (
                pred_velocity - uncond_pred_velocity
            )
        else:
            pred_velocity = dit(
                x,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=attention_mask,
            )
    return pred_velocity

@torch.no_grad()
def generate(
    diffusion_model,
    device,
    shape,
    steps,
    text_embed,
    null_embed,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    cfg,
    scheduler_scale,
    conf,
    seed,
    pbar,
    visual_cond=None,
    visual_cond_mask=None,
    global_step_offset=0,
    total_steps=None,
    video_previewer=None,
    attention_mode="sdpa",
):

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    model_dtype = next(diffusion_model.parameters()).dtype
    current_latent = torch.randn(shape, generator=g, device=device, dtype=torch.float32)

    if torch.isnan(current_latent).any() or torch.isinf(current_latent).any():
        current_latent = torch.randn(shape, device=device, dtype=torch.float32)

    sparse_params = get_sparse_params(conf, shape, device, attention_mode)

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for i in range(steps):
        t_now = timesteps[i]
        t_next = timesteps[i+1]
        dt = t_next - t_now

        if diffusion_model.visual_cond:
            visual_cond_input = torch.zeros_like(current_latent)
            visual_cond_mask_input = torch.zeros(
                [*current_latent.shape[:-1], 1], dtype=current_latent.dtype, device=current_latent.device
            )
            if visual_cond is not None:
                visual_cond_typed = visual_cond.to(device=current_latent.device, dtype=current_latent.dtype)
                current_latent[:1] = visual_cond_typed[:1]
                visual_cond_mask_input[:1] = 1
            model_input = torch.cat([current_latent, visual_cond_input, visual_cond_mask_input], dim=-1)
        else:
            model_input = current_latent

        pred_velocity = get_velocity(
            diffusion_model,
            model_input,
            t_now.unsqueeze(0),
            text_embed,
            null_embed,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            cfg,
            conf,
            sparse_params=sparse_params,
        )

        if torch.isnan(pred_velocity).any() or torch.isinf(pred_velocity).any():
            pred_velocity = torch.nan_to_num(pred_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            t_scalar = float(t_now)
        except Exception:
            t_scalar = 0.0
        x0_approx = current_latent - t_scalar * pred_velocity
        x0_approx = torch.nan_to_num(x0_approx, nan=0.0, posinf=0.0, neginf=0.0)

        if video_previewer is not None:
            try:
                x0_video = x0_approx.permute(3, 0, 1, 2).unsqueeze(0)
                video_previewer.decode_latent_to_preview_image("JPEG", x0_video)
            except Exception as e:
                print("Kandinsky video preview error:", e)
                video_previewer = None

        pred_velocity = pred_velocity.float()
        current_latent = current_latent + dt * pred_velocity

        if torch.isnan(current_latent).any() or torch.isinf(current_latent).any():
            current_latent = torch.nan_to_num(current_latent, nan=0.0, posinf=0.0, neginf=0.0)

        global_step = global_step_offset + i
        if hasattr(pbar, "update_absolute") and total_steps is not None:
            pbar.update_absolute(global_step + 1, total_steps, None)
        else:
            pbar.update(1)

    return current_latent

class KandinskySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_Sampler",
            display_name="Kandinsky 5 Sampler",
            category="Kandinsky",
            description="Performs the specific Flow Matching sampling loop for Kandinsky-5 models.",
            inputs=[
                io.Model.Input("model", tooltip="The Kandinsky 5 model patcher from the Kandinsky 5 Loader."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=200, tooltip="50, 16 for distilled version."),
                io.Float.Input("cfg", default=5.0, min=1.0, max=20.0, step=0.1, tooltip="1.0 for distilled and nocfg, 5.0 for others."),
                io.Float.Input("scheduler_scale", default=10.0, min=1.0, max=20.0, step=0.1, tooltip="10.0 for 5s, 5.0 for 10s."),
                io.Combo.Input("attention_mode", options=["sdpa", "sage", "nabla"], default="sdpa", tooltip="'nabla' is only for 20B models, gives a big boost to speed."),
                io.Conditioning.Input("positive", tooltip="Positive conditioning from Kandinsky 5 Text Encode."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning from Kandinsky 5 Text Encode."),
                io.Latent.Input("latent_image", tooltip="Empty latent from Empty Kandinsky 5 Latent."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, model, seed, steps, cfg, scheduler_scale, attention_mode, positive, negative, latent_image) -> io.NodeOutput:
        patcher = model
        set_sage_attention(attention_mode == "sage")

        comfy.model_management.load_model_gpu(patcher)
        k_handler = patcher.model
        diffusion_model = k_handler.diffusion_model
        conf = k_handler.conf
        device = patcher.load_device
        model_dtype = next(diffusion_model.parameters()).dtype

        use_magcache = conf.get('use_magcache', False)
        is_magcache_active = hasattr(diffusion_model, '_magcache_enabled') and diffusion_model._magcache_enabled

        if use_magcache:
            if hasattr(conf, "magcache"):
                threshold = conf.get('magcache_threshold', 0.12)

                set_magcache_params(diffusion_model, conf.magcache.mag_ratios, steps,
                                  cfg == 1.0, threshold=threshold,
                                  start_percent=0.2, end_percent=1.0)
            else:
                print("Warning: use_magcache is True but no magcache config found")
        elif is_magcache_active:
            disable_magcache(diffusion_model)

        latent = latent_image["samples"].to(device)
        B, C, F, H, W = latent.shape

        visual_cond = None
        visual_cond_mask = None
        if "visual_cond" in latent_image and "visual_cond_mask" in latent_image:
            visual_cond = latent_image["visual_cond"].to(device=device)
            visual_cond_mask = latent_image["visual_cond_mask"].to(device=device)

        pos_cond = positive[0][1].get("kandinsky_embeds")
        neg_cond = negative[0][1].get("kandinsky_embeds")

        for key in pos_cond:
            if key == "attention_mask":
                if pos_cond[key] is not None:
                    pos_cond[key] = pos_cond[key].to(device=device, dtype=torch.bool)
                if neg_cond[key] is not None:
                    neg_cond[key] = neg_cond[key].to(device=device, dtype=torch.bool)
            else:
                pos_cond[key] = pos_cond[key].to(device=device)
                neg_cond[key] = neg_cond[key].to(device=device)

        patch_size = conf.model.dit_params.patch_size
        visual_rope_pos = [
            torch.arange(F // patch_size[0], device=device),
            torch.arange(H // patch_size[1], device=device),
            torch.arange(W // patch_size[2], device=device)
        ]

        text_rope_pos = torch.arange(pos_cond["text_embeds"].shape[1], device=device)
        null_text_rope_pos = torch.arange(neg_cond["text_embeds"].shape[1], device=device)

        output_latents = []
        total_steps = steps * B
        pbar = comfy.utils.ProgressBar(total_steps)

        try:
            video_previewer = get_kandinsky_video_previewer()
        except Exception as e:
            print("Could not init Kandinsky video previewer:", e)
            video_previewer = None

        if model_dtype == torch.bfloat16:
            autocast_dtype = torch.bfloat16
        elif torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(model_dtype != torch.float32)):
            for i in range(B):
                current_seed = seed + i

                batch_visual_cond = None
                batch_visual_cond_mask = None
                if visual_cond is not None and visual_cond_mask is not None:
                    batch_visual_cond = visual_cond[i].permute(1, 2, 3, 0)
                    batch_visual_cond_mask = visual_cond_mask[i]

                final_latent_unbatched = generate(
                    diffusion_model,
                    device,
                    (F, H, W, C),
                    steps,
                    pos_cond,
                    neg_cond,
                    visual_rope_pos,
                    text_rope_pos,
                    null_text_rope_pos,
                    cfg,
                    scheduler_scale,
                    conf,
                    current_seed,
                    pbar,
                    visual_cond=batch_visual_cond,
                    visual_cond_mask=batch_visual_cond_mask,
                    global_step_offset=i * steps,
                    total_steps=total_steps,
                    video_previewer=video_previewer,
                    attention_mode=attention_mode,
                )
                output_latents.append(final_latent_unbatched.permute(3, 0, 1, 2))

        final_latents = torch.stack(output_latents, dim=0)

        if hasattr(diffusion_model, 'clear_loaded_blocks'):
            diffusion_model.clear_loaded_blocks()

        if torch.isnan(final_latents).any() or torch.isinf(final_latents).any():
            final_latents = torch.nan_to_num(final_latents, nan=0.0, posinf=0.0, neginf=0.0)

        return io.NodeOutput({"samples": final_latents.to(comfy.model_management.intermediate_device())})
