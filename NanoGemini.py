import os
import json
import base64
from io import BytesIO

# --- optional deps (ComfyUI App có thể cài qua Manager -> Check Missing) ---
import requests
from PIL import Image
import torch
import numpy as np

# ------------------------------------------------------------
# Small config helpers
# ------------------------------------------------------------
p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        with open(os.path.join(p, 'config.json'), 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(config):
    with open(os.path.join(p, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

# ------------------------------------------------------------
# The node class
# ------------------------------------------------------------
class ComfyUI_Nano3shop:
    """
    Nano3shop: Image generation/edit node that calls Gemini image model.
    Requires paid API key via env GEMINI_API_KEY or the `api_key` widget.
    """
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        # ignore common placeholders
        placeholders = {
            "token_here", "place_token_here", "your_api_key",
            "api_key_here", "enter_your_key", "<api_key>"
        }

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key
            if self.api_key is None:
                self.api_key = get_config().get("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Cô gái Việt Nam xinh đẹp",
                    "multiline": True,
                    "tooltip": "Mô tả nội dung bạn muốn tạo/chỉnh sửa"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Chọn kiểu thao tác ảnh"
                }),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 1"}),
                "reference_image_2": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 2"}),
                "reference_image_3": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 3"}),
                "reference_image_4": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 4"}),
                "reference_image_5": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 5"}),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (trả phí). Để trống nếu đã set env GEMINI_API_KEY."
                }),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "quality": (["standard", "high"], {"default": "high"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "character_consistency": ("BOOLEAN", {"default": True}),
                "enable_safety": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "nano_banana_generate"
    CATEGORY = "Nano3shop"  # nhóm riêng để dễ tìm trong UI
    DESCRIPTION = "Generate/edit images via Gemini image model (paid key required)."

    # ----------------------------- helpers -----------------------------
    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # lấy ảnh đầu nếu batch
        img_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(img_np, mode='RGB')

    def create_placeholder_image(self, w=512, h=512):
        img = Image.new('RGB', (w, h), color=(100, 100, 100))
        try:
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            d.text((w//2-70, h//2-10), "Generation Failed", fill=(255, 255, 255))
        except Exception:
            pass
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def _image_to_base64_part(self, pil_image):
        buf = BytesIO()
        pil_image.save(buf, format='PNG')
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8")
            }
        }

    def prepare_images_for_api(self, *imgs):
        parts = []
        for img in imgs:
            if img is None:
                continue
            if isinstance(img, torch.Tensor):
                pil = self.tensor_to_image(img[0] if len(img.shape) == 4 else img)
                parts.append(self._image_to_base64_part(pil))
        return parts

    def build_prompt_for_operation(self, prompt, operation, has_refs, aspect_ratio, keep_char):
        aspect_text = {
            "1:1": "square format",
            "16:9": "widescreen landscape format",
            "9:16": "portrait format",
            "4:3": "standard landscape format",
            "3:4": "standard portrait format",
        }.get(aspect_ratio, "square format")

        base = "Generate a high-quality, photorealistic image"
        if operation == "generate":
            if has_refs:
                out = f"{base} inspired by the reference images. {prompt}. in {aspect_text}."
            else:
                out = f"{base} of: {prompt}. in {aspect_text}."
        elif operation == "edit":
            if not has_refs:
                return "Error: Edit operation requires reference images"
            out = f"Edit the provided reference image(s). {prompt}. Keep composition and quality."
        elif operation == "style_transfer":
            if not has_refs:
                return "Error: Style transfer requires reference images"
            out = f"Apply the style from the reference images to create: {prompt}. in {aspect_text}."
        elif operation == "object_insertion":
            if not has_refs:
                return "Error: Object insertion requires reference images"
            out = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting and perspective. in {aspect_text}."
        else:
            out = prompt

        if keep_char and has_refs:
            out += " Maintain character consistency and visual identity from the reference images."
        return out

    # ------------------------- API call (google-genai) -------------------------
    def call_nano_banana_api(self, prompt, encoded_images, temperature, batch_count, enable_safety):
        try:
            # yêu cầu gói: google-genai
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            gen_cfg = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=["Text", "Image"]
            )

            # parts = [{text}, {inline_data}, ...]
            parts = [{"text": prompt}]
            parts.extend(encoded_images)
            contents = [{"parts": parts}]

            all_images = []
            op_log = ""

            for i in range(batch_count):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.0-flash-exp-image-generation",
                        contents=contents,
                        config=gen_cfg
                    )
                    batch_imgs = []
                    resp_text = ""

                    if getattr(resp, "candidates", None):
                        for cand in resp.candidates:
                            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                                for part in cand.content.parts:
                                    if getattr(part, "text", None):
                                        resp_text += part.text + "\n"
                                    if getattr(part, "inline_data", None):
                                        try:
                                            batch_imgs.append(part.inline_data.data)
                                        except Exception as e:
                                            op_log += f"Extract image error: {e}\n"

                    if batch_imgs:
                        all_images.extend(batch_imgs)
                        op_log += f"Batch {i+1}: {len(batch_imgs)} image(s)\n"
                    else:
                        op_log += f"Batch {i+1}: no images. Text: {resp_text[:120]}...\n"

                except Exception as e:
                    op_log += f"Batch {i+1} error: {e}\n"

            tensors = []
            for b in all_images:
                try:
                    im = Image.open(BytesIO(b))
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    arr = np.array(im).astype(np.float32) / 255.0
                    tensors.append(torch.from_numpy(arr)[None, ...])
                except Exception as e:
                    op_log += f"Process image error: {e}\n"

            return tensors, op_log

        except ImportError:
            return [], "google-genai not installed. Add it to requirements.txt and run Manager → Check Missing.\n"
        except Exception as e:
            return [], f"API error: {e}\n"

    # ------------------------- main entry -------------------------
    def nano_banana_generate(
        self, prompt, operation,
        reference_image_1=None, reference_image_2=None, reference_image_3=None,
        reference_image_4=None, reference_image_5=None, api_key="",
        batch_count=1, temperature=0.7, quality="high", aspect_ratio="1:1",
        character_consistency=True, enable_safety=True
    ):
        # key from widget overrides
        if api_key.strip():
            self.api_key = api_key.strip()
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            msg = (
                "NANO3SHOP ERROR: No API key!\n"
                "Gemini image models require a PAID API key.\n"
                "Set env GEMINI_API_KEY or fill the api_key field."
            )
            return (self.create_placeholder_image(), msg)

        try:
            encoded_refs = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5
            )
            has_refs = len(encoded_refs) > 0

            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_refs, aspect_ratio, character_consistency
            )
            if final_prompt.startswith("Error:"):
                return (self.create_placeholder_image(), final_prompt)

            if quality == "high":
                final_prompt += " Use the highest quality settings available."

            log = [
                "NANO3SHOP OPERATION LOG",
                f"Operation: {operation}",
                f"Refs: {len(encoded_refs)}",
                f"Batch: {batch_count}",
                f"Temp: {temperature}",
                f"Quality: {quality}",
                f"AR: {aspect_ratio}",
                f"Keep character: {character_consistency}",
                f"Safety: {enable_safety}",
                f"Prompt: {final_prompt[:160]}..."
            ]
            op_log = "\n".join(log) + "\n\n"

            tensors, api_log = self.call_nano_banana_api(
                final_prompt, encoded_refs, temperature, batch_count, enable_safety
            )
            op_log += api_log

            if tensors:
                out = torch.cat(tensors, dim=0)
                est_cost = len(tensors) * 0.039
                op_log += f"\nEstimated cost: ~${est_cost:.3f}\nOK: {len(tensors)} image(s)."
                return (out, op_log)

            op_log += "\nNo images were generated."
            return (self.create_placeholder_image(), op_log)

        except Exception as e:
            return (self.create_placeholder_image(),
                    f"NANO3SHOP ERROR: {e}\nCheck API key, internet, or dependencies.")

# ------------------------------------------------------------
# Node registration (BẮT BUỘC)
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ComfyUI_Nano3shop": ComfyUI_Nano3shop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_Nano3shop": "Nano3shop",
}
# đặt nhóm hiển thị riêng

ComfyUI_Nano3shop.CATEGORY = "Nano3shop"

