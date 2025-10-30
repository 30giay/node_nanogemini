# NanoGemini.py
import os
import json
import base64
from io import BytesIO

from PIL import Image
import torch
import numpy as np

# ------------------------------------------------------------
# Small config helpers
# ------------------------------------------------------------
p = os.path.dirname(os.path.realpath(__file__))


def _cfg_path():
    return os.path.join(p, "config.json")


def get_config():
    try:
        with open(_cfg_path(), "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(config: dict):
    try:
        with open(_cfg_path(), "w") as f:
            json.dump(config, f, indent=4)
    except Exception:
        pass


# ------------------------------------------------------------
# The node class (Gemini 2.5 Image)
# ------------------------------------------------------------
class ComfyUI_NanoGemini:
    """
    NanoGemini: Image generation/edit node using Google Gemini 2.5 Image.
    Requires a PAID API key (env GEMINI_API_KEY or the `api_key` widget).
    """

    # Candidate model ids (first working one will be used)
    # You can override via env GEMINI_IMAGE_MODEL
    MODEL_CANDIDATES = [
        # Stable/preview names may vary across accounts/regions – keep a few fallbacks:
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-generation",
        "gemini-2.5-flash",  # will still return images if the account exposes image modality
        # Older exp fallback (last resort):
        "gemini-2.0-flash-exp-image-generation",
    ]

    def __init__(self, api_key=None):
        # 1) Env first
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()

        placeholders = {
            "token_here",
            "place_token_here",
            "your_api_key",
            "api_key_here",
            "enter_your_key",
            "<api_key>",
        }

        if env_key and env_key.lower() not in placeholders:
            self.api_key = env_key
        else:
            # 2) Node param
            self.api_key = (api_key or "").strip() or None
            # 3) Local config
            if self.api_key is None:
                self.api_key = (get_config().get("GEMINI_API_KEY") or "").strip() or None

        # Resolve model id (env override > default list)
        self.model_id = os.environ.get("GEMINI_IMAGE_MODEL", "").strip() or None

    # ---- ComfyUI schema ----
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Cô gái Việt Nam xinh đẹp",
                        "multiline": True,
                        "tooltip": "Mô tả nội dung bạn muốn tạo/chỉnh sửa",
                    },
                ),
                "operation": (
                    ["generate", "edit", "style_transfer", "object_insertion"],
                    {"default": "generate", "tooltip": "Chọn kiểu thao tác ảnh"},
                ),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 1"}),
                "reference_image_2": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 2"}),
                "reference_image_3": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 3"}),
                "reference_image_4": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 4"}),
                "reference_image_5": ("IMAGE", {"forceInput": False, "tooltip": "Ảnh tham chiếu 5"}),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Gemini API key (trả phí). Để trống nếu đã set env GEMINI_API_KEY.",
                    },
                ),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "quality": (["standard", "high"], {"default": "high"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "character_consistency": ("BOOLEAN", {"default": True}),
                "enable_safety": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "nano_gemini_generate"
    CATEGORY = "Nano / Gemini 2.5"
    DESCRIPTION = "Generate/edit images via Google Gemini 2.5 Image (paid API key required)."

    # ----------------------------- helpers -----------------------------
    def tensor_to_image(self, tensor):
        t = tensor.cpu()
        if len(t.shape) == 4:
            t = t[0]  # lấy ảnh đầu nếu batch
        arr = t.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(arr, mode="RGB")

    def create_placeholder_image(self, w=512, h=512):
        img = Image.new("RGB", (w, h), color=(100, 100, 100))
        try:
            from PIL import ImageDraw

            d = ImageDraw.Draw(img)
            d.text((w // 2 - 70, h // 2 - 10), "Generation Failed", fill=(255, 255, 255))
        except Exception:
            pass
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def _image_to_base64_part(self, pil_image):
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
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

    # ------------------------- API call (google-genai v1) -------------------------
    def _pick_model_id(self):
        """Choose model id: env override or first working candidate."""
        if self.model_id:
            return [self.model_id]  # user-forced
        return self.MODEL_CANDIDATES

    def _call_gemini_images(self, prompt, encoded_images, temperature, batch_count, enable_safety):
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            return [], f"google-genai not installed or import error: {e}\n"

        if not self.api_key:
            return [], "No API key. Set env GEMINI_API_KEY or fill api_key field.\n"

        client = genai.Client(api_key=self.api_key)

        gen_cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["Text", "Image"],  # crucial to receive images
            safety_settings=None if not enable_safety else None,
        )

        parts = [{"text": prompt}]
        parts.extend(encoded_images)
        contents = [{"parts": parts}]

        all_images = []
        op_log = ""
        model_used = None
        last_error = ""

        # Try models in order
        for mid in self._pick_model_id():
            try:
                # probe 1 small call (no batches) to validate this model id quickly
                resp_test = client.models.generate_content(model=mid, contents=contents, config=gen_cfg)
                model_used = mid
                break
            except Exception as e:
                last_error = str(e)
                continue

        if not model_used:
            return [], f"Cannot use any Gemini 2.5 image model. Last error: {last_error}\n"

        op_log += f"Using model: {model_used}\n"

        for i in range(batch_count):
            try:
                resp = client.models.generate_content(model=model_used, contents=contents, config=gen_cfg)
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
                    op_log += f"Batch {i+1}: no images. Text: {resp_text[:160]}...\n"

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

    # ------------------------- main entry -------------------------
    def nano_gemini_generate(
        self,
        prompt,
        operation,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        api_key="",
        batch_count=1,
        temperature=0.7,
        quality="high",
        aspect_ratio="1:1",
        character_consistency=True,
        enable_safety=True,
    ):
        # api key override from widget
        if api_key.strip():
            self.api_key = api_key.strip()
            save_config({"GEMINI_API_KEY": self.api_key})

        if not self.api_key:
            msg = (
                "NANOGEMINI ERROR: No API key!\n"
                "Gemini 2.5 image models require a PAID API key.\n"
                "Set env GEMINI_API_KEY or fill the api_key field."
            )
            return (self.create_placeholder_image(), msg)

        try:
            encoded_refs = self.prepare_images_for_api(
                reference_image_1,
                reference_image_2,
                reference_image_3,
                reference_image_4,
                reference_image_5,
            )
            has_refs = len(encoded_refs) > 0

            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_refs, aspect_ratio, character_consistency
            )
            if final_prompt.startswith("Error:"):
                return (self.create_placeholder_image(), final_prompt)

            if quality == "high":
                final_prompt += " Use the highest quality settings available."

            header = [
                "NANOGEMINI OPERATION LOG",
                f"Operation: {operation}",
                f"Refs: {len(encoded_refs)}",
                f"Batch: {batch_count}",
                f"Temp: {temperature}",
                f"Quality: {quality}",
                f"AR: {aspect_ratio}",
                f"Keep character: {character_consistency}",
                f"Safety: {enable_safety}",
                f"Prompt: {final_prompt[:200]}...",
            ]
            op_log = "\n".join(header) + "\n\n"

            tensors, api_log = self._call_gemini_images(
                final_prompt, encoded_refs, temperature, batch_count, enable_safety
            )
            op_log += api_log

            if tensors:
                out = torch.cat(tensors, dim=0)
                est_cost = len(tensors) * 0.04  # ước tính gần đúng (cập nhật theo pricing của bạn)
                op_log += f"\nEstimated cost: ~${est_cost:.3f}\nOK: {len(tensors)} image(s)."
                return (out, op_log)

            op_log += "\nNo images were generated."
            return (self.create_placeholder_image(), op_log)

        except Exception as e:
            return (
                self.create_placeholder_image(),
                f"NANOGEMINI ERROR: {e}\nCheck API key, internet, or dependencies.",
            )


# ------------------------------------------------------------
# Node registration (BẮT BUỘC)
# ------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ComfyUI_NanoGemini": ComfyUI_NanoGemini,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_NanoGemini": "NanoGemini (Gemini 2.5 Image)",
}

# Nhóm hiển thị
ComfyUI_NanoGemini.CATEGORY = "Nano / Gemini 2.5"
