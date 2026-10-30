"""Model loading, prompt building and flow gates, shared by train_lora.py and eval_gqa.py."""
import copy
import sys

import torch
from PIL import Image

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

DTYPE = torch.bfloat16  # training and evaluation both run in bf16 on the A40


def load_model(model_path):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path),
        device_map="auto", torch_dtype="bfloat16", attn_implementation=None,
    )
    model.get_vision_tower().to(dtype=DTYPE)  # the builder leaves CLIP in a different dtype
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer, model, image_processor


def build_prompt(human, answer=None, conv_mode="vicuna_v1"):
    """human is "<image>\\n<question>\\nAnswer the question using a single word or phrase."
    With answer=None the prompt ends at "ASSISTANT:", ready for generation."""
    conv = copy.deepcopy(conv_templates[conv_mode])
    conv.append_message(conv.roles[0], human)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def tokenize(prompt, tokenizer):
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")


def load_image(path, image_processor, model_config):
    image = Image.open(path).convert("RGB")
    return process_images([image], image_processor, model_config)[0].to(DTYPE), image.size


class LoraTokens:
    """LoRA at text positions only (keep="text") or image positions only (keep="image"), train_lora.py --lora-tokens.

    Zeroes the output of every lora_A at the other positions, so the adapter cannot be merged.
    Call set_spans() before every forward or generate.
    """

    def __init__(self, model, keep):
        self.keep_image = keep == "image"
        self.left_pad = getattr(model.config, "tokenizer_padding_side", "right") == "left"
        self.spans, self.image_len, self.mask = None, None, None
        model.get_model().mm_projector.register_forward_hook(
            lambda module, inp, out: setattr(self, "image_len", out.shape[-2]))
        for name, module in model.named_modules():
            if name.endswith("lora_A.default"):
                module.register_forward_hook(self._hook)

    def set_spans(self, input_ids, attention_mask=None):
        """Per sample, before the image is expanded: image position and length."""
        self.spans = [(int((ids == IMAGE_TOKEN_INDEX).nonzero()[0]),
                       int(attention_mask[i].sum()) if attention_mask is not None else len(ids))
                      for i, ids in enumerate(input_ids)]
        self.mask = None

    def _hook(self, module, args, out):
        q_len = out.shape[1]
        if q_len == 1:  # generating answer tokens with the cache: text
            return out * (not self.keep_image)
        if self.mask is None or self.mask.shape[1] != q_len:
            if max(n - 1 + self.image_len for _, n in self.spans) != q_len:
                sys.exit(f"LoRA tokens: expected {q_len} tokens after expanding the image")
            image = torch.zeros(len(self.spans), q_len, 1, dtype=torch.bool, device=out.device)
            for i, (pos, n) in enumerate(self.spans):
                start = pos + (q_len - (n - 1 + self.image_len) if self.left_pad else 0)
                image[i, start:start + self.image_len] = True
            self.mask = image if self.keep_image else ~image
        return out * self.mask.to(out.dtype)


FLOW_GATE_LAYERS = range(5, 16)  # where the GQA knockout shows the Image->Question flow


class FlowGates(torch.nn.Module):
    """Learnable knobs on the Image->Question flow (train_lora.py --flow-gates).

    In each layer of FLOW_GATE_LAYERS and each head, log_scale is added to the attention logits from the prompt
    text after the image (question, then "ASSISTANT:") to the image tokens. 0 = unchanged (the start), > 0 = the
    text takes in more from the image, < 0 = less. Answer tokens are not gated. Needs eager or sdpa attention.
    Call set_spans() before every forward or generate.
    With knockout=True the Image->Question attention is cut instead (train_lora.py --contrast flow), only in the
    samples where set_spans(knock=...) is True.
    """

    def __init__(self, model, layers=FLOW_GATE_LAYERS, knockout=False):
        super().__init__()
        scale = torch.zeros(len(layers), model.config.num_attention_heads, device=model.device)
        if knockout:
            self.register_buffer("log_scale", scale.fill_(torch.finfo(DTYPE).min))
        else:
            self.log_scale = torch.nn.Parameter(scale)
        self.left_pad = getattr(model.config, "tokenizer_padding_side", "right") == "left"
        self.spans, self.image_len, self.knock = None, None, None
        model.get_model().mm_projector.register_forward_hook(
            lambda module, inp, out: setattr(self, "image_len", out.shape[-2]))
        for gate, layer in enumerate(layers):
            model.get_model().layers[layer].self_attn.register_forward_pre_hook(self._hook(gate), with_kwargs=True)

    def set_spans(self, input_ids, labels=None, attention_mask=None, knock=None):
        """Per sample, before the image is expanded: image position, prompt end (first answer token) and length."""
        self.knock = knock
        self.spans = []
        for i, ids in enumerate(input_ids):
            n = int(attention_mask[i].sum()) if attention_mask is not None else len(ids)
            end = int((labels[i] != IGNORE_INDEX).nonzero()[0]) if labels is not None else n
            self.spans.append((int((ids == IMAGE_TOKEN_INDEX).nonzero()[0]), end, n))

    def _hook(self, gate):
        def hook(module, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            q_len, n_img = hidden.shape[1], self.image_len
            if self.spans is None or q_len == 1:  # one query = generating answer tokens with the cache
                return None
            if max(n - 1 + n_img for _, _, n in self.spans) != q_len:
                sys.exit(f"Flow gates: expected {q_len} tokens after expanding the image")
            mask = kwargs.get("attention_mask")
            if mask is None:  # sdpa without padding gets no mask: build the causal one
                mask = torch.full((q_len, q_len), torch.finfo(hidden.dtype).min, device=hidden.device,
                                  dtype=hidden.dtype).triu(1)[None, None]
            elif mask.dim() != 4:
                sys.exit("Flow gates need eager or sdpa attention (flash attention takes no per-position mask)")
            bias = torch.zeros(len(self.spans), self.log_scale.shape[1], q_len, mask.shape[-1],
                               device=mask.device, dtype=mask.dtype)
            for i, (image, end, n) in enumerate(self.spans):
                if self.knock is not None and not self.knock[i]:
                    continue
                offset = q_len - (n - 1 + n_img) if self.left_pad else 0
                img_start, img_end = offset + image, offset + image + n_img
                bias[i, :, img_end:offset + end - 1 + n_img, img_start:img_end] = \
                    self.log_scale[gate].to(mask.dtype)[:, None, None]
            kwargs["attention_mask"] = mask + bias
            return args, kwargs
        return hook
