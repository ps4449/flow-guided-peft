"""Model loading and prompt building shared by train_lora.py and eval_gqa.py."""
import copy

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
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
