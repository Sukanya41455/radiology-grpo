# src/vision_model.py

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTImageProcessor,
)
from dataclasses import dataclass
from typing import Tuple

from .config import TrainConfig


@dataclass
class VisionTrainConfig(TrainConfig):
    image_size: int = 224
    encoder_decoder_model_name: str = "nlpconnect/vit-gpt2-image-captioning"
    num_epochs: int = 6
    batch_size: int = 2      
    max_length: int = 64
    lr: float = 3e-5


def load_vision_model_and_processor(cfg: VisionTrainConfig):
    """
    Load a ViT-GPT2 encoder-decoder model for image-to-text.
    We'll fine-tune this on IU X-ray (image -> findings).
    """
    model = VisionEncoderDecoderModel.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    # Image processor for ViT encoder
    image_processor = ViTImageProcessor.from_pretrained(
        cfg.encoder_decoder_model_name
    )

    # Tokenizer for GPT2 decoder
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder_decoder_model_name
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Important: set special tokens & generation params
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id


    # Reasonable defaults for generation
    model.config.max_length = cfg.max_length
    model.config.num_beams = 3

    return model, tokenizer, image_processor
