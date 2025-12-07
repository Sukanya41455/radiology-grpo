# src/vision_model.py

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
)
from dataclasses import dataclass
from typing import Tuple

from .config import TrainConfig


@dataclass
class VisionTrainConfig(TrainConfig):
    image_size: int = 224
    # Separate encoder / decoder so we can upgrade the text model
    encoder_name: str = "google/vit-base-patch16-224-in21k"
    decoder_name: str = "gpt2-medium"   # upgraded from plain gpt2
    num_epochs: int = 6
    batch_size: int = 2      # keep small for GPU memory with gpt2-medium
    max_length: int = 64     # you can try 96/128 later if GPU allows
    lr: float = 3e-5


def load_vision_model_and_processor(cfg: VisionTrainConfig):
    """
    Load a ViT + GPT-2-medium encoder-decoder model for image-to-text.
    We'll fine-tune this on IU X-ray (image -> findings).
    """
    # Build encoder-decoder from separate vision + text backbones
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        cfg.encoder_name,
        cfg.decoder_name,
    )

    # Image processor for ViT encoder
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.encoder_name
    )

    # Tokenizer for GPT-2-medium decoder
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.decoder_name
    )

    # Make sure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Also ensure BOS exists (GPT-2 usually has no BOS by default)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    # Important: set special tokens & generation params
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Make sure decoder vocab size matches tokenizer
    if hasattr(model.config, "decoder"):
        model.config.vocab_size = model.config.decoder.vocab_size

    # Reasonable defaults for generation
    model.config.max_length = cfg.max_length
    model.config.num_beams = 3

    return model, tokenizer, image_processor
