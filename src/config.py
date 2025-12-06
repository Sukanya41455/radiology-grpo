from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "distilgpt2"
    max_length: int = 128
    batch_size: int = 4
    num_epochs: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.01
    device: str = "cuda"


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 5e-6          # was 1e-5
    weight_decay: float = 0.01
    epsilon_clip: float = 0.2
    kl_coeff: float = 0.05    # was 0.01
    entropy_coeff: float = 0.01  # was 0.001
    max_grad_norm: float = 1.0
    device: str = "cuda"

