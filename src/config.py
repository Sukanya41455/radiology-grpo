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
    group_size: int = 2           # smaller K to start, cheaper & more stable
    max_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 5e-6              # smaller LR for RL updates
    weight_decay: float = 0.01
    epsilon_clip: float = 0.2
    kl_coeff: float = 0.02        # slightly stronger KL to keep close to supervised
    entropy_coeff: float = 0.003  # a bit more entropy to avoid collapse
    max_grad_norm: float = 1.0
    device: str = "cuda"
