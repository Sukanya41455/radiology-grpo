from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "microsoft/biogpt"  # instead of "distilgpt2"
    max_length: int = 128
    batch_size: int = 4      # if you see OOM in GRPO, drop to 2
    num_epochs: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.01
    device: str = "cuda"



@dataclass
class GRPOConfig:
    group_size: int = 4           # K candidates per prompt
    max_new_tokens: int = 40      # a bit shorter
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 5e-6              # smaller LR for stability
    weight_decay: float = 0.01
    epsilon_clip: float = 0.1     # smaller PPO clip
    kl_coeff: float = 0.1         # much stronger KL (was 0.01)
    entropy_coeff: float = 0.001
    max_grad_norm: float = 1.0
    device: str = "cuda"

