"""Central configuration for Pokemon DDPM v3."""

# Image settings
IMAGE_SIZE = 64
NUM_CHANNELS = 3

# UNet architecture (sized for ~4 GB GPU)
MODEL_CHANNELS = 32           # Base channel count
CHANNEL_MULT = (1, 2, 4, 8)  # -> 32, 64, 128, 256
NUM_RES_BLOCKS = 2            # Residual blocks per resolution level
ATTENTION_RESOLUTIONS = (16,) # Apply attention at 16x16
NUM_HEADS = 4                 # Multi-head attention heads
DROPOUT = 0.1                 # Dropout in residual blocks
TIME_EMB_DIM = 128            # Dimension of time embedding (4 * MODEL_CHANNELS)

# Diffusion process
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
BETA_SCHEDULE = "linear"      # "linear" or "cosine"

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 1000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.999
GRAD_CLIP = 1.0               # Gradient clipping max norm
WARMUP_STEPS = 500            # Linear warmup steps

# EMA
EMA_DECAY = 0.9999

# Data augmentation (same as v1/v2)
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER = (0.1, 0.1, 0.1, 0.02)

# Paths
DATA_DIR = "../../dataset"
OUTPUT_DIR = "../../outputs/stable_diffusion"
CHECKPOINT_DIR = "../../outputs/stable_diffusion/checkpoints"
SAMPLE_DIR = "../../outputs/stable_diffusion/samples"

# Logging
SAVE_INTERVAL = 50
SAMPLE_INTERVAL = 10
NUM_SAMPLE_IMAGES = 16

# Sampling
DDIM_STEPS = 50               # Steps for accelerated DDIM sampling
DDIM_ETA = 0.0                # 0.0 = deterministic DDIM, 1.0 = DDPM
