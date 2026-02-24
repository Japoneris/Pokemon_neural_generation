"""Central configuration for Pokemon WGAN-GP v2 (enhanced architecture)."""

# Image settings
IMAGE_SIZE = 64
NUM_CHANNELS = 3
LATENT_DIM = 128

# Architecture
GEN_FEATURE_MAPS = 96
DISC_FEATURE_MAPS = 96

# Training
BATCH_SIZE =32
NUM_EPOCHS = 2000
LEARNING_RATE_G = 1e-4   # Equal to D: large discriminator already has spectral norm + GP
LEARNING_RATE_D = 1e-4
BETA1 = 0.0
BETA2 = 0.9
N_CRITIC = 3             # Reduced from 5: gives generator more frequent updates
LAMBDA_GP = 10.0

# EMA
EMA_DECAY = 0.995        # Reduced from 0.999: faster warm-up on early training

# Data augmentation
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER = (0.1, 0.1, 0.1, 0.02)

# Paths
DATA_DIR = "../../dataset"
OUTPUT_DIR = "../../outputs/WGAN_large"
CHECKPOINT_DIR = "../../outputs/WGAN_large/checkpoints"
SAMPLE_DIR = "../../outputs/WGAN_large/samples"

# Logging
SAVE_INTERVAL = 50
SAMPLE_INTERVAL = 10
NUM_SAMPLE_IMAGES = 64
