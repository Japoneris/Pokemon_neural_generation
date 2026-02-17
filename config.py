"""Central configuration for Pokemon WGAN-GP."""

# Image settings
IMAGE_SIZE = 64
NUM_CHANNELS = 3
LATENT_DIM = 128

# Architecture
GEN_FEATURE_MAPS = 64
DISC_FEATURE_MAPS = 64

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 1000
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
BETA1 = 0.0
BETA2 = 0.9
N_CRITIC = 5
LAMBDA_GP = 10.0

# Data augmentation
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER = (0.1, 0.1, 0.1, 0.02)

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
SAMPLE_DIR = "outputs/samples"

# Logging
SAVE_INTERVAL = 50
SAMPLE_INTERVAL = 10
NUM_SAMPLE_IMAGES = 64
