# Pokemon Generation

This project use the technics of stable diffusion to generate new pokemon from randomness.

`data/`: Where are located images



# Train (1000 epochs by default)
source venv/bin/activate
python3 train.py

# Resume from a checkpoint
python3 train.py --resume outputs/checkpoints/checkpoint_epoch_0500.pt

# Generate images from a trained model
python3 generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0950.pt --num 16


Sample grids are saved every 10 epochs to outputs/samples/, and checkpoints every 50 epochs to outputs/checkpoints/.

