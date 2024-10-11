#Description:

This repository contains files related to a trained 3D cGAN (Conditional Generative Adversarial Network) model designed to correct artifacts caused by breathing motion in diffusion-weighted images (DWI). Both the generator and discriminator networks are based on a pretrained 3D U-Net architecture, which has been fine-tuned for artifact correction in medical imaging.

UNetGAN_2.ipynb contains the code to train the cGAN (without w&B)
UNetGAN_wandb.ipynb contains the code to train the cGAN (with integration of w&B for hyperparamters tuning)

