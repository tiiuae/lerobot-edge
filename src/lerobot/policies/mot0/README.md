# MOT0 policy

This repository contains the implementation of the MoT policy, adapted from the `pi05` policy in the same repo.

It is designed to support:
1. Pure MoT training  
2. MoT with both discrete and continuous action generation  
3. Integration of VLM data during training  
4. Reasoning within the MoT structure, including:
   - Chain-of-thought (CoT) reasoning  
   - Visual reasoning  
   - Latent reasoning  

## What has been achieved

- [x] MoT implementation: follows the same structure as `pi05`  
- [x] Training and evaluation: stable loss decrease and reasonable performance on LIBERO  
- [x] Implementation of Hybrid action generation: supports both discrete and continuous losses  
- [x] Speed optimization in Hybrid action generation:
  - Gradient checkpointing  
  - Compilation  
  - DeepSpeed ZeRO-2 does not work due to NaN issues (caused by setting all parameters to bf16)  
- [x] Accuracy validation in Hybrid action generation:
  - Setting the discrete token loss to zero achieves similar performance as using continuous loss alone on LIBERO  

## TODO LIST

- [ ] Grid search for the best loss weight ratio between MSE (flow matching) and CE (discrete)  
- [ ] Add LIBERO-Plus to the benchmark  
- [ ] Data loader supporting mixed VLM and action data (captioning)  
- [ ] Data loader supporting mixed VLM and action data (grounding)  
- [ ] Study the impact of VLM data (type and ratio) on speed and performance  

# Install

Follow the same installation steps as in: https://github.com/huggingface/lerobot

Note: Python version used is 3.12 (tested)

# Train, Resume and Eval

Please refer to `train_test_script.sh`.

Update the following as needed:
- All paths in the script.
- W&B project name  
- Job name  
