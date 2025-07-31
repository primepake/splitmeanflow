# SplitMeanFlow - Unofficial Implementation


This is an **unofficial implementation** of SplitMeanFlow from the paper **"SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling"** by ByteDance. [[arxiv](https://arxiv.org/abs/2507.16884)]. This implementation is based on my understanding of the paper and uses the DiT architecture as the backbone.

**Note: This is not the official implementation. For the official code, please refer to the authors' repository when it becomes available.**

## Overview

SplitMeanFlow introduces a novel approach for few-step generation by learning average velocity fields through an algebraic consistency principle. The key innovation is avoiding expensive Jacobian-Vector Product (JVP) computations while achieving one-step generation quality comparable to traditional multi-step methods.

## Setup

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate splitmeanflow
```


## Training

### Stage 1: Teacher Model
First, train a standard flow matching teacher:
```bash
python train_teacher_mnist.py
```

Samples in training:

![Samples](assets/trajectory_cfg_5-99999.gif)

The training log here: ![training log comet](assets/mnist_log.png)

You can download the teacher model at [model](https://github.com/primepake/splitmeanflow/releases/tag/mnist)
### Stage 2: Student Model  
Then train the SplitMeanFlow student:
```bash
python train_student.py --teacher_path checkpoints/teacher_final.pt
```

The training follows Algorithm 1 from the paper with some modifications based on my interpretation.

## Usage

```python
from models.splitmeanflow import SplitMeanFlowDiT
from sampler import Sampler

# Load model
model = SplitMeanFlowDiT(
    input_size=32,
    patch_size=2,
    in_channels=3,
    dim=384,
    depth=12,
    num_heads=6,
    num_classes=10,
).cuda()

model.load_state_dict(torch.load('checkpoints/student_best.pt'))

# One-step sampling
sampler = Sampler(model)
samples = sampler.sample_onestep(batch_size=16)
```

## Key Implementation Details

### Interval Splitting Consistency Loss
My implementation of the core training objective:
```python
def interval_splitting_loss(model, x, teacher_model=None, flow_ratio=0.5):
    # Sample times r < s < t
    r, t = sorted(torch.rand(2))
    lambda_val = torch.rand(1)
    s = (1 - lambda_val) * t + lambda_val * r
    
    # Implementation details following Algorithm 1
    # ...
```

### Architecture Modifications
The DiT model is modified to accept interval inputs:
```python
class SplitMeanFlowDiT(nn.Module):
    def forward(self, x, r, t, y=None):
        # Added interval embedding layer
        interval_emb = self.interval_embed(torch.stack([r, t], dim=-1))
        # ... rest of DiT forward pass
```

## Experimental Results

**Disclaimer: These are preliminary results from my implementation and may not match the paper's performance.**

### CIFAR-10 (200k steps)
- 1-step generation: FID ~8.5 (paper reports ~3.6)
- 2-step generation: FID ~6.2
- Training time: ~18 hours on single RTX 3090

The gap in performance could be due to:
- Implementation differences
- Hyperparameter tuning
- Missing components

## Known Issues & Differences from Paper

1. **Performance Gap**: My implementation doesn't quite reach the paper's reported metrics
2. **Training Stability**: Occasionally encounter NaN losses - added gradient clipping as workaround
3. **CFG Handling**: My interpretation of CFG-free training might differ from the official approach
4. **Time Sampling**: Using uniform sampling instead of log-normal as mentioned in paper

## TODOs

- [ ] Implement log-normal time sampling
- [ ] Add proper evaluation pipeline (FID, IS, etc.)
- [ ] Experiment with different architectures (U-Net, etc.)
- [ ] Add visualization tools for debugging
- [ ] Implement the MeanFlow baseline for comparison
- [ ] Multi-GPU distributed training

## Contributing

This is an unofficial implementation created for learning purposes. If you find any bugs or have suggestions for improvements, please open an issue or submit a PR. If you have insights about the paper or the official implementation, I'd love to hear them!

## Citation

Please cite the original paper:
```bibtex
@article{guo2025splitmeanflow,
  title={SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling},
  author={Guo, Yi and Wang, Wei and Yuan, Zhihang and others},
  journal={arXiv preprint arXiv:2507.16884},
  year={2025}
}
```

## Acknowledgments

- Original paper by ByteDance Seed team
- DiT implementation adapted from [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- Training loop structure inspired by [Just-a-DIT](https://github.com/ArchiMickey/Just-a-DiT)
- Thanks to the authors for the clear paper and algorithm descriptions

## License

This unofficial implementation is released under MIT License. Please refer to the original paper and official implementation for any commercial use.