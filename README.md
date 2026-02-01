# UltraGen: High-Resolution Video Generation with Hierarchical Attention [AAAI 2026]

<p align="center">
  <a href="https://sjtuplayer.github.io/projects/UltraGen/">🌐 Project Page</a> •
  <a href="https://arxiv.org/abs/2510.18775">📄 Paper</a> •
  <a href="https://huggingface.co/JTUplayer/Ultragen">🤗 Hugging Face</a> •
  <a href="https://github.com/zhangzjn/T3-Video">🚀 T3-Video (4K)</a> •
  <a href="#citation">📚 Citation</a>
</p>


---

## Overview

**UltraGen** is a novel video generation framework that enables **efficient and end-to-end native high-resolution** video synthesis. This is the official implementation of our AAAI 2026 paper.

### Highlights

- **🎯 First Native 4K Model**: Achieve native high-quality 4K video generation, eliminating "pseudo-high-resolution" limitations
- **⚡ Efficient Architecture**: Hierarchical dual-branch attention with 4.78× speedup for 4K and 2.69× speedup for 1080P
- **🏆 Superior Quality**: State-of-the-art performance across all metrics (HD-FVD, HD-MSE, HD-LPIPS, CLIP scores)
- **💡 Novel Design**: Global-local attention decomposition solving O((T·H·W)²) complexity bottleneck

For more details, visit our [project page](https://sjtuplayer.github.io/projects/UltraGen/) or read the [paper](https://arxiv.org/abs/2510.18775).

### Key Features

- 🎬 **Native High-Resolution**: 1080P/2K/4K video generation without super-resolution pipeline
- ⚡ **Hierarchical Attention**: Dual-branch architecture (local + global) for efficient computation
- 🎨 **Flexible Control**: Fine-grained control over generation parameters
- 🔧 **Easy to Use**: Simple one-line command interface
- 🚀 **Production Ready**: Optimized for both research and practical applications

### Additional Resources

- 🌐 **Project Page**: [https://sjtuplayer.github.io/projects/UltraGen/](https://sjtuplayer.github.io/projects/UltraGen/)
- 📄 **Paper**: [arXiv:2510.18775](https://arxiv.org/abs/2510.18775)
- 🚀 **T3-Video (4K)**: [10x+ faster 4K generation](https://github.com/zhangzjn/T3-Video)

---

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/your-org/UltraGen.git
cd UltraGen

# Create environment
conda create -n ultragen python=3.10 -y
conda activate ultragen

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -e .
```

### Download Models

**Hugging Face**: [https://huggingface.co/JTUplayer/Ultragen](https://huggingface.co/JTUplayer/Ultragen)

```bash
# Download base model (required)
python -c "from modelscope import snapshot_download; \
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='checkpoints/Wan2.1-T2V-1.3B')"

# Download finetuned checkpoint from Hugging Face
# Visit https://huggingface.co/JTUplayer/Ultragen for the latest model
huggingface-cli download JTUplayer/Ultragen --local-dir checkpoints/ultragen
```

---

## Quick Start

### Generate Your First Video

```bash
# Simple one-liner
bash inference.sh "The video captures a breathtaking view of a mountainous landscape at sunrise, with a sea of clouds enveloping the valleys and rolling hills."
```

Output: `outputs/video_0000.mp4`

### More Examples

```bash
# Underwater scene
bash inference.sh "The video captures an underwater scene featuring a clownfish swimming near a sea anemone in a vibrant coral reef environment"

# Urban landscape
bash inference.sh "The video showcases a stunning aerial view of a modern cityscape with a mix of historic and contemporary architecture."

# Natural phenomena
bash inference.sh "The video captures the intense and fiery eruption of a volcano, showcasing the raw power of nature as molten lava flows and spews into the air."

# Time-lapse
bash inference.sh "A time-lapse video captures a cityscape at night with a lightning strike illuminating the sky above a busy highway."
```

> **Note**: UltraGen works best with **landscape and scenic content**. Complex human actions and fast movements may have limited support in the current version.

### Advanced Usage

```bash
# Use Python API for more control
python tools/inference/generate.py \
  --model_dir checkpoints/Wan2.1-T2V-1.3B \
  --checkpoint checkpoints/ultragen_1080p.ckpt \
  --prompt "Your amazing prompt" \
  --output_dir outputs/custom
```

### Batch Generation

```bash
# Create prompts.txt with one prompt per line
cat > prompts.txt << EOF
The video captures a breathtaking view of a mountainous landscape at sunrise, with a sea of clouds enveloping the valleys and rolling hills.
The video showcases a stunning aerial view of a modern cityscape with a mix of historic and contemporary architecture.
The video captures an underwater scene featuring a clownfish swimming near a sea anemone in a vibrant coral reef environment
EOF

# Generate all videos
python tools/inference/generate.py \
  --checkpoint checkpoints/ultragen_1080p.ckpt
  --model_dir checkpoints/Wan2.1-T2V-1.3B \
  --prompt_file prompts.txt
```

---

## Technical Details

UltraGen features a **hierarchical dual-branch attention** architecture:

### Architecture Innovation

1. **Global-Local Attention Decomposition**
   - Local branch: High-fidelity regional details
   - Global branch: Overall semantic consistency
   - Avoids O((T·H·W)²) complexity

2. **Spatially Compressed Global Modeling**
   - Efficient learning of global dependencies
   - Reduced computational overhead

3. **Hierarchical Cross-Window Local Attention**
   - Enhanced information flow across windows
   - Lower computational cost

### Performance

Compared to baseline Wan2.1:

| Method | Resolution | Speedup | HD-FVD ↓ | HD-MSE ↑ | HD-LPIPS ↑ |
|--------|------------|---------|----------|----------|------------|
| Wan2.1 | 1080P | 1.00× | 245.37 | 375.82 | 0.5201 |
| **UltraGen** | **1080P** | **2.69×** | **214.12** | **390.19** | **0.5455** |
| Wan2.1 | 4K | 1.00× | 486.29 | 362.45 | 0.6102 |
| **UltraGen** | **4K** | **4.78×** | **424.61** | **386.01** | **0.6450** |

**Key Improvements**:
- 🚀 **2.69× faster** at 1080P, **4.78× faster** at 4K
- 📊 Better quality across all metrics (lower FVD, higher MSE and LPIPS)
- ⚡ Efficient hierarchical attention without sacrificing quality

For more technical details, see our [paper](https://arxiv.org/abs/2510.18775).

---

## 4K Generation with T3-Video

For even faster 4K generation (10x+ acceleration), check out **[T3-Video](https://github.com/zhangzjn/T3-Video)**:

- **Transform Trained Transformer** architecture
- Native 4K support (3840x2176, 81 frames)
- More stable and consistent results
- Built on UltraGen base

**Paper**: [Transform Trained Transformer: Accelerating Naive 4K Video Generation Over 10×](https://arxiv.org/abs/2512.13492)

---

## Training

### Prepare Dataset

**Recommended Dataset**: We recommend using [UltraVideo](https://github.com/xzc-zju/UltraVideo) - a high-quality UHD 4K video dataset with comprehensive captions (NeurIPS 2025).

```bash
# Download UltraVideo dataset
huggingface-cli download --repo-type dataset APRIL-AIGC/UltraVideo \
  --local-dir ./data/UltraVideo --resume-download
```

**Custom Dataset Format**:

```
dataset/
├── videos/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
└── captions.json  # {"video001.mp4": "description", ...}
```

### Single-GPU Training

```bash
python tools/training/train.py \
  --task train \
  --dataset_path data/UltraVideo \
  --output_path experiments/my_model \
  --dit_path checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --train_architecture full \
  --height 1088 --width 1920 --num_frames 81 \
  --learning_rate 1e-4 \
  --use_gradient_checkpointing
```

### Multi-GPU Training (Single Machine)

```bash
# 8 GPUs on one machine
torchrun --nproc_per_node=8 tools/training/train.py \
  --task train \
  --dataset_path data/UltraVideo \
  --output_path experiments/my_model \
  --dit_path checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --train_architecture full \
  --learning_rate 1e-4 \
  --use_gradient_checkpointing
```

### Multi-Machine Distributed Training

For large-scale training across multiple machines, use `scripts/training/run.sh`:

```bash
# 1. Create a hosts file listing all machine IPs (one per line)
cat > hosts.txt << EOF
192.168.1.10
192.168.1.11
192.168.1.12
192.168.1.13
EOF

# 2. Configure the training script
# Edit scripts/training/run.sh to set:
#   - N_NODES: number of machines
#   - DATASET_PATH: path to UltraVideo or your dataset
#   - DIT_PATH: base model checkpoint path
#   - OUTPUT_PATH: where to save trained models
#   - Training hyperparameters (learning rate, resolution, etc.)

# 3. Launch distributed training on each node
bash scripts/training/run.sh
```

**Training Configuration**:
- Automatic node rank detection and master node setup
- Supports both 1080P and 4K training (adjust resolution parameters)
- Compatible with pssh for parallel node deployment

---

## Project Structure

```
UltraGen/
├── README.md                  # This file
├── inference.sh               # Quick inference script
├── tools/
│   ├── inference/
│   │   └── generate.py        # Main inference script
│   └── training/
│       ├── train.py           # Training script
├── diffsynth/                 # Core library
├── checkpoints/               # Model checkpoints
├── outputs/                   # Generated videos
└── prompts_example.txt        # Example prompts
```


---

## Related Projects

- 🚀 **[T3-Video](https://github.com/zhangzjn/T3-Video)**: Transform Trained Transformer for 10x+ faster 4K generation
- 🎨 **[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)**: Diffusion synthesis framework
- 📹 **[Wan2.1](https://github.com/Wan-Video/Wan2.1)**: Base video generation model

## Acknowledgements

We thank the contributors of DiffSynth Studio, Wan-Video, and the open-source community for their valuable work.

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

## Citation

If you find UltraGen useful in your research, please cite:

```bibtex
@inproceedings{hu2026ultragen,
  title={UltraGen: High-Resolution Video Generation with Hierarchical Attention},
  author={Hu, Teng and Zhang, Jiangning and Su, Zihan and Yi, Ran},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

**Paper**: [https://arxiv.org/abs/2510.18775](https://arxiv.org/abs/2510.18775)  
**Project Page**: [https://sjtuplayer.github.io/projects/UltraGen/](https://sjtuplayer.github.io/projects/UltraGen/)

### T3-Video Citation

If you use T3-Video for 4K generation:

```bibtex
@misc{t3video,
    title={Transform Trained Transformer: Accelerating Naive 4K Video Generation Over 10×}, 
    author={Jiangning Zhang and Junwei Zhu and Teng Hu and Yabiao Wang and Donghao Luo and Weijian Cao and Zhenye Gan and Xiaobin Hu and Zhucun Xue and Chengjie Wang},
    year={2025},
    eprint={2512.13492},
    archivePrefix={arXiv}
}
```

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/UltraGen/issues)
- **Project Page**: [https://sjtuplayer.github.io/projects/UltraGen/](https://sjtuplayer.github.io/projects/UltraGen/)
- **Paper**: [arXiv:2510.18775](https://arxiv.org/abs/2510.18775)

---

<p align="center">
<strong>UltraGen: High-Resolution Video Generation with Hierarchical Attention</strong><br>
AAAI 2026<br>
Made with ❤️ by Shanghai Jiao Tong University & Zhejiang University
</p>
