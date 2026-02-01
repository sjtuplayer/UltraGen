#!/usr/bin/env python3
"""
UltraGen Video Generation - Inference Tool
Generate high-quality videos from text prompts using the UltraGen model.
"""

import os
import sys
import argparse
import json
import torch
import warnings
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from diffsynth import ModelManager, WanVideoPipeline, save_video


class VideoGenerator:
    """Video generation pipeline wrapper"""
    
    def __init__(self, model_dir, checkpoint_path=None, mode='full', device='cuda'):
        """
        Initialize the video generator
        
        Args:
            model_dir: Directory containing base model files
            checkpoint_path: Path to finetuned checkpoint (optional)
            mode: 'full' or 'lora'
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_dir = Path(model_dir)
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.device = device
        
        logging.info("Initializing UltraGen Video Generator...")
        logging.info(f"Model directory: {self.model_dir}")
        if checkpoint_path:
            logging.info(f"Checkpoint: {checkpoint_path}")
        
        # Load base models
        model_manager = ModelManager(device="cpu")
        
        model_paths = [
            str(self.model_dir / "diffusion_pytorch_model.safetensors"),
            str(self.model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(self.model_dir / "Wan2.1_VAE.pth"),
        ]
        
        # Validate model files
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        logging.info("Loading base models...")
        model_manager.load_models(
            model_paths,
            torch_dtype=torch.bfloat16,
        )
        
        # Initialize pipeline
        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, 
            torch_dtype=torch.bfloat16, 
            device=device
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            if mode == 'lora':
                logging.info(f"Loading LoRA weights...")
                model_manager.load_lora(checkpoint_path, lora_alpha=0.5)
            else:
                logging.info(f"Loading full checkpoint...")
                self.pipe.dit.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # Enable VRAM management
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        logging.info("Model initialized successfully!")
    
    def generate(
        self,
        prompt,
        negative_prompt=None,
        height=1088,
        width=1920,
        num_frames=81,
        num_inference_steps=30,
        cfg_scale=5.0,
        seed=0,
        tiled=True
    ):
        """
        Generate video from prompt
        
        Args:
            prompt: Text description of the video
            negative_prompt: Things to avoid in generation
            height: Video height in pixels
            width: Video width in pixels
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps (higher = better quality)
            cfg_scale: Classifier-free guidance scale (higher = more prompt adherence)
            seed: Random seed for reproducibility
            tiled: Use tiled VAE for memory efficiency
        
        Returns:
            Generated video tensor
        """
        if negative_prompt is None:
            negative_prompt = (
                "Aerial view, overexposed, low quality, deformation, "
                "bad composition, bad hands, bad teeth, bad eyes, "
                "bad limbs, distortion, blurring, text, subtitles"
            )
        
        logging.info(f"Generating video...")
        logging.info(f"  Prompt: {prompt}")
        logging.info(f"  Resolution: {width}x{height}, {num_frames} frames")
        logging.info(f"  Steps: {num_inference_steps}, CFG: {cfg_scale}, Seed: {seed}")
        
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            seed=seed,
            tiled=tiled,
        )
        
        return video


def load_prompts(prompt_file):
    """Load prompts from text or JSON file"""
    ext = Path(prompt_file).suffix.lower()
    
    if ext == '.json':
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = []
                for item in data:
                    if isinstance(item, dict):
                        prompts.append(item.get('prompt_en') or item.get('prompt') or str(item))
                    else:
                        prompts.append(str(item))
                return prompts
            return [data.get('prompt', str(data))]
    else:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser(
        description='UltraGen Video Generation - Generate high-quality videos from text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1080P video
  python generate.py --model_dir checkpoints/Wan2.1-T2V-1.3B \\
                     --prompt "A dog running in a park"
  
  # Use finetuned checkpoint
  python generate.py --model_dir checkpoints/Wan2.1-T2V-1.3B \\
                     --checkpoint checkpoints/ultragen_1080p.ckpt \\
                     --prompt "A beautiful sunset"
  
  # Generate 4K video
  python generate.py --model_dir checkpoints/Wan2.1-T2V-1.3B \\
                     --checkpoint checkpoints/ultragen_4k.ckpt \\
                     --prompt "Mountains at dawn" \\
                     --width 3840 --height 2160 --num_frames 29
  
  # Batch generation
  python generate.py --model_dir checkpoints/Wan2.1-T2V-1.3B \\
                     --prompt_file prompts.txt \\
                     --output_dir outputs/batch1
        """
    )
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing base model files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to finetuned checkpoint (optional)')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'lora'],
                        help='Checkpoint mode: full or lora')
    
    # Prompt arguments
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt', type=str,
                              help='Text prompt for video generation')
    prompt_group.add_argument('--prompt_file', type=str,
                              help='File containing prompts (txt or json)')
    
    # Generation parameters
    parser.add_argument('--negative_prompt', type=str, default=None,
                        help='Negative prompt (optional)')
    parser.add_argument('--height', type=int, default=1088,
                        help='Video height (default: 1088 for 1080P)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Video width (default: 1920 for 1080P)')
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of frames (default: 81 for 1080P)')
    parser.add_argument('--steps', type=int, default=30,
                        help='Number of inference steps (default: 30)')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='CFG scale (default: 5.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory (default: outputs)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Output video FPS (default: 15)')
    parser.add_argument('--quality', type=int, default=8,
                        help='Video quality 1-10 (default: 8)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = load_prompts(args.prompt_file)
    
    logging.info(f"Total prompts to process: {len(prompts)}")
    
    # Initialize generator
    generator = VideoGenerator(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        device=args.device
    )
    
    # Generate videos
    for idx, prompt in enumerate(prompts):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {idx + 1}/{len(prompts)}")
        
        video = generator.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        
        # Save video
        output_path = output_dir / f"video_{idx:04d}.mp4"
        save_video(video, str(output_path), fps=args.fps, quality=args.quality)
        
        logging.info(f"✓ Saved: {output_path}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"✓ All videos generated successfully!")
    logging.info(f"Output directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
