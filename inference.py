#!/usr/bin/env python3
"""
UltraGen Video Generation Inference Script
Simple and clean inference script for generating videos from text prompts.
"""

import os
import argparse
import json
import torch
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from diffsynth import ModelManager, WanVideoPipeline, save_video


class VideoGenerator:
    """Video generation pipeline wrapper"""
    
    def __init__(self, checkpoint_path, mode='full', device='cuda'):
        """
        Initialize the video generator
        
        Args:
            checkpoint_path: Path to model checkpoint
            mode: 'full' or 'lora'
            device: Device to run on ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.device = device
        
        logging.info("Loading models...")
        model_manager = ModelManager(device="cpu")
        
        # Try to load from dockerdata first, fallback to models directory
        model_paths = [
            "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ]
        
        model_manager.load_models(
            model_paths,
            torch_dtype=torch.bfloat16,
        )
        
        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, 
            torch_dtype=torch.bfloat16, 
            device=device
        )
        
        # Load checkpoint
        if checkpoint_path:
            if mode == 'lora':
                logging.info(f"Loading LoRA from: {checkpoint_path}")
                model_manager.load_lora(checkpoint_path, lora_alpha=0.5)
            else:
                logging.info(f"Loading full model from: {checkpoint_path}")
                self.pipe.dit.load_state_dict(torch.load(checkpoint_path))
        
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        logging.info("Model loaded successfully!")
    
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
        fps=15,
        quality=8
    ):
        """
        Generate video from prompt
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt (optional)
            height: Video height
            width: Video width
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed
            fps: Frames per second for output video
            quality: Video quality (1-10, higher is better)
        
        Returns:
            Generated video tensor
        """
        if negative_prompt is None:
            negative_prompt = (
                "Aerial view, overexposed, low quality, deformation, "
                "bad composition, bad hands, bad teeth, bad eyes, "
                "bad limbs, distortion, blurring, text, subtitles"
            )
        
        logging.info(f"Generating video for prompt: {prompt[:60]}...")
        
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            seed=seed,
            tiled=True,
        )
        
        return video


def load_prompts_from_file(file_path):
    """Load prompts from a text or JSON file"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                # Try to extract 'prompt' or 'prompt_en' field
                prompts = []
                for item in data:
                    if isinstance(item, dict):
                        prompts.append(item.get('prompt_en') or item.get('prompt') or str(item))
                    else:
                        prompts.append(str(item))
                return prompts
            return [data.get('prompt', str(data))]
    else:
        # Treat as text file, one prompt per line
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='UltraGen Video Generation - Generate videos from text prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from a single prompt
  python inference.py --checkpoint checkpoints/model.ckpt --prompt "A dog running in the park"
  
  # Generate from multiple prompts in a file
  python inference.py --checkpoint checkpoints/model.ckpt --prompt_file prompts.txt
  
  # Generate 4K video
  python inference.py --checkpoint checkpoints/4k.ckpt --prompt "..." --width 3840 --height 2160
        """
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'lora'],
                        help='Checkpoint mode: full or lora (default: full)')
    
    # Prompt arguments (one of these is required)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt', type=str,
                              help='Single prompt for video generation')
    prompt_group.add_argument('--prompt_file', type=str,
                              help='File containing prompts (txt or json)')
    
    # Generation parameters
    parser.add_argument('--negative_prompt', type=str, default=None,
                        help='Negative prompt (optional)')
    parser.add_argument('--height', type=int, default=1088,
                        help='Video height (default: 1088)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Video width (default: 1920)')
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of frames (default: 81)')
    parser.add_argument('--steps', type=int, default=30,
                        help='Number of inference steps (default: 30)')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='Classifier-free guidance scale (default: 5.0)')
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = load_prompts_from_file(args.prompt_file)
    
    logging.info(f"Loaded {len(prompts)} prompt(s)")
    
    # Initialize generator
    generator = VideoGenerator(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        device=args.device
    )
    
    # Generate videos
    for idx, prompt in enumerate(prompts):
        logging.info(f"Processing {idx + 1}/{len(prompts)}: {prompt[:60]}...")
        
        video = generator.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            fps=args.fps,
            quality=args.quality
        )
        
        # Save video
        output_path = os.path.join(args.output_dir, f"video_{idx:04d}.mp4")
        save_video(video, output_path, fps=args.fps, quality=args.quality)
        
        logging.info(f"Saved: {output_path}")
    
    logging.info("All videos generated successfully!")


if __name__ == '__main__':
    main()
