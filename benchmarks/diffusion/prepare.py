#!/usr/bin/env python
from dataclasses import dataclass
import os



@dataclass
class TrainingConfig:
    model: str = "stabilityai/stable-diffusion-2"
    dataset: str = "lambdalabs/naruto-blip-captions"
    revision: str = None
    variant: str = None
    cache: str = None


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig)
    args, _ = parser.parse_known_args()
    # --

    if args.cache:
        os.environ["XDG_CACHE_HOME"] = str(args.cache)

    # --
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from datasets import load_dataset

    _ = load_dataset(args.dataset)

    _ = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    _ = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", revision=args.revision, variant=args.variant
    )

    _ = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", revision=args.revision, variant=args.variant
    )

    _ = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", revision=args.revision
    )

    _ = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")


if __name__ == "__main__":
    main()
