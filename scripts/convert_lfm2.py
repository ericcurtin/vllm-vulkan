#!/usr/bin/env python3
"""Export the exact native packed weights used by explicit LFM2 residency."""

import argparse

from vllm_vulkan._rs import Lfm2Model


def main():
    """Parse export options and write a new checkpoint with canonical packed weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination", help="new directory; must not exist")
    parser.add_argument(
        "--quant",
        required=True,
        choices=[
            "auto",
            "f32",
            "f16",
            "bf16",
            "q4_0",
            "q4_k",
            "q8_0",
            "mlx2",
            "mlx4",
            "mlx6",
            "mlx8",
        ],
    )
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    Lfm2Model.convert_checkpoint(
        args.source, args.destination, args.quant, args.group_size
    )
    print(args.destination)


if __name__ == "__main__":
    main()
