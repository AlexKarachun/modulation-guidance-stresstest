import argparse
import os

from generate import generate

# Parse arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--data_config",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default='logs',
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.0,
        required=False,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_clip",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--apply_clip_pooled",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output_dir',
        required=False,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        required=False,
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        required=False,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False,
    )
    parser.add_argument(
        "--w",
        type=float,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--shift_type",
        type=str,
        default="realism",
    )
    parser.add_argument(
        "--do_init_run",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
# ----------------------------------------------------------------------------------------------------------------------

# Input spot
if __name__ == "__main__":
    args = parse_args()

    generate(args)
