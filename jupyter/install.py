#!/usr/bin/env python3
"""
Install the CJC Jupyter kernel.

Usage:
    python install.py              # install for current user
    python install.py --sys-prefix # install into current virtualenv/conda env
    python install.py --prefix /p  # install into specific prefix
"""

import argparse
import json
import os
import shutil
import sys

from jupyter_client.kernelspec import KernelSpecManager


def install_kernel(args):
    # Build kernel spec directory in a temp location
    kernel_dir = os.path.join(os.path.dirname(__file__), "cjc_kernel")
    kernel_json_path = os.path.join(kernel_dir, "kernel.json")

    if not os.path.isfile(kernel_json_path):
        print(f"ERROR: kernel.json not found at {kernel_json_path}", file=sys.stderr)
        sys.exit(1)

    # Read and patch kernel.json to use the correct Python
    with open(kernel_json_path) as f:
        spec = json.load(f)

    spec["argv"][0] = sys.executable  # use the current Python interpreter

    # Write to a staging directory
    staging = os.path.join(os.path.dirname(__file__), "_staging_cjc")
    os.makedirs(staging, exist_ok=True)
    with open(os.path.join(staging, "kernel.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # Install via jupyter_client
    ksm = KernelSpecManager()

    kwargs = {"kernel_name": "cjc"}
    if args.sys_prefix:
        kwargs["prefix"] = sys.prefix
    elif args.prefix:
        kwargs["prefix"] = args.prefix
    else:
        kwargs["user"] = True

    dest = ksm.install_kernel_spec(staging, **kwargs)
    print(f"CJC kernel installed to: {dest}")
    print("Run `jupyter kernelspec list` to verify.")

    # Cleanup staging
    shutil.rmtree(staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Install the CJC Jupyter kernel")
    parser.add_argument(
        "--sys-prefix",
        action="store_true",
        help="Install into sys.prefix (virtualenv/conda)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Install into specific prefix directory",
    )
    args = parser.parse_args()
    install_kernel(args)


if __name__ == "__main__":
    main()
