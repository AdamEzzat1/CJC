"""Entry point: python -m cjc_kernel"""
from .kernel import CJCKernel
from ipykernel.kernelapp import IPKernelApp

IPKernelApp.launch_instance(kernel_class=CJCKernel)
