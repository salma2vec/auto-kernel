"""Template workload definition for auto-kernel.

Fill in either:

1. static baseline times for portable planning, or
2. input kwargs so profile.py can benchmark the kernels directly.
"""

WORKLOAD_NAME = "custom-workload"


def get_workload():
    return [
        {
            "name": "stage0.example_kernel",
            "kernel_type": "rms_norm",
            "kernel_path": "kernels/rms_norm.py",
            "input_kwargs": {"batch_size": 8, "seq_len": 128, "hidden_size": 512},
        },
        # For a static profile that does not require live benchmarking:
        # {
        #     "name": "stage1.estimated_kernel",
        #     "kernel_type": "rms_norm",
        #     "kernel_path": "kernels/rms_norm.py",
        #     "baseline_time_ms": 3.5,
        # },
    ]
