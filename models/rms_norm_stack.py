WORKLOAD_NAME = "rms-norm-stack"


def get_workload():
    return [
        {
            "name": "encoder.block0.rms_norm",
            "kernel_type": "rms_norm",
            "kernel_path": "kernels/rms_norm.py",
            "input_kwargs": {"batch_size": 8, "seq_len": 128, "hidden_size": 512},
        },
        {
            "name": "encoder.block1.rms_norm",
            "kernel_type": "rms_norm",
            "kernel_path": "kernels/rms_norm.py",
            "input_kwargs": {"batch_size": 8, "seq_len": 128, "hidden_size": 512},
        },
        {
            "name": "decoder.block0.rms_norm",
            "kernel_type": "rms_norm",
            "kernel_path": "kernels/rms_norm.py",
            "input_kwargs": {"batch_size": 4, "seq_len": 256, "hidden_size": 1024},
        },
    ]
