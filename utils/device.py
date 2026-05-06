import torch


def get_device_info() -> dict:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        vram_total_gb = total / (1024 ** 3)
        vram_free_gb = free / (1024 ** 3)
        vram_used_gb = vram_total_gb - vram_free_gb
        cuda_version = torch.version.cuda
    else:
        device_name = "CPU"
        vram_total_gb = 0.0
        vram_free_gb = 0.0
        vram_used_gb = 0.0
        cuda_version = None

    return {
        "device": "cuda" if cuda_available else "cpu",
        "device_name": device_name,
        "vram_total_gb": round(vram_total_gb, 2),
        "vram_free_gb": round(vram_free_gb, 2),
        "vram_used_gb": round(vram_used_gb, 2),
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
    }


def get_recommended_settings(vram_gb: float) -> dict:
    if vram_gb <= 0:
        return {
            "max_params_fp16": 500_000_000,
            "max_params_4bit": 3_000_000_000,
            "recommended_quant": "4bit",
            "can_train_sae": False,
            "warnings": ["No GPU detected. Running on CPU will be very slow for large models."],
        }

    warnings = []
    if vram_gb < 8:
        warnings.append(f"Low VRAM ({vram_gb:.1f} GB). Use 4-bit quantization for models > 1B.")

    bytes_per_param_fp16 = 2
    bytes_per_param_4bit = 0.5
    overhead_factor = 1.3

    max_fp16 = int(vram_gb * (1024 ** 3) / (bytes_per_param_fp16 * overhead_factor))
    max_4bit = int(vram_gb * (1024 ** 3) / (bytes_per_param_4bit * overhead_factor))

    if vram_gb >= 20:
        rec_quant = "none"
    elif vram_gb >= 10:
        rec_quant = "8bit"
    else:
        rec_quant = "4bit"

    can_train = vram_gb >= 8

    return {
        "max_params_fp16": max_fp16,
        "max_params_4bit": max_4bit,
        "recommended_quant": rec_quant,
        "can_train_sae": can_train,
        "warnings": warnings,
    }


def estimate_model_vram(num_params: int, quant: str) -> float:
    if quant == "4bit":
        bytes_per_param = 0.5
    elif quant == "8bit":
        bytes_per_param = 1.0
    else:
        bytes_per_param = 2.0
    overhead = 1.3
    return (num_params * bytes_per_param * overhead) / (1024 ** 3)
