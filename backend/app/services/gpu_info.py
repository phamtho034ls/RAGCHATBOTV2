"""
Kiểm tra trạng thái GPU / CUDA.
"""

from app.models.schemas import GpuStatus


def get_gpu_status() -> GpuStatus:
    try:
        import torch

        if not torch.cuda.is_available():
            return GpuStatus(available=False)

        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        cuda_version = torch.version.cuda

        return GpuStatus(
            available=True,
            device_name=device_name,
            vram_total_mb=round(vram_total, 1),
            vram_used_mb=round(vram_used, 1),
            cuda_version=cuda_version,
        )
    except ImportError:
        return GpuStatus(available=False)
    except Exception:
        return GpuStatus(available=False)
