"""检查 PyTorch 与 CUDA 运行环境。"""

import torch


def main() -> None:
    """输出当前 PyTorch 和 CUDA 环境信息。"""

    print("=" * 80)
    print("PyTorch / CUDA Environment")
    print("=" * 80)

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    cuda_available = torch.cuda.is_available()

    print("CUDA available:", cuda_available)

    if not cuda_available:
        print()
        print("❌ 当前 PyTorch 无法使用 CUDA")
        return

    print(
        "CUDA device count:",
        torch.cuda.device_count(),
    )

    print(
        "Current device index:",
        torch.cuda.current_device(),
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    properties = torch.cuda.get_device_properties(0)

    total_memory_gb = (
        properties.total_memory
        / 1024**3
    )

    print(
        "GPU memory:",
        f"{total_memory_gb:.2f} GB",
    )

    print()
    print("✅ PyTorch CUDA 环境正常")

x = torch.tensor(
    [1.0, 2.0, 3.0],
    device="cuda",
)

print("Tensor device:", x.device)
print("Tensor:", x)

if __name__ == "__main__":
    main()