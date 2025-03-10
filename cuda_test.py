import subprocess
import torch

def test_cuda():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Please check your driver and CUDA installation.")
        return False

    print(f"\n(CUDA available) GPU Name: {torch.cuda.get_device_name(0)}\n")

    try:
        output = subprocess.check_output(['nvcc', '--version'])
        print("CUDA Toolkit is installed.\n")
    except FileNotFoundError:
        print("WARNING: CUDA Toolkit is not installed.")
        return False

    # Matmul test
    tensor_size = 20
    tensor_a = torch.rand(tensor_size, tensor_size).cuda()
    tensor_b = torch.rand(tensor_size, tensor_size).cuda()


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    print(f"matmul test on {tensor_size}x{tensor_size} matrix...")
    result = torch.matmul(tensor_a, tensor_b)
    end_event.record()

    torch.cuda.synchronize()
    print(f"Time taken for matrix multiplication: {start_event.elapsed_time(end_event):.3f} ms")

    return True

if __name__ == "__main__":
    test_cuda()
