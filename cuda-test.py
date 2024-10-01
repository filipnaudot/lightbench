import torch

def test_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your driver and CUDA installation.")
        return False
    

    print(f"\n(CUDA available) GPU Name: {torch.cuda.get_device_name(0)}\n\n")
    
    # Create two random tensors and move them to GPU
    tensor_a = torch.rand(3, 3).cuda()
    tensor_b = torch.rand(3, 3).cuda()

    # Perform a simple matrix multiplication on the GPU
    result = torch.matmul(tensor_a, tensor_b)

    print("Matrix A:")
    print(tensor_a)
    print("Matrix B:")
    print(tensor_b)
    print("\nResult of Matrix Multiplication A * B on GPU:")
    print(result)
    
    return True



if __name__ == "__main__":
    test_cuda()
