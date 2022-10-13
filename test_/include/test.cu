#include <torch/extension.h>
#include <cuda_runtime.h>

a = torch::rand(5);
b = a.packed_accessor32<float, 1, torch::RestrictPtrTraits> ();
printf((b))