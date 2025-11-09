# import pycuda
import pycuda.driver as drv

drv.init()

print("CUDA device query (PyCUDA version) \n")

print(f"Detected {drv.Device.count()} CUDA Capable device(s) \n")

for i in range(drv.Device.count()):
    gpu_device = drv.Device(i)
    print(f"Device {i}: {gpu_device.name()}")
    compute_capability = float("%d.%d" % gpu_device.compute_capability())
    print(f"\t Compute Capability: {compute_capability}")
    print(f"\t Total Memory: {gpu_device.total_memory() // (1024**2)} megabytes")

    # The following will give us all remaining device attributes as seen
    # in the original deviceQuery.
    # We set up a dictionary as such so that we can easily index
    # the values using a string descriptor.

    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}

    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v

    num_mp = device_attributes["MULTIPROCESSOR_COUNT"]

    # Cores per multiprocessor is not reported by the GPU!
    # We must use a lookup table based on compute capability.
    # See the following:
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

    cores_per_mp_lookup = {
        5.0: 128,
        5.1: 128,
        5.2: 128,
        6.0: 64,
        6.1: 128,
        6.2: 128,
    }
    cuda_cores_per_mp = cores_per_mp_lookup.get(compute_capability)
    if not cuda_cores_per_mp:
        print(
            "\t Warning: cores-per-multiprocessor lookup missing for this compute capability."
        )
        cuda_cores_per_mp = 0

    print(
        f"\t ({num_mp}) Multiprocessors, ({cuda_cores_per_mp}) CUDA Cores / Multiprocessor: {num_mp * cuda_cores_per_mp} CUDA Cores"
    )

    device_attributes.pop("MULTIPROCESSOR_COUNT")

    for k in device_attributes.keys():
        print(f"\t {k}: {device_attributes[k]}")
