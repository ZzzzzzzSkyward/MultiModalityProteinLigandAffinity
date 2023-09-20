import os
BLOCKCUDA = 0


def get_gpu_memory():
    """
    Get the current GPU memory usage.

    Returns:
        A list of dictionaries. Each dictionary contains the following keys:
        'id': the ID of the GPU as integer.
        'memory_free': the amount of free memory in MB as integer.
    """
    # command = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader"
    command = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"
    memory_info = os.popen(command).readlines()
    memory_info = [x.strip() for x in memory_info]
    gpu_memory = []
    for line in memory_info:
        memory_items = line.split(',')
        gpu_id = int(memory_items[0])
        memory_free = int(memory_items[1].strip().split()[0])
        gpu_memory.append({'id': gpu_id, 'memory_free': memory_free})
    return gpu_memory


def get_best_gpu():
    """
    Get the GPU with the maximum free memory.

    Returns:
        A dictionary containing the following keys:
        'id': the ID of the GPU as integer.
        'memory_free': the amount of free memory in MB as integer.
    """
    gpu_memory = get_gpu_memory()
    sorted_gpu = sorted(gpu_memory, key=lambda x: (
        x['memory_free'], x['id']), reverse=True)
    for i in sorted_gpu:
        if i['id']>=BLOCKCUDA:
            return i['id']-BLOCKCUDA

if __name__ == '__main__':
    print(get_gpu_memory())
    print(get_best_gpu())
