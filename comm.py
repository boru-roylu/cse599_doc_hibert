import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return -1
    if not dist.is_initialized():
        return -1
    return dist.get_rank()


def get_device():
    if not dist.is_available() or not dist.is_initialized():
        return torch.device('cuda', 0)
    else:
        return torch.device('cuda', get_rank())


def is_local_master():
    return get_rank() in [-1, 0]


def is_distributed():
    return get_rank() != -1


def wait_master():
    if is_local_master():
        dist.barrier()


def wait_others():
    if not is_local_master():
        dist.barrier()


def wait_all():
    dist.barrier()


def obj_to_device(obj, device):
    # only put tensor into GPU
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, (list, tuple)):
        obj = list(obj)
        for v_i, v in enumerate(obj):
            obj[v_i] = obj_to_device(v, device)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = obj_to_device(v, device)
    return obj


def all_gather(data, to_cpu=False):
    world_size = get_world_size()
    if world_size == 1:
        data = torch.tensor(data)
        if to_cpu:
            data = data.cpu()
        return [data]

    device = get_device()

    if not torch.is_tensor(data):
        data = torch.Tensor(data)
    data = data.to(device)

    rest_size = data.size()[1:]

    local_size = torch.LongTensor([data.size(0)]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    
    # +1 for a weird thing happen when local size == max(size_list)
    max_size = max(size_list) + 1

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.zeros(size=(max_size,)+rest_size).to(device))

    padding = torch.zeros(size=(max_size-local_size,)+rest_size).to(device)
    tensor = torch.cat((data, padding), dim=0)

    # we already + 1, the equality never happens
    #if local_size != max_size:
    #    padding = torch.zeros(size=(max_size-local_size,)+rest_size).to(device)
    #    tensor = torch.cat((data, padding), dim=0)
    #else:
    #    tensor = data

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        data_list.append(tensor[:size])

    if to_cpu:
        data_list = obj_to_device(data_list, 'cpu')

    return data_list