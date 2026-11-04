from hardware.htraversal import *
from hardware.hcase import nvl_bw, nvl_lat

def peer_to_peer_communication_time_cost(src_device: HwUnit, dst_device: HwUnit, comm_byte_size: int):
    if src_device == dst_device:
        return 0.0
    common_ancestor = find_least_common_ancestor(src_device, dst_device)
    # print("{} and {}'s least common ancestor is {}".format(src_device.name, dst_device.name, common_ancestor.path()))
    tmp_src = src_device
    tmp_dst = dst_device
    comm_time_cost_ms = 0.0
    while tmp_src is not None and tmp_dst is not None:
        src_comm_time_cost_ms = tmp_src.parent.peer_to_peer_communication(comm_byte_size)
        dst_comm_time_cost_ms = tmp_dst.parent.peer_to_peer_communication(comm_byte_size)
        comm_time_cost_ms += max(dst_comm_time_cost_ms, src_comm_time_cost_ms)

        tmp_src = tmp_src.parent
        tmp_dst = tmp_dst.parent

        if tmp_src == tmp_dst:
            assert tmp_src == common_ancestor
            break
    return comm_time_cost_ms # 单位：ms

def all_reduce_communication_time_cost(device_list: List[HwUnit], comm_byte_size: int):
    N = len(device_list)
    S = comm_byte_size
    # A100 NVLink
    # B = 250 * pow(10,9)   # BW = 250 GB/s
    # L = 10000 / pow(10, 9) # LAT = 5000 ns

    # H100 NVLink
    B = nvl_bw * pow(10,9)
    L = nvl_lat / pow(10,9)

    # B = 300 * pow(10,9)   # BW = 300 GB/s
    # L = 10000 / pow(10, 9) # LAT = 10000 ns

    # B = 350 * pow(10,9)   # BW = 300 GB/s
    # L = 3000 / pow(10, 9) # LAT = 10000 ns

    return 2*(N-1)*(S/(N*B)+L)*pow(10,3) # 单位：ms