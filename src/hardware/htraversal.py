from hardware.hnode import *

def fill_descendant_set(device: HwUnit):
    parent_node = device.parent
    while parent_node is not None:
        parent_node.descendant_set.add(device.name)
        parent_node = parent_node.parent


def find_least_common_ancestor(src_device: HwUnit, dst_device: HwUnit):
    # 寻找第一个共同祖先
    tmp_node = src_device.parent
    while tmp_node is not None and dst_device.name not in tmp_node.descendant_set:
        tmp_node = tmp_node.parent
    assert tmp_node is not None, "cannot find the communication path"
    return tmp_node