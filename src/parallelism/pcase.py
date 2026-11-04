from parallelism.pnode import *

def build_case_0(request_type_num,
                 total_layer_num):
    # ------------------ CASE 0 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 # 对于 DP：
                                 # parallel_attr 一共有 request_type_num 个 list
                                 # 每个 list 是该 req_type 的切分比例
                                 parallel_attr=[[0.5,0.5],
                                                [0.5,0.5],
                                                [0.5,0.5]])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.5, 0.5])
    node_2 = ModuleParallelismNode(name='p2', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = ModuleParallelismNode(name='p4', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]

def build_case_1(request_type_num,
                 total_layer_num):
    # ------------------ CASE 1 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = PipelineParallelismNode(name='p1', parallel_attr=[0.4, 0.6])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.5, 0.5])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]

def build_case_2(request_type_num,
                 total_layer_num):
    # ------------------ CASE 2 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = PipelineParallelismNode(name='p1', parallel_attr=[0.2, 0.8])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.2, 0.8])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.6, 0.4])
    # 对于 DP，一共有 request_type_num 个 list，每个 list 代表了一种 req 的划分比例
    node_4 = DataParallelismNode(name='p4', parallel_attr=[[0.5,0.5], [0.5, 0.5], [0.5,0.5]])
    node_5 = PipelineParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.8, 0.2])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]


def build_case_3(request_type_num,
                 total_layer_num):
    # ------------------ CASE 3 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    # node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.22355223288338252,0.2574972039927545,0.2522329860435903,0.2378661043326462])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_1.add_child(leaf_2)
    node_1.add_child(leaf_3)
    node_2.add_child(leaf_4)
    node_2.add_child(leaf_5)
    node_2.add_child(leaf_6)
    node_2.add_child(leaf_7)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7]

def build_case_4(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.5, 0.5],
                                                [0.5, 0.5],
                                                [0.5, 0.5]])
    # node_0 = PipelineParallelismNode(name='p0',
    #                                  dp_attr=[[0.0, 1.0]] * request_type_num,
    #                                  pp_attr=[0, total_layer_num - 1],
    #                                  tp_attr=[0.0, 1.0],
    #                                  xp_attr=XpTag.BOTH,
    #                                  parallel_attr=[0.5, 0.5])
    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = ModuleParallelismNode(name='p4', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    # node_4 = DataParallelismNode(name='p4', parallel_attr=[[0.5,0.5], [0.5,0.5], [0.5,0.5]])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_4)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_2, leaf_1, leaf_3, leaf_4, leaf_6, leaf_5, leaf_7]

def build_case_5(request_type_num,
                 total_layer_num):
    # ------------------ CASE 5 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.5, 0.5])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_2, leaf_1, leaf_3]


def build_case_6(request_type_num,
                 total_layer_num):
    # ------------------ CASE 6 --------------------------
    node_0 = TensorParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0, total_layer_num - 1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[0.25, 0.25, 0.25, 0.25])

    leaf_0 = BasicHardwareNode(idx=0, name='l0')
    leaf_1 = BasicHardwareNode(idx=1, name='l1')
    leaf_2 = BasicHardwareNode(idx=2, name='l2')
    leaf_3 = BasicHardwareNode(idx=3, name='l3')

    node_0.add_child(leaf_0)
    node_0.add_child(leaf_1)
    node_0.add_child(leaf_2)
    node_0.add_child(leaf_3)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3]


def build_case_7(request_type_num,
                 total_layer_num):
    # ------------------ CASE 7 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = PipelineParallelismNode(name='p1', parallel_attr=[0.7, 0.3])
    node_2 = PipelineParallelismNode(name='p2', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3]



def build_case_8(request_type_num,
                 total_layer_num):
    # ------------------ CASE 8 --------------------------
    node_0 = PipelineParallelismNode(name='p0',
                                     dp_attr=[[0.0, 1.0]] * request_type_num,
                                     pp_attr=[0, total_layer_num - 1],
                                     tp_attr=[0.0, 1.0],
                                     xp_attr=XpTag.BOTH,
                                     parallel_attr=[0.5, 0.5])
    node_1 = ModuleParallelismNode( parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = PipelineParallelismNode(name='p2', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3]


def build_case_9(request_type_num,
                 total_layer_num):
    # ------------------ CASE 9 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.5,0.5], [0.5,0.5], [0.5,0.5]])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.5, 0.5])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    return node_0, [leaf_0, leaf_2, leaf_1, leaf_3]


def build_case_10(request_type_num,
                 total_layer_num):
    # ------------------ CASE 6 --------------------------
    node_0 = TensorParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0, total_layer_num - 1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[0.25, 0.25])

    leaf_0 = BasicHardwareNode(idx=0, name='l0')
    leaf_1 = BasicHardwareNode(idx=1, name='l1')

    node_0.add_child(leaf_0)
    node_0.add_child(leaf_1)

    return node_0, [leaf_0, leaf_1]


def build_case_11(request_type_num,
                 total_layer_num):
    # ------------------ CASE 6 --------------------------
    node_0 = PipelineParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0, total_layer_num - 1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[0.25, 0.25, 0.25, 0.25])

    leaf_0 = BasicHardwareNode(idx=0, name='l0')
    leaf_1 = BasicHardwareNode(idx=1, name='l1')
    leaf_2 = BasicHardwareNode(idx=2, name='l2')
    leaf_3 = BasicHardwareNode(idx=3, name='l3')

    node_0.add_child(leaf_0)
    node_0.add_child(leaf_1)
    node_0.add_child(leaf_2)
    node_0.add_child(leaf_3)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3]


def build_case_12(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 # parallel_attr=[[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 1.0]])
                                 parallel_attr=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
                                 # parallel_attr=[[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.5, 0.5])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_4)
    node_0.add_child(node_5)
    node_1.add_child(node_2)
    node_1.add_child(node_3)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_4.add_child(leaf_4)
    node_4.add_child(leaf_5)
    node_5.add_child(leaf_6)
    node_5.add_child(leaf_7)

    return node_0, [leaf_0, leaf_2, leaf_1, leaf_3, leaf_4, leaf_6, leaf_5, leaf_7]
    # return node_0, [leaf_0, leaf_1, leaf_4, leaf_5, leaf_2, leaf_3, leaf_6, leaf_7]


def build_case_13(request_type_num,
                 total_layer_num):
    # ------------------ CASE 6 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0, total_layer_num - 1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])

    leaf_0 = BasicHardwareNode(idx=0, name='l0')
    leaf_1 = BasicHardwareNode(idx=1, name='l1')

    node_0.add_child(leaf_0)
    node_0.add_child(leaf_1)

    return node_0, [leaf_0, leaf_1]


def build_case_14(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.5,0.5], [0.5,0.5], [0.5,0.5]])
    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    # return node_0, [leaf_0, leaf_2, leaf_1, leaf_3]
    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3]


def build_case_15(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.37518363073869065,2.1249432908284036],[0.428767370950575,1.1761577836166395],[0.3025002629648586,1.870890017253797]])
    node_1 = DataParallelismNode(name='p1', parallel_attr=[[2.5569088334965993, 1.2419193777389295],[1.2396602904424328, 1.1575439514392112],[0.5920507392405951, 1.5695420571338758]])
    node_3 = ModuleParallelismNode(name='p3', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_7 = ModuleParallelismNode(name='p7', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_8 = TensorParallelismNode(name='p8', parallel_attr=[1.095859672999804])
    node_9 = ModuleParallelismNode(name='p9', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_10 = TensorParallelismNode(name='p10', parallel_attr=[0.1950169052957875, 11.62741587100428])
    node_11 = TensorParallelismNode(name='p11', parallel_attr=[1.2202647184623128])
    node_12 = TensorParallelismNode(name='p12', parallel_attr=[0.2275909069665696])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.9509423254287418])
    node_2 = DataParallelismNode(name='p2', parallel_attr=[[2.426829178604129, 1.2144506643070796], [0.15073781380699044, 0.2722029191517329], [1.0601270190141008, 2.3881737917012784]])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[1.158283805772237])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[1.410808250478341])

    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_1.add_child(node_3)
    node_1.add_child(node_4)
    node_3.add_child(node_7)
    node_3.add_child(node_12)
    node_7.add_child(node_8)
    node_7.add_child(node_9)
    node_9.add_child(node_10)
    node_9.add_child(node_11)
    node_2.add_child(node_5)
    node_2.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_8.add_child(leaf_6)
    node_10.add_child(leaf_2)
    node_10.add_child(leaf_5)
    node_11.add_child(leaf_7)
    node_12.add_child(leaf_0)
    node_4.add_child(leaf_4)
    node_5.add_child(leaf_1)
    node_6.add_child(leaf_3)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7]

def build_case_16(request_type_num,
                 total_layer_num):
    # ------------------ CASE 3 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_1.add_child(leaf_2)
    node_1.add_child(leaf_3)
    node_1.add_child(leaf_4)
    node_1.add_child(leaf_5)
    node_1.add_child(leaf_6)
    node_1.add_child(leaf_7)
    node_2.add_child(leaf_8)
    node_2.add_child(leaf_9)
    node_2.add_child(leaf_10)
    node_2.add_child(leaf_11)
    node_2.add_child(leaf_12)
    node_2.add_child(leaf_13)
    node_2.add_child(leaf_14)
    node_2.add_child(leaf_15)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_8, leaf_1, leaf_9, leaf_2, leaf_10, leaf_3, leaf_11,
                    leaf_4, leaf_12, leaf_5, leaf_13, leaf_6, leaf_14, leaf_7, leaf_15]

def build_case_17(request_type_num,
                 total_layer_num):
    # ------------------ CASE 3 --------------------------
    node_0 = PipelineParallelismNode(name='p0',
                                     dp_attr=[[0.0, 1.0]] * request_type_num,
                                     pp_attr=[0, total_layer_num - 1],
                                     tp_attr=[0.0, 1.0],
                                     xp_attr=XpTag.BOTH,
                                     parallel_attr = [0.5, 0.5])
    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = ModuleParallelismNode(name='p2', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_1.add_child(node_3)
    node_1.add_child(node_4)
    node_2.add_child(node_5)
    node_2.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    node_3.add_child(leaf_0)
    node_3.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_4.add_child(leaf_4)
    node_4.add_child(leaf_5)
    node_4.add_child(leaf_6)
    node_4.add_child(leaf_7)
    node_5.add_child(leaf_8)
    node_5.add_child(leaf_9)
    node_5.add_child(leaf_10)
    node_5.add_child(leaf_11)
    node_6.add_child(leaf_12)
    node_6.add_child(leaf_13)
    node_6.add_child(leaf_14)
    node_6.add_child(leaf_15)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7,
                    leaf_8, leaf_12, leaf_9, leaf_13, leaf_10, leaf_14, leaf_11, leaf_15]

def build_case_18(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[1.0, 0.3, 0.1], [1.0, 0.4, 0.1], [0.5, 0.7, 0.4]]) # Arch 9
                                 # parallel_attr=[[1.0, 0.8, 0.3], [0.8, 0.9, 0.3], [0.8, 0.5, 0.8]])   # Arch 8

    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_4)
    node_0.add_child(node_5)
    node_1.add_child(node_2)
    node_1.add_child(node_3)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    # 4 NPU
    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    # 4 PIM
    node_3.add_child(leaf_4)
    node_3.add_child(leaf_5)
    node_3.add_child(leaf_6)
    node_3.add_child(leaf_7)

    # 4 NPU
    node_4.add_child(leaf_8)
    node_4.add_child(leaf_9)
    node_4.add_child(leaf_10)
    node_4.add_child(leaf_11)

    # 4 PIM
    node_5.add_child(leaf_12)
    node_5.add_child(leaf_13)
    node_5.add_child(leaf_14)
    node_5.add_child(leaf_15)

    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7,
                    leaf_8, leaf_12, leaf_9, leaf_13, leaf_10, leaf_14, leaf_11, leaf_15]


def build_case_19(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 # parallel_attr=[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.8, 0.8], [0.3, 0.3, 1.0, 1.0]]) # Arch8
                                 parallel_attr=[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.2, 0.2], [1, 1.2, 1.0, 1.0]]) # Arch9

    # node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.5, 0.25, 0.25, 0.25])
    # node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.25, 0.7, 0.25, 0.25])
    # node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.25, 0.25, 0.5, 0.25])
    # node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.6, 0.25, 0.2, 0.25])

    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_0.add_child(node_3)
    node_0.add_child(node_4)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    # 4 NPU
    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_1.add_child(leaf_2)
    node_1.add_child(leaf_3)

    # 4 NPU
    node_2.add_child(leaf_4)
    node_2.add_child(leaf_5)
    node_2.add_child(leaf_6)
    node_2.add_child(leaf_7)

    # 4 PIM
    node_3.add_child(leaf_8)
    node_3.add_child(leaf_9)
    node_3.add_child(leaf_10)
    node_3.add_child(leaf_11)

    # 4 PIM
    node_4.add_child(leaf_12)
    node_4.add_child(leaf_13)
    node_4.add_child(leaf_14)
    node_4.add_child(leaf_15)

    return node_0, [leaf_0, leaf_8, leaf_1, leaf_9, leaf_2, leaf_10, leaf_3, leaf_11,
                    leaf_4, leaf_12, leaf_5, leaf_13, leaf_6, leaf_14, leaf_7, leaf_15]

def build_case_20(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    node_1 = ModuleParallelismNode(name="p1", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = ModuleParallelismNode(name="p2", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])

    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_7 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    node_1.add_child(node_4)
    node_1.add_child(node_5)
    node_2.add_child(node_6)
    node_2.add_child(node_7)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    # 4 NPU
    node_4.add_child(leaf_0)
    node_4.add_child(leaf_1)
    node_4.add_child(leaf_2)
    node_4.add_child(leaf_3)

    # 4 PIM
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_5.add_child(leaf_6)
    node_5.add_child(leaf_7)

    # 4 NPU
    node_6.add_child(leaf_8)
    node_6.add_child(leaf_9)
    node_6.add_child(leaf_10)
    node_6.add_child(leaf_11)

    # 4 PIM
    node_7.add_child(leaf_12)
    node_7.add_child(leaf_13)
    node_7.add_child(leaf_14)
    node_7.add_child(leaf_15)

    return node_0, [leaf_0,  leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7,
                    leaf_8, leaf_12, leaf_9, leaf_13, leaf_10, leaf_14, leaf_11, leaf_15]


def build_case_21(request_type_num,
                 total_layer_num):
    # ------------------ CASE 3 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_1.add_child(leaf_2)
    node_1.add_child(leaf_3)
    node_2.add_child(leaf_4)
    node_2.add_child(leaf_5)

    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_3]


# def build_case_22(request_type_num,
#                  total_layer_num):
#     # ------------------ CASE 4 --------------------------
#     node_0 = DataParallelismNode(name='p0',
#                                  dp_attr=[[0.0, 1.0]] * request_type_num,
#                                  pp_attr=[0, total_layer_num - 1],
#                                  tp_attr=[0.0, 1.0],
#                                  xp_attr=XpTag.BOTH,
#                                  parallel_attr=[[1.0, 0.2], [1.0, 0.6], [1.0, 0.6]])
#     node_1 = PipelineParallelismNode(name='p1',  parallel_attr=[0.5, 0.5])
#     # node_1 = DataParallelismNode(name='p1', parallel_attr=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
#     node_2 = TensorParallelismNode(name='p2',  parallel_attr=[0.25, 0.25, 0.25, 0.25])
#
#     node_3 = ModuleParallelismNode(name="p3", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
#     node_4 = ModuleParallelismNode(name="p4", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
#
#     node_5 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
#     node_6 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
#     node_7 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])
#     node_8 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])
#
#     node_0.add_child(node_1)
#     node_0.add_child(node_2)
#     node_1.add_child(node_3)
#     node_1.add_child(node_4)
#     node_3.add_child(node_5)
#     node_3.add_child(node_6)
#     node_4.add_child(node_7)
#     node_4.add_child(node_8)
#
#     leaf_0 = BasicHardwareNode(idx = 0, name='l0')
#     leaf_1 = BasicHardwareNode(idx = 1, name='l1')
#     leaf_2 = BasicHardwareNode(idx = 2, name='l2')
#     leaf_3 = BasicHardwareNode(idx = 3, name='l3')
#     leaf_4 = BasicHardwareNode(idx = 4, name='l4')
#     leaf_5 = BasicHardwareNode(idx = 5, name='l5')
#     leaf_6 = BasicHardwareNode(idx = 6, name='l6')
#     leaf_7 = BasicHardwareNode(idx = 7, name='l7')
#     leaf_8 = BasicHardwareNode(idx = 8, name='l8')
#     leaf_9 = BasicHardwareNode(idx = 9, name='l9')
#     leaf_10 = BasicHardwareNode(idx = 10, name='l10')
#     leaf_11 = BasicHardwareNode(idx = 11, name='l11')
#     leaf_12 = BasicHardwareNode(idx = 12, name='l12')
#     leaf_13 = BasicHardwareNode(idx = 13, name='l13')
#     leaf_14 = BasicHardwareNode(idx = 14, name='l14')
#     leaf_15 = BasicHardwareNode(idx = 15, name='l15')
#
#     # 4 NPU
#     node_5.add_child(leaf_0)
#     node_5.add_child(leaf_1)
#     node_5.add_child(leaf_2)
#     node_5.add_child(leaf_3)
#
#     # 2 PIM
#     node_6.add_child(leaf_4)
#     node_6.add_child(leaf_5)
#
#     # 4 NPU
#     node_7.add_child(leaf_6)
#     node_7.add_child(leaf_7)
#     node_7.add_child(leaf_8)
#     node_7.add_child(leaf_9)
#
#     # 2 PIM
#     node_8.add_child(leaf_10)
#     node_8.add_child(leaf_11)
#
#     # 4 PIM
#     node_2.add_child(leaf_12)
#     node_2.add_child(leaf_13)
#     node_2.add_child(leaf_14)
#     node_2.add_child(leaf_15)
#
#     return node_0, [leaf_0,  leaf_4, leaf_1, leaf_5, leaf_2, leaf_10, leaf_3, leaf_11,
#                     leaf_6, leaf_12, leaf_7, leaf_13, leaf_8, leaf_14, leaf_9, leaf_15]

def build_case_22(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[1.0, 0.7], [1.0, 1.0], [0.5, 0.8]])
    node_1 = PipelineParallelismNode(name='p1',  parallel_attr=[0.5, 0.5])
    # node_1 = DataParallelismNode(name='p1', parallel_attr=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    node_2 = TensorParallelismNode(name='p2',  parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_3 = ModuleParallelismNode(name="p3", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_4 = ModuleParallelismNode(name="p4", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])

    node_5 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_6 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_7 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_8 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_1.add_child(node_3)
    node_1.add_child(node_4)
    node_3.add_child(node_6)
    node_3.add_child(node_5)
    node_4.add_child(node_8)
    node_4.add_child(node_7)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    # 4 NPU
    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_2.add_child(leaf_2)
    node_2.add_child(leaf_3)

    # 4 PIM
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_5.add_child(leaf_6)
    node_5.add_child(leaf_7)

    # 2 NPU
    node_6.add_child(leaf_8)
    node_6.add_child(leaf_9)

    # 4 PIM
    node_7.add_child(leaf_10)
    node_7.add_child(leaf_11)
    node_7.add_child(leaf_12)
    node_7.add_child(leaf_13)

    # 2 NPU
    node_8.add_child(leaf_14)
    node_8.add_child(leaf_15)

    return node_0, [leaf_0,  leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7,
                    leaf_8, leaf_10, leaf_9, leaf_11, leaf_14, leaf_12, leaf_15, leaf_13]

def build_case_23(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.8, 0.4, 0.2], [0.9, 0.4, 0.3], [0.8, 0.9, 0.3]])

    node_1 = ModuleParallelismNode(name="p1", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = ModuleParallelismNode(name="p2", parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_3 = TensorParallelismNode(name='p3',  parallel_attr=[0.5, 0.5])

    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_7 = TensorParallelismNode(name='p7', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_0.add_child(node_3)

    node_1.add_child(node_4)
    node_1.add_child(node_5)
    node_2.add_child(node_6)
    node_2.add_child(node_7)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')
    leaf_8 = BasicHardwareNode(idx = 8, name='l8')
    leaf_9 = BasicHardwareNode(idx = 9, name='l9')
    leaf_10 = BasicHardwareNode(idx = 10, name='l10')
    leaf_11 = BasicHardwareNode(idx = 11, name='l11')
    leaf_12 = BasicHardwareNode(idx = 12, name='l12')
    leaf_13 = BasicHardwareNode(idx = 13, name='l13')
    leaf_14 = BasicHardwareNode(idx = 14, name='l14')
    leaf_15 = BasicHardwareNode(idx = 15, name='l15')

    # 2 PIM
    node_3.add_child(leaf_0)
    node_3.add_child(leaf_1)

    # 4 NPU
    node_4.add_child(leaf_2)
    node_4.add_child(leaf_3)
    node_4.add_child(leaf_4)
    node_4.add_child(leaf_5)

    # 4 PIM
    node_5.add_child(leaf_6)
    node_5.add_child(leaf_7)
    node_5.add_child(leaf_8)
    node_5.add_child(leaf_9)

    # 4 NPU
    node_6.add_child(leaf_10)
    node_6.add_child(leaf_11)
    node_6.add_child(leaf_12)
    node_6.add_child(leaf_13)

    # 2 PIM
    node_7.add_child(leaf_14)
    node_7.add_child(leaf_15)

    return node_0, [leaf_2,  leaf_0, leaf_3, leaf_1, leaf_4, leaf_6, leaf_5, leaf_7,
                    leaf_10, leaf_8, leaf_11, leaf_9, leaf_12, leaf_14, leaf_13, leaf_15]
