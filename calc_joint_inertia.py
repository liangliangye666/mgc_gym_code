#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 URDF 中每个关节的"等效转动惯量"（绕关节旋转轴）。

方法：Composite Rigid Body Algorithm (CRBA) 的惯性聚合思路。
- 每个连杆先把自己 + 下游子树的全部质量/惯量聚合到自身坐标系；
- 每个关节的等效转动惯量 = 子树总惯量张量投影到关节轴方向上的标量。
"""
import sys
import math
import xml.etree.ElementTree as ET

import numpy as np


def parse_float(s):
    return float(s.strip())


def rpy_to_matrix(r, p, y):
    """roll-pitch-yaw (XYZ) 到旋转矩阵"""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    links = {}
    joints = {}

    for link in root.findall('link'):
        name = link.get('name')
        info = {'mass': 0.0, 'com': np.zeros(3), 'inertia': np.zeros((3, 3))}
        inert = link.find('inertial')
        if inert is not None:
            m = inert.find('mass')
            if m is not None:
                info['mass'] = parse_float(m.get('value'))
            origin = inert.find('origin')
            if origin is not None:
                xyz = [parse_float(x) for x in origin.get('xyz').split()]
                rpy = [parse_float(x) for x in origin.get('rpy').split()]
                info['com'] = np.array(xyz)
                info['com_R'] = rpy_to_matrix(*rpy)
            else:
                info['com_R'] = np.eye(3)
            inertia = inert.find('inertia')
            if inertia is not None:
                ixx = parse_float(inertia.get('ixx'))
                iyy = parse_float(inertia.get('iyy'))
                izz = parse_float(inertia.get('izz'))
                ixy = parse_float(inertia.get('ixy', '0'))
                ixz = parse_float(inertia.get('ixz', '0'))
                iyz = parse_float(inertia.get('iyz', '0'))
                info['inertia'] = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ])
        links[name] = info

    for joint in root.findall('joint'):
        if joint.get('type') == 'fixed':
            continue
        name = joint.get('name')
        origin = joint.find('origin')
        xyz = [parse_float(x) for x in origin.get('xyz').split()]
        rpy = [parse_float(x) for x in origin.get('rpy').split()]
        axis = joint.find('axis')
        ax = [parse_float(x) for x in axis.get('xyz').split()]
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        joints[name] = {
            'parent': parent,
            'child': child,
            'xyz': np.array(xyz),
            'R': rpy_to_matrix(*rpy),
            'axis': np.array(ax),
        }

    return links, joints


def build_tree(links, joints):
    """建立父子关系，返回每个 link 的父 joint 名"""
    parent_joint = {}
    children = {name: [] for name in links}
    for jname, j in joints.items():
        parent_joint[j['child']] = jname
        children[j['parent']].append(j['child'])
    # 找到根（没有父 joint 的 link）
    roots = [name for name in links if name not in parent_joint]
    return parent_joint, children, roots


def project_inertia_to_axis(I, axis, R_axis_to_link):
    """把惯量张量 I（在 link 坐标系下）投影到关节轴方向上，得到绕该轴的标量惯量"""
    # 关节轴在 link 坐标系下的方向
    a = R_axis_to_link @ axis
    a = a / np.linalg.norm(a)
    return float(a @ I @ a)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/hll/mgc_code_git/gac-robotics/resources/robots/l5a/urdf/l5aurdf20260521.urdf"
    links, joints = parse_urdf(path)
    parent_joint, children, roots = build_tree(links, joints)

    # 聚合子树惯量：从叶子往根方向
    # aggregated[name] = (总质量, 总惯量(在 name 坐标系下), 质心)
    aggregated = {}

    def aggregate(name):
        link = links[name]
        m = link['mass']
        com = link['com']
        I_com = link['inertia']
        # 自身惯量平移到 link 原点（link 坐标系）
        # I_origin = I_com + m * (com^T com * I - com com^T)
        d = com
        d2 = float(d @ d)
        I_origin = I_com + m * (d2 * np.eye(3) - np.outer(d, d))

        total_m = m
        total_I = I_origin
        # 质心加权（先不严格求质心，只聚合质量与绕原点的惯量）
        for child in children[name]:
            child_m, child_I_origin, _ = aggregate(child)
            # 子 link 原点在父 link 坐标系下的位置 = joint xyz
            jname = parent_joint[child]
            j = joints[jname]
            p = j['xyz']
            R = j['R']
            # 子惯量（在子 link 原点）变换到父 link 坐标系
            R = R.T  # 注意 R 是 parent->child，反向用
            # 子 link 原点在父 link 系下的位置
            # 平行轴：I_parent = R * I_child * R^T + m_child * (p^T p I - p p^T)
            I_child_in_parent = R @ child_I_origin @ R.T
            p2 = float(p @ p)
            total_I = total_I + I_child_in_parent + child_m * (p2 * np.eye(3) - np.outer(p, p))
            total_m += child_m

        aggregated[name] = (total_m, total_I, com)
        return aggregated[name]

    for root in roots:
        aggregate(root)

    # 计算每个关节的等效转动惯量
    print("=" * 90)
    print(f"URDF: {path}")
    print("=" * 90)
    print(f"{'关节名':<28} {'轴方向':<12} {'等效转动惯量 (kg·m²)':<24}")
    print("-" * 90)

    results = []
    for jname, j in joints.items():
        child = j['child']
        child_m, child_I_origin, _ = aggregated[child]
        axis = j['axis']
        # 关节轴在 child link 坐标系下：关节 origin 的 R 是 parent->child，
        # 轴是定义在 joint 坐标系（= parent 坐标系）下的，需变换到 child 系
        R_parent_to_child = j['R'].T
        axis_in_child = R_parent_to_child @ axis
        axis_in_child = axis_in_child / np.linalg.norm(axis_in_child)
        I_eff = float(axis_in_child @ child_I_origin @ axis_in_child)
        results.append((jname, axis, I_eff, child_m))
        print(f"{jname:<28} {str(axis):<12} {I_eff:<24.6f}")

    print("-" * 90)
    print("\n说明：等效转动惯量 = 该关节下游子树(含自身)的总惯量张量，投影到关节旋转轴上的标量。")
    print("单位 kg·m²。值越大，该关节驱动器转起来越'费劲'（越难加速/减速）。")


if __name__ == '__main__':
    main()
