"""自然语言任务需求到卫星类型级流程图生成系统 (Top-Level Entry Program)。

用法示例:
    1. 命令行指定任务描述文本:
       python main.py --text "任务中心接收地震灾区应急成像任务后，卫星进入目标区域过境窗口..."

    2. 从文本文件读取并导出下游调度 JSON 文件:
       python main.py --input task.txt --output satellite_workflow_graph.json
"""

import os
import sys
import json
import argparse
from typing import Dict

from src.pipeline import TwoStageWorkflowPipeline


def print_formatted_results(result: Dict):
    """格式化打印两阶段生成结果。"""
    text = result['input_text']
    ge_data = result['stage1_event_graph_GE']
    gs_data = result['stage2_satellite_graph_GS']

    print("=" * 80)
    print(" 异构卫星网络任务需求到卫星类型级流程图生成系统 (Two-Stage Task Graph Generator)")
    print("=" * 80)
    print(f"\n【输入自然语言任务描述 X】:\n\"{text}\"\n")

    print("-" * 80)
    print("【阶段一产出】事件级任务流程图 G^E (Event-Level Task Workflow Graph)")
    print("-" * 80)
    print(f"抽取有效事件节点 |E|: {ge_data['num_events']}")
    for evt in ge_data['events']:
        print(f"  - [{evt['id']}] ({evt['type']}): \"{evt['text']}\" [字符区间: {evt['span']}]")

    print(f"\n时间依赖关系 Y_T ({len(ge_data['temporal_relations'])}):")
    for r in ge_data['temporal_relations']:
        print(f"  - {r['head']}  --->  {r['tail']} ({r['type']})")

    print(f"\n因果依赖关系 Y_C ({len(ge_data['causal_relations'])}):")
    for r in ge_data['causal_relations']:
        print(f"  - {r['head']}  --->  {r['tail']} ({r['type']})")

    print("\n" + "-" * 80)
    print("【阶段二产出】卫星类型级可调度任务流程图 G^S (Satellite-Type-Level Workflow Graph)")
    print("-" * 80)
    print(f"卫星类型执行节点 |Z|: {gs_data['num_nodes']}")
    for node in gs_data['nodes']:
        print(f"  - 节点 [{node['id']}]: \"{node['event_text']}\"")
        print(f"    |-- 匹配专家资源: {node['matched_resource']}")
        print(f"    |-- 最优主匹配类型 Z*: {node['primary_satellite_type']}")
        print(f"    +-- 全部候选卫星类型集合 Z: {node['candidate_satellite_types']}")

    print(f"\n继承的时序调度硬约束 ({len(gs_data['temporal_edges'])}):")
    for e in gs_data['temporal_edges']:
        print(f"  - [{e['head']} ({e['head_primary_sat_type']} | 候选:{e['head_candidate_sat_types']})]  --[{e['type']}]-->  [{e['tail']} ({e['tail_primary_sat_type']} | 候选:{e['tail_candidate_sat_types']})]")

    print(f"\n继承的因果调度前置约束 ({len(gs_data['causal_edges'])}):")
    for e in gs_data['causal_edges']:
        print(f"  - [{e['head']} ({e['head_primary_sat_type']} | 候选:{e['head_candidate_sat_types']})]  --[{e['type']}]-->  [{e['tail']} ({e['tail_primary_sat_type']} | 候选:{e['tail_candidate_sat_types']})]")

    print("\n" + "=" * 80)
    print(" 转换成功！生成的 G^S 流程图已具备显式执行逻辑与资源指向，可直接用于调度算法解算！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Natural Language Task to Satellite Type Workflow Graph Converter')
    parser.add_argument('--text', type=str, default=None, help='自然语言任务描述文本')
    parser.add_argument('--input', '--file', type=str, default=None, help='输入任务描述文本文件路径 (.txt)')
    parser.add_argument('--output', type=str, default=None, help='导出的 G^S JSON 文件路径 (可选)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer', type=str, default='checkpoints/tokenizer.json', help='Tokenizer 路径')
    parser.add_argument('--event_th', type=float, default=0.35, help='事件抽取判定阈值')
    parser.add_argument('--rel_th', type=float, default=0.35, help='关系分类判定阈值')

    args = parser.parse_args()

    # 确定输入文本
    if args.text:
        input_text = args.text
    elif args.input and os.path.exists(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    else:
        # 默认内置示例任务文本
        input_text = (
            "任务中心接收地震灾区应急成像任务后，卫星进入目标区域过境窗口，"
            "完成姿态调整并开启光学载荷，对震中区域进行高分辨率成像，"
            "随后执行云检测和道路断裂识别，压缩图像数据并缓存结果；"
            "当地面站可见时，建立数传链路并下传灾情图像。"
        )

    # 初始化两阶段流水线
    pipeline = TwoStageWorkflowPipeline(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    # 执行端到端转换
    result = pipeline.process_task(
        text=input_text,
        event_threshold=args.event_th,
        rel_threshold=args.rel_th,
        use_decoder=True,
    )

    # 打印格式化终端信息
    print_formatted_results(result)

    # 如果指定了输出文件路径，则保存 JSON
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result['stage2_satellite_graph_GS'], f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Satellite Type Workflow Graph G^S exported to: {args.output}\n")


if __name__ == '__main__':
    main()
