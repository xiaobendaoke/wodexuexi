#!/usr/bin/env python3
"""分析所有提取的文献内容"""

import json
from pathlib import Path
import re

# 目录配置
EXTRACTED_DIR = Path("/home/PengYanghan/Lunwen/wodexuexi/extracted_references")
OUTPUT_FILE = Path("/home/PengYanghan/Lunwen/wodexuexi/analysis_results/detailed_analysis.json")

def analyze_document(content, filename):
    """分析单篇文献"""
    content_lower = content.lower()

    # 1. 检测局限性章节
    limitation_section = bool(re.search(
        r'(##?\s*(limitation|limitations|不足|局限性|缺点|弱点))',
        content_lower
    ))

    # 2. 检测未来工作章节
    future_work_section = bool(re.search(
        r'(##?\s*(future\s+work|future\s+research|future\s+direction|未来工作|未来研究|未来方向|进一步改进|进一步研究))',
        content_lower
    ))

    # 3. 检测明确承认不足的表述
    explicit_limitations = []
    patterns = [
        r'our\s+method\s+has\s+limitation',
        r'the\s+proposed\s+(method|approach)\s+cannot',
        r'our\s+(method|approach)\s+(is|are)\s+not\s+able',
        r'the\s+proposed\s+(method|approach)\s+(has|have)\s+limitation',
        r'我们的方法.{0,30}(不足|局限|缺点)',
        r'所提方法.{0,30}(无法|未能|存在)',
        r'本文方法.{0,30}(不足|局限|缺点)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            explicit_limitations.extend(matches)

    # 4. 检测对比实验中的劣势
    comparison_weakness = bool(re.search(
        r'(outperform|better\s+than|superior\s+to|优于).{0,80}(our|proposed|本文|所提)',
        content_lower
    ))

    # 5. 提取局限性相关句子
    limitation_sentences = []
    sentences = re.split(r'[.。!！?？]', content)
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if any(kw in sent_lower for kw in ['limitation', 'weakness', 'drawback', 'shortcoming', '不足', '局限', '缺点']):
            if len(sent.strip()) > 20 and len(sent.strip()) < 300:
                limitation_sentences.append(sent.strip())

    # 6. 提取未来工作相关句子
    future_work_sentences = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if any(kw in sent_lower for kw in ['future work', 'future research', 'future direction', '未来工作', '未来研究', '进一步']):
            if len(sent.strip()) > 20 and len(sent.strip()) < 300:
                future_work_sentences.append(sent.strip())

    # 7. 计算得分
    score = 0
    if limitation_section:
        score += 3
    if future_work_section:
        score += 1
    if explicit_limitations:
        score += 2
    if comparison_weakness:
        score += 1

    return {
        "filename": filename,
        "has_limitation_section": limitation_section,
        "has_future_work_section": future_work_section,
        "explicit_limitations": len(explicit_limitations) > 0,
        "comparison_weakness": comparison_weakness,
        "score": score,
        "details": {
            "limitation_count": len(re.findall(r'limitation|不足|局限', content_lower)),
            "future_work_count": len(re.findall(r'future\s+work|未来工作', content_lower)),
            "limitation_sentences": limitation_sentences[:5],  # 最多5个句子
            "future_work_sentences": future_work_sentences[:5],
        }
    }

def main():
    """主函数"""
    # 获取所有提取的 Markdown 文件
    md_files = sorted(EXTRACTED_DIR.glob("*.md"))

    print(f"找到 {len(md_files)} 篇已提取的文献")
    print("开始分析...")
    print("="*80)

    results = []

    for i, md_path in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] 分析: {md_path.name[:50]}...")

        # 读取内容
        try:
            content = md_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            continue

        # 分析文献
        analysis = analyze_document(content, md_path.name)
        results.append(analysis)

        # 打印简要结果
        score = analysis['score']
        status = "✅" if score >= 4 else "⚠️" if score >= 2 else "❌"
        print(f"  得分: {score}/7 {status}")

        if analysis['has_limitation_section']:
            print(f"    - 有局限性章节")
        if analysis['has_future_work_section']:
            print(f"    - 有未来工作章节")
        if analysis['explicit_limitations']:
            print(f"    - 明确承认不足")

    # 保存结果
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print(f"✅ 分析完成，结果保存到: {OUTPUT_FILE}")

    # 生成汇总表格
    generate_summary_table(results)

def generate_summary_table(results):
    """生成汇总表格"""
    print("\n" + "="*120)
    print("详细分析汇总表格")
    print("="*120)
    print(f"{'序号':<4} {'文献名称':<55} {'局限性章节':<10} {'未来工作':<10} {'明确承认不足':<12} {'得分':<6}")
    print("-"*120)

    for i, r in enumerate(results, 1):
        name = r['filename'][:52] + "..." if len(r['filename']) > 55 else r['filename']
        lim = "✅" if r['has_limitation_section'] else "❌"
        future = "✅" if r['has_future_work_section'] else "❌"
        explicit = "✅" if r['explicit_limitations'] else "❌"
        score = r['score']

        print(f"{i:<4} {name:<55} {lim:<10} {future:<10} {explicit:<12} {score:<6}")

    print("-"*120)

    # 统计
    total = len(results)
    has_lim = sum(1 for r in results if r['has_limitation_section'])
    has_future = sum(1 for r in results if r['has_future_work_section'])
    has_explicit = sum(1 for r in results if r['explicit_limitations'])
    avg_score = sum(r['score'] for r in results) / total if total > 0 else 0

    print(f"\n📊 统计结果:")
    print(f"  总文献数: {total}")
    print(f"  有局限性章节: {has_lim} ({has_lim/total*100:.1f}%)")
    print(f"  有未来工作: {has_future} ({has_future/total*100:.1f}%)")
    print(f"  明确承认不足: {has_explicit} ({has_explicit/total*100:.1f}%)")
    print(f"  平均得分: {avg_score:.2f}/7")

if __name__ == "__main__":
    main()
