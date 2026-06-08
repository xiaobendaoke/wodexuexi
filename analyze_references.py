#!/usr/bin/env python3
"""分析参考文献：检查是否只夸优势还是也提到了不足"""

import os
import subprocess
import json
from pathlib import Path

# 文献目录
REF_DIR = Path("/home/PengYanghan/Lunwen/CanKaoWenXian")

# 获取所有 PDF 文件
pdf_files = sorted(REF_DIR.glob("*.pdf"))

print(f"找到 {len(pdf_files)} 篇 PDF 文献\n")

# 提取文本的函数
def extract_text(pdf_path, max_pages=10):
    """提取 PDF 前 max_pages 页的文本"""
    try:
        result = subprocess.run(
            ["pdftotext", "-l", str(max_pages), "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

# 分析文本的函数
def analyze_text(text, filename):
    """分析文本中是否提到优势和不足"""
    text_lower = text.lower()

    # 优势关键词（英文+中文）
    advantage_keywords = [
        "advantage", "benefit", "superior", "outperform", "better than",
        "improve", "enhance", "effective", "efficient", "optimal",
        "significant", "remarkable", "excellent", "promising",
        "优势", "优点", "好处", "提升", "改进", "优于", "显著", "有效"
    ]

    # 不足关键词
    limitation_keywords = [
        "limitation", "weakness", "drawback", "shortcoming", "challenge",
        "future work", "future research", "further improvement", "need to",
        "however", "but", "although", "despite", "constraint",
        "不足", "局限", "缺点", "挑战", "未来工作", "改进方向", "但是", "然而"
    ]

    # 检查是否提到优势
    has_advantage = any(kw in text_lower for kw in advantage_keywords)

    # 检查是否提到不足
    has_limitation = any(kw in text_lower for kw in limitation_keywords)

    # 提取相关句子
    advantage_sentences = []
    limitation_sentences = []

    sentences = text.replace('\n', ' ').split('.')
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in advantage_keywords) and len(sent.strip()) > 20:
            advantage_sentences.append(sent.strip()[:150])
        if any(kw in sent_lower for kw in limitation_keywords) and len(sent.strip()) > 20:
            limitation_sentences.append(sent.strip()[:150])

    return {
        "filename": filename,
        "has_advantage": has_advantage,
        "has_limitation": has_limitation,
        "advantage_count": len(advantage_sentences),
        "limitation_count": len(limitation_sentences),
        "advantage_examples": advantage_sentences[:3],  # 最多3个例子
        "limitation_examples": limitation_sentences[:3]
    }

# 主分析流程
results = []

for i, pdf_path in enumerate(pdf_files, 1):
    print(f"[{i}/{len(pdf_files)}] 分析: {pdf_path.name[:60]}...")

    # 提取文本
    text = extract_text(pdf_path, max_pages=15)

    if text.startswith("Error"):
        print(f"  ⚠️ 提取失败: {text}")
        results.append({
            "filename": pdf_path.name,
            "has_advantage": None,
            "has_limitation": None,
            "error": text
        })
        continue

    # 分析文本
    analysis = analyze_text(text, pdf_path.name)
    results.append(analysis)

    # 打印简要结果
    status_adv = "✅" if analysis["has_advantage"] else "❌"
    status_lim = "✅" if analysis["has_limitation"] else "❌"
    print(f"  优势: {status_adv} ({analysis['advantage_count']}条) | 不足: {status_lim} ({analysis['limitation_count']}条)")

# 保存结果
output_path = REF_DIR.parent / "wodexuexi" / "reference_analysis.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 分析完成，结果保存到: {output_path}")

# 生成汇总表格
print("\n" + "="*80)
print("汇总表格")
print("="*80)
print(f"{'序号':<4} {'文献名称':<55} {'提到优势':<10} {'提到不足':<10}")
print("-"*80)

for i, r in enumerate(results, 1):
    name = r["filename"][:52] + "..." if len(r["filename"]) > 55 else r["filename"]
    adv = "✅" if r.get("has_advantage") else "❌"
    lim = "✅" if r.get("has_limitation") else "❌"
    print(f"{i:<4} {name:<55} {adv:<10} {lim:<10}")

print("-"*80)
print(f"总计: {len(results)} 篇文献")
print(f"提到优势: {sum(1 for r in results if r.get('has_advantage'))} 篇")
print(f"提到不足: {sum(1 for r in results if r.get('has_limitation'))} 篇")
