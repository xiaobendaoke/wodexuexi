#!/usr/bin/env python3
"""批量提取 PDF 文献内容"""

import subprocess
import os
from pathlib import Path
import time

# 文献目录
REF_DIR = Path("/home/PengYanghan/Lunwen/CanKaoWenXian")
OUTPUT_DIR = Path("/home/PengYanghan/Lunwen/wodexuexi/extracted_references")

# 获取所有 PDF 文件
pdf_files = sorted(REF_DIR.glob("*.pdf"))

print(f"找到 {len(pdf_files)} 篇 PDF 文献")
print("开始批量提取...")
print("="*80)

success_count = 0
fail_count = 0
skip_count = 0

for i, pdf_path in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] 提取: {pdf_path.name[:60]}...")

    # 生成输出文件名（MinerU 会自动命名）
    # 检查是否已存在
    existing_files = list(OUTPUT_DIR.glob(f"*{pdf_path.stem}*"))
    if existing_files:
        print(f"  ✅ 已存在，跳过")
        skip_count += 1
        continue

    # 调用 MinerU 提取
    cmd = f'no_proxy="*.aliyuncs.com" mineru-open-api flash-extract "{pdf_path}" --language ch -o "{OUTPUT_DIR}"'

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  ✅ 成功 ({elapsed:.1f}s)")
            success_count += 1
        else:
            print(f"  ❌ 失败: {result.stderr[:100]}")
            fail_count += 1

    except subprocess.TimeoutExpired:
        print(f"  ⏰ 超时 (>5分钟)")
        fail_count += 1
    except Exception as e:
        print(f"  ❌ 错误: {str(e)[:100]}")
        fail_count += 1

print("\n" + "="*80)
print(f"✅ 批量提取完成")
print(f"  成功: {success_count}")
print(f"  跳过: {skip_count}")
print(f"  失败: {fail_count}")
print(f"  总计: {len(pdf_files)}")
