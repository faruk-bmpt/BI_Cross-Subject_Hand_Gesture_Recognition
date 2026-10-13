#!/usr/bin/env python3
# ================================================================
# RUN_ALL_BLOCKS.py
# ================================================================
# MASTER EXECUTION SCRIPT
# ================================================================

import os
import sys
import subprocess
from datetime import datetime

print("="*80)
print("RUNNING ALL BIO-IMPEDANCE PIPELINE BLOCKS")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

blocks = [
    "BI_BLOCK_0_BASE_MODEL_PIPELINE.py",
    "BI_BLOCK_1_DATASET_ANALYSIS.py",
    "BI_BLOCK_2_SUBJECT_INDEPENDENT.py",
    "BI_BLOCK_3_SUBJECT_DEPENDENT.py",
    "BI_BLOCK_4_ABLATION_STUDY.py",
    "BI_BLOCK_5_EXPLAINABILITY.py",
    "BI_BLOCK_6_STATISTICAL_ANALYSIS.py",
    "BI_BLOCK_7_PRACTICAL_DEPLOYMENT.py",
    "BI_BLOCK_8_INTERNAL_VISUALIZATION.py"
]

for i, block in enumerate(blocks):
    print(f"\n{'='*60}")
    print(f"Running Block {i}: {block}")
    print('='*60)
    
    if os.path.exists(block):
        start_time = datetime.now()
        result = subprocess.run([sys.executable, block], capture_output=False)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"✅ Block {i} completed in {duration:.2f} seconds")
        else:
            print(f"❌ Block {block} failed with return code {result.returncode}")
            inp = input("Continue with next block? (y/n): ")
            if inp.lower() != 'y':
                break
    else:
        print(f"⚠️ File not found: {block}")
        inp = input("Continue with next block? (y/n): ")
        if inp.lower() != 'y':
            break

print("\n" + "="*80)
print(f"✅ PIPELINE EXECUTION COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)