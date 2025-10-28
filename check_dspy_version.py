#!/usr/bin/env python3
"""DSPy 3.0.3でのOpenAI設定を確認するスクリプト"""

import inspect
import dspy

# 利用可能なLMクラスを確認
print("=== DSPy LM Configuration ===\n")
print(f"DSPy Version: {dspy.__version__}\n")

# dspy.clientsで利用可能なモジュールを確認
print("Available in dspy.clients:")
import dspy.clients
for item in dir(dspy.clients):
    if not item.startswith('_'):
        print(f"  - {item}")

print("\n=== Trying to import OpenAI ===")
try:
    from dspy.clients.openai import ChatCompletionClient
    print("✓ ChatCompletionClient imported successfully")
except ImportError as e:
    print(f"✗ ChatCompletionClient import failed: {e}")

try:
    from dspy import LM
    print("✓ LM imported successfully")
    print(f"  LM class: {LM}")
except ImportError as e:
    print(f"✗ LM import failed: {e}")

# configureの使用方法を確認
print("\n=== Testing dspy.configure ===")
try:
    # テスト用の設定
    dspy.configure(lm=dspy.LM(model="openai/gpt-3.5-turbo", api_key="test"))
    print("✓ dspy.configure with LM model='openai/gpt-3.5-turbo' works")
except Exception as e:
    print(f"✗ Error: {e}")
