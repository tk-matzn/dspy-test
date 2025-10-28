#!/usr/bin/env python3
"""
DSPy内部のプロンプト流れを簡潔に表示
"""

import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI設定
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if azure_key and azure_endpoint:
    dspy.configure(lm=dspy.LM(
        model=f"azure/gpt-35-turbo",
        api_key=azure_key,
        api_base=azure_endpoint,
        api_version="2024-02-15-preview",
        max_tokens=200
    ))

# ========== プロンプト流れの詳細 ==========
print("=" * 70)
print("DSPy プロンプト内部フロー")
print("=" * 70)

print("\n📊 DSPyのプロンプト実行フロー:\n")

flow_diagram = """
┌────────────────────────────────────────────────────────────┐
│ 1. ユーザーコード                                          │
│    predict = dspy.Predict("question -> answer")             │
│    result = predict(question="Pythonとは？")              │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 2. シグネチャ解析                                          │
│    • 入力フィールド: question                             │
│    • 出力フィールド: answer                               │
│    • 自動的にプロンプトテンプレート生成                 │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 3. プロンプト構築                                          │
│                                                             │
│  [システムプロンプト]                                     │
│  You are an expert at answering questions in a              │
│  clear, concise, and informative manner...                 │
│                                                             │
│  [タスクプロンプト]                                       │
│  Follow the following format.                              │
│                                                             │
│  Question: ${question}                                     │
│  Answer: ${answer}                                         │
│                                                             │
│  [入力データ]                                             │
│  Question: Pythonとは？                                   │
│  Answer:                                                    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 4. LM API呼び出し                                          │
│    • Azure OpenAI APIにリクエスト送信                     │
│    • 完全なプロンプトを含むメッセージ                    │
│    • Temperature, Max Tokens等の設定を適用               │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 5. LLM処理                                                  │
│    LLM内部で:                                             │
│    • プロンプトをトークン化                              │
│    • トークンシーケンス予測                              │
│    • テキスト生成                                        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 6. 応答受信・解析                                          │
│    Answer: Python はオブジェクト指向の                    │
│             プログラミング言語です...                    │
│    ↓                                                        │
│    • LLMからの完全な応答を受信                           │
│    • 出力フィールド（answer）を抽出                     │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 7. 結果返却                                                 │
│    Prediction(answer="Python は...")                       │
│    • ユーザーアプリケーションに返却                      │
│    • 構造化された形式                                    │
└────────────────────────────────────────────────────────────┘
"""

print(flow_diagram)

# ========== 詳細な設定情報 ==========
print("\n📋 DSPy設定の詳細:\n")

lm = dspy.settings.lm
print(f"LMクラス: {lm.__class__.__name__}")
print(f"モデル: {lm.model}")
if hasattr(lm, 'max_tokens'):
    print(f"Max Tokens: {lm.max_tokens}")

# ========== プロンプトテンプレートの例 ==========
print("\n" + "=" * 70)
print("プロンプトテンプレートの実例")
print("=" * 70)

# Predictを作成
predict = dspy.Predict("topic -> explanation")

# シグネチャを確認
sig = predict.signature
print(f"\nシグネチャ: {sig}")
print(f"入力: {list(sig.input_fields.keys())}")
print(f"出力: {list(sig.output_fields.keys())}")

# プロンプトを実行（実際のAPI呼び出しはスキップ）
print("\n✓ プロンプトテンプレート準備完了")
print("  実際のAPI呼び出しではこのテンプレートが使用されます")

# ========== 複数ステップの流れ ==========
print("\n" + "=" * 70)
print("複数ステップのプロンプト流れ")
print("=" * 70)

multi_step_flow = """
例: 3ステップのパイプライン

┌──────────────────────────────────┐
│  入力: "AIについて説明して"      │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Step 1: キーポイント抽出         │
│  Prompt: "Text -> KeyPoints"     │
│  ↓ API呼び出し1回目 ↓           │
│  Output: "1. ML 2. NLP ..."      │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Step 2: 詳細説明作成            │
│  Prompt: "KeyPoints -> Details"  │
│  ↓ API呼び出し2回目 ↓           │
│  Output: "ML は..."              │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Step 3: 要約作成                │
│  Prompt: "Details -> Summary"    │
│  ↓ API呼び出し3回目 ↓           │
│  Output: "AIは..."               │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  最終結果: 構造化された回答      │
└──────────────────────────────────┘

💡 ポイント:
  • 各ステップは独立したAPI呼び出し
  • 前のステップの出力が次のステップの入力に
  • DSPyがこの流れを自動管理
"""

print(multi_step_flow)

# ========== プロンプト最適化の流れ ==========
print("\n" + "=" * 70)
print("プロンプト最適化の概念")
print("=" * 70)

optimization_flow = """
【プロンプト最適化プロセス】

初期プロンプト:
  "question -> answer"

           ↓ BootstrapFewShot等で最適化
           
改善されたプロンプト:
  "Instruction: あなたは専門家です..."
  
  Example 1:
    Question: ...
    Answer: ...
  
  Example 2:
    Question: ...
    Answer: ...
  
  Question: ${input_question}
  Answer: """

print(optimization_flow)

print("\n" + "=" * 70)
print("✅ プロンプト流れの説明が完了しました")
print("=" * 70)
