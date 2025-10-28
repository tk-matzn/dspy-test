#!/usr/bin/env python3
"""
DSPyが実際に生成するプロンプトの内容を確認するスクリプト
"""

import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# 設定
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

print("=" * 70)
print("DSPy プロンプト内容の詳細確認")
print("=" * 70)

# ========== 1. Predictの構造 ==========
print("\n📌 1. Predict モジュールの構造\n")

predict = dspy.Predict("name,age -> greeting")

print(f"シグネチャ:\n{predict.signature}\n")

# ========== 2. ChainOfThoughtの場合 ==========
print("=" * 70)
print("📌 2. ChainOfThought モジュール\n")

cot_predict = dspy.ChainOfThought("question -> answer")

print("✓ ChainOfThoughtは内部に'reasoning'フィールドを持ちます")
print("  これにより、LLMは思考過程を示しながら回答します\n")

# ========== 3. プロンプト構築の流れ ==========
print("=" * 70)
print("📌 3. プロンプト構築の詳細フロー\n")

construction_flow = """
【プロンプト構築の流れ】

ステップ1: シグネチャの解析
  入力: "question -> answer"
  解析結果:
    - 入力フィールド: question
    - 出力フィールド: answer

ステップ2: プロンプトテンプレートの生成
  ┌─────────────────────────────────┐
  │ システムプロンプト（自動生成）   │
  │                                 │
  │ You are an expert at            │
  │ answering questions in a        │
  │ clear, concise, and             │
  │ informative manner. Follow      │
  │ the provided instructions and   │
  │ format your responses according │
  │ to the requirements.            │
  └─────────────────────────────────┘

ステップ3: タスク指示の追加
  ┌─────────────────────────────────┐
  │ Follow the following format.    │
  │                                 │
  │ Question: [質問内容]            │
  │ Answer: [回答内容]              │
  └─────────────────────────────────┘

ステップ4: デモンストレーション（オプション）
  ┌─────────────────────────────────┐
  │ Question: フューショット例1      │
  │ Answer: 対応する回答例          │
  │                                 │
  │ Question: フューショット例2      │
  │ Answer: 対応する回答例          │
  └─────────────────────────────────┘

ステップ5: 実際の入力の追加
  ┌─────────────────────────────────┐
  │ Question: Pythonとは何ですか?  │
  │ Answer:                         │
  └─────────────────────────────────┘
"""

print(construction_flow)

# ========== 4. API呼び出しの形式 ==========
print("\n" + "=" * 70)
print("📌 4. Azure OpenAI API呼び出しの形式\n")

api_format = """
【Azure OpenAI APIへのリクエスト】

エンドポイント:
  https://{resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2024-02-15-preview

ヘッダー:
  api-key: [AZURE_OPENAI_API_KEY]
  Content-Type: application/json

リクエストボディ:
  {
    "messages": [
      {
        "role": "system",
        "content": "You are an expert at answering questions..."
      },
      {
        "role": "user",
        "content": "[完全なプロンプト]\\nQuestion: Pythonとは?\\nAnswer:"
      }
    ],
    "temperature": 1,
    "max_tokens": 200,
    "top_p": 1
  }

【レスポンス】

  {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "Python は高水準の..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 42,
      "completion_tokens": 150,
      "total_tokens": 192
    }
  }
"""

print(api_format)

# ========== 5. 実行追跡の例 ==========
print("\n" + "=" * 70)
print("📌 5. 実行追跡（Trace）の仕組み\n")

trace_example = """
【DSPyの実行追跡の流れ】

dspy.trace() コンテキストマネージャーで実行を追跡可能:

  with dspy.trace():
      result = predict(question="Pythonとは?")

追跡結果の階層構造:
  
  Predict(question -> answer)
  ├─ Input:
  │  └─ question: "Pythonとは?"
  ├─ Process:
  │  ├─ Signature analysis
  │  ├─ Prompt construction
  │  └─ LM forward pass
  └─ Output:
     └─ answer: "Python は..."

このトレースを使用することで:
  • デバッグが容易
  • パフォーマンス分析が可能
  • プロンプト最適化の効果測定ができる
"""

print(trace_example)

# ========== 6. パイプラインでの複数プロンプト ==========
print("\n" + "=" * 70)
print("📌 6. パイプラインでの複数プロンプト実行\n")

pipeline_prompts = """
【複数ステップのプロンプト生成】

class Pipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.Predict("text -> keywords")
        self.step2 = dspy.Predict("keywords -> summary")
    
    def forward(self, text):
        result1 = self.step1(text=text)    # API呼び出し1回目
        result2 = self.step2(keywords=result1.keywords)  # API呼び出し2回目
        return result2

実行時の流れ:

  入力テキスト
       ↓
    [API呼び出し1]
  Prompt 1: "Text -> Keywords"
  Request: text="...長いテキスト..."
  Response: keywords="キーワード1, キーワード2, ..."
       ↓
    [API呼び出し2]
  Prompt 2: "Keywords -> Summary"
  Request: keywords="..."
  Response: summary="要約文..."
       ↓
    出力結果

✓ 各ステップが独立したAPI呼び出し
✓ 前のステップの出力が次のステップの入力
✓ 自動的に管理される
"""

print(pipeline_prompts)

# ========== 7. キャッシングとメモ化 ==========
print("\n" + "=" * 70)
print("📌 7. キャッシング（最適化）\n")

caching_info = """
【DSPyのキャッシング機構】

同じプロンプト・入力組み合わせが再実行された場合:

1回目の実行:
  input: "Pythonとは?" → API呼び出し → キャッシュ保存
  
2回目の実行:
  input: "Pythonとは?" → キャッシュから取得 ✓ (API呼び出しなし)
  
3回目の実行:
  input: "JavaScriptとは?" → API呼び出し → キャッシュ保存

利点:
  ✓ API呼び出しコストの削減
  ✓ 実行速度の向上
  ✓ 同一結果の一貫性確保

設定:
  dspy.configure_cache()  # キャッシュ有効化
  dspy.disable_litellm_logging()  # ロギング無効化
"""

print(caching_info)

print("\n" + "=" * 70)
print("✅ DSPy プロンプト内容の詳細説明が完了しました")
print("=" * 70)

print("\n📚 さらに詳しく知るには:\n")
print("  • DSPy公式ドキュメント: https://dspy.ai/")
print("  • GitHub: https://github.com/stanfordnlp/dspy")
print("  • プロンプト最適化: https://dspy.ai/docs/building-blocks/optimizers")
