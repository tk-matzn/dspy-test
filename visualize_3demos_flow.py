#!/usr/bin/env python3
"""
sample_program.py の3つのデモのプロンプト流れを可視化
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

print("=" * 80)
print("sample_program.py の3つのデモ - プロンプト流れの可視化")
print("=" * 80)

# ========== デモ 1: シンプルなPredict ==========
print("\n" + "▶" * 40)
print("デモ 1️⃣ : シンプルデモ")
print("▶" * 40)

simple_flow = """
【シンプルデモの構成】

コード:
  qa = dspy.Predict("question -> answer")
  result = qa(question="Pythonとは何ですか？")

プロンプト流れ:

  ┌────────────────────────────────────────────────┐
  │ 入力                                           │
  │ question = "Pythonとは何ですか？"              │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ Step 1: Signature解析                        │
  │ "question -> answer"                          │
  │ • 入力: question                              │
  │ • 出力: answer                                │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ Step 2: プロンプト構築                        │
  │                                               │
  │ [システムプロンプト]                          │
  │ You are an expert at answering questions      │
  │ in a clear, concise manner...                │
  │                                               │
  │ [タスク指示]                                  │
  │ Given the fields `question`, produce          │
  │ the fields `answer`.                          │
  │                                               │
  │ [形式]                                        │
  │ Question: ${question}                         │
  │ Answer: ${answer}                             │
  │                                               │
  │ [実際の入力]                                  │
  │ Question: Pythonとは何ですか？              │
  │ Answer:                                       │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ Step 3: Azure OpenAI API呼び出し              │
  │ • 完全なプロンプトを送信                       │
  │ • max_tokens=200で設定                       │
  │ • temperature=1（デフォルト）                 │
  │                                               │
  │ [API呼び出し 1回]                            │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ Step 4: LLM処理                              │
  │ • プロンプトのトークン化                      │
  │ • トークンシーケンス予測                      │
  │ • テキスト生成                               │
  │   "Python はオブジェクト指向の..."            │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ Step 5: 応答解析                             │
  │ Answer: フィールドから抽出                    │
  │ "Python はオブジェクト指向の..."              │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ 出力                                          │
  │ Prediction(answer="Python は...")             │
  └────────────────────────────────────────────────┘

📊 統計:
  • API呼び出し数: 1回
  • Signature個数: 1個
  • 処理ステップ: 5段階
  • 複雑度: ⭐ (最もシンプル)
"""

print(simple_flow)

# ========== デモ 2: パイプライン ==========
print("\n" + "▶" * 40)
print("デモ 2️⃣ : パイプラインデモ")
print("▶" * 40)

pipeline_flow = """
【パイプラインデモの構成】

コード:
  class SimplePipeline(dspy.Module):
      def __init__(self):
          self.extract_points = dspy.Predict("question -> key_points")
          self.generate_answer = dspy.Predict("key_points -> answer")
      
      def forward(self, question):
          points = self.extract_points(question=question)
          result = self.generate_answer(key_points=points.key_points)
          return result

プロンプト流れ（複数ステップ）:

  ┌────────────────────────────────────────────────┐
  │ 入力                                           │
  │ question = "日本の首都は？その特徴は？"        │
  └────────────────────────────────────────────────┘
                       ↓
  ┌════════════════════════════════════════════════┐
  │ 【ステップ1】キーポイント抽出                 │
  ├────────────────────────────────────────────────┤
  │ Signature: "question -> key_points"           │
  │                                               │
  │ プロンプト:                                   │
  │ ─────────────────────────────────────────     │
  │ [システムプロンプト]                          │
  │ You are an expert...                         │
  │                                               │
  │ [タスク]                                      │
  │ Given question, produce key_points           │
  │                                               │
  │ [形式]                                        │
  │ Question: [入力]                             │
  │ KeyPoints: [抽出結果]                        │
  │                                               │
  │ [実際の入力]                                  │
  │ Question: 日本の首都は？その特徴は？         │
  │ KeyPoints:                                   │
  │ ─────────────────────────────────────────     │
  │                                               │
  │ [API呼び出し 1/2]                            │
  │ 出力: "東京、経済中心、文化..."               │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ 中間結果の保存                                │
  │ key_points = "東京、経済中心、文化..."        │
  └────────────────────────────────────────────────┘
                       ↓
  ┌════════════════════════════════════════════════┐
  │ 【ステップ2】回答生成                         │
  ├────────────────────────────────────────────────┤
  │ Signature: "key_points -> answer"             │
  │                                               │
  │ プロンプト:                                   │
  │ ─────────────────────────────────────────     │
  │ [システムプロンプト]                          │
  │ You are an expert...                         │
  │                                               │
  │ [タスク]                                      │
  │ Given key_points, produce answer             │
  │                                               │
  │ [形式]                                        │
  │ KeyPoints: [前ステップの出力]                │
  │ Answer: [最終回答]                           │
  │                                               │
  │ [実際の入力]                                  │
  │ KeyPoints: 東京、経済中心、文化...           │
  │ Answer:                                       │
  │ ─────────────────────────────────────────     │
  │                                               │
  │ [API呼び出し 2/2]                            │
  │ 出力: "東京は日本の首都で..."                │
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────────┐
  │ 最終出力                                       │
  │ Prediction(                                   │
  │   question="...",                             │
  │   key_points="...",                           │
  │   answer="..."                                │
  │ )                                             │
  └────────────────────────────────────────────────┘

📊 統計:
  • API呼び出し数: 2回
  • Signature個数: 2個
  • 処理ステップ: 2つの独立したパイプラインステップ
  • 複雑度: ⭐⭐⭐ (中程度)
  • データフロー: 線形（Step1の出力 → Step2の入力）
"""

print(pipeline_flow)

# ========== デモ 3: 最適化デモ ==========
print("\n" + "▶" * 40)
print("デモ 3️⃣ : 最適化デモ")
print("▶" * 40)

optimization_flow = """
【最適化デモの構成】

コード:
  trainset = [
      dspy.Example(question="2+2は?", answer="4"),
      dspy.Example(question="3×5は?", answer="15"),
  ]
  program = dspy.Predict("question -> answer")
  
  # 実際の最適化はBootstrapFewShot等で実行
  optimizer = BootstrapFewShot(metric=accuracy)
  optimized_program = optimizer.compile(program, trainset)

プロンプト流れ（イメージ）:

┌─────────────────────────────────────────────────────┐
│ フェーズ1: 初期プログラム                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 初期プロンプト（最適化前）:                        │
│ ┌─────────────────────────────────────────────┐   │
│ │ You are an expert...                        │   │
│ │                                             │   │
│ │ Question: ${question}                       │   │
│ │ Answer: ${answer}                           │   │
│ │                                             │   │
│ │ Question: 2+2は?                           │   │
│ │ Answer:                                     │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ このプロンプトで trainset を実行                 │
│ • Example 1: 2+2 → 期待値: 4                    │
│ • Example 2: 3×5 → 期待値: 15                   │
│                                                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ フェーズ2: プロンプト最適化                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ BootstrapFewShot等により:                         │
│ 1. 成功したデモを特定                             │
│ 2. それらをプロンプトに追加                       │
│ 3. 新しいプロンプトテンプレートを生成             │
│                                                     │
│ 最適化後のプロンプト:                             │
│ ┌─────────────────────────────────────────────┐   │
│ │ You are an expert at math problems...      │   │
│ │ Answer clearly and concisely.              │   │
│ │                                             │   │
│ │ Follow the examples below:                  │   │
│ │                                             │   │
│ │ 【デモンストレーション1】                   │   │
│ │ Question: 2+2は?                          │   │
│ │ Answer: 4                                   │   │
│ │                                             │   │
│ │ 【デモンストレーション2】                   │   │
│ │ Question: 3×5は?                          │   │
│ │ Answer: 15                                  │   │
│ │                                             │   │
│ │ Question: ${question}                       │   │
│ │ Answer: ${answer}                           │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ [複数回のAPI呼び出しで最適化 - N回]              │
│                                                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ フェーズ3: 最適化されたプログラム                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 改善されたプロンプトテンプレート:                 │
│ • より詳細な指示                                 │
│ • 成功したデモンストレーション                   │
│ • 優れた例から学習                              │
│                                                     │
│ このプログラムで新しい質問を実行:                │
│ Question: 10+5は?                              │
│ → より正確な回答が得られる確率が向上             │
│                                                     │
└─────────────────────────────────────────────────────┘

📊 統計:
  • 初期API呼び出し: 1回（ベースライン）
  • 最適化フェーズのAPI呼び出し: N回（最適化試行）
  • Signature個数: 1個
  • 複雑度: ⭐⭐⭐⭐⭐ (最も複雑)
  • 実行モード: メタレベル（プロンプト自体を改善）

💡 このサンプルでは最適化の"概念"を示しています
   実際の最適化には BootstrapFewShot などの
   テレプロンプター（optimizer）が必要です。
"""

print(optimization_flow)

# ========== 比較表 ==========
print("\n" + "=" * 80)
print("3つのデモの比較")
print("=" * 80)

comparison = """
┌──────────────────┬──────────────┬──────────────────┬──────────────────┐
│ 項目             │ デモ1:Simple │ デモ2:Pipeline   │ デモ3:最適化     │
├──────────────────┼──────────────┼──────────────────┼──────────────────┤
│ API呼び出し数    │ 1回          │ 2回              │ N回              │
│ Signature数      │ 1個          │ 2個              │ 1個              │
│ 処理段階         │ 5ステップ    │ 2ステップ×2      │ 3フェーズ        │
│ データフロー     │ 単純         │ 線形パイプライン │ メタレベル       │
│ 複雑度           │ 低           │ 中               │ 高               │
│ 実行時間         │ 短           │ 中               │ 長               │
│ コスト(API)      │ 低           │ 中               │ 高               │
│ 出力型           │ 単一値       │ 複数値           │ プログラム       │
├──────────────────┼──────────────┼──────────────────┼──────────────────┤
│ ユースケース     │ 単純な質問応答│ 複数ステップ処理 │ 精度向上が必要   │
│                  │              │ (複雑な分析)     │ (本番環境向け)   │
└──────────────────┴──────────────┴──────────────────┴──────────────────┘
"""

print(comparison)

# ========== 全体フロー図 ==========
print("\n" + "=" * 80)
print("全体のプロンプト処理フロー")
print("=" * 80)

overall_flow = """
【DSPyプログラム全体の処理フロー】

ユーザーコード (sample_program.py)
  ├─ デモ1: qa = dspy.Predict("question -> answer")
  │                    ↓
  │         [1つのAPI呼び出し]
  │                    ↓
  │         Prediction(answer="...")
  │
  ├─ デモ2: pipeline.forward(question)
  │         Step 1: extract_points()
  │                    ↓
  │         [1つのAPI呼び出し]
  │                    ↓
  │         Step 2: generate_answer()
  │                    ↓
  │         [1つのAPI呼び出し]
  │                    ↓
  │         Prediction(...)
  │
  └─ デモ3: optimizer.compile(program, trainset)
           複数回の試行・評価
                    ↓
           改善されたプロンプト
                    ↓
           最適化されたプログラム

【API呼び出しの詳細】

デモ1の流れ:
  User Input → Signature Parse → Prompt Build → API Call → Parse Response → Output
  
デモ2の流れ:
  User Input → 
    Signature1 Parse → Prompt Build → API Call1 → Parse Response1 →
    (中間値) →
    Signature2 Parse → Prompt Build → API Call2 → Parse Response2 →
  Output

デモ3の流れ:
  Program + TrainSet → 
    (多くのAPI呼び出しで試行・評価) →
    改善されたプロンプトテンプレート →
    New Program

【API呼び出し数の合計】
  デモ1: 1回
  デモ2: 2回
  デモ3: 複数回（最適化アルゴリズムに依存）
  ─────────────
  合計: 3回以上のAPI呼び出し
"""

print(overall_flow)

print("\n" + "=" * 80)
print("✅ 3つのデモのプロンプト流れの可視化が完了しました")
print("=" * 80)

print("\n📚 詳細を知るには:")
print("  • sample_program.py: 実際の実装")
print("  • prompt_flow_explanation.py: 基本的なフロー")
print("  • prompt_content_detail.py: プロンプト内容の詳細")
