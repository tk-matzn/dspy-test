#!/usr/bin/env python3
"""
DSPy内部のプロンプト流れを確認するスクリプト
プロンプト、入力、出力の詳細なログを表示します
"""

import os
import dspy
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# ========== 1. DSPyの設定 ==========
print("=" * 70)
print("1. DSPy設定")
print("=" * 70)

# Azure OpenAI設定
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")

if azure_key and azure_endpoint:
    dspy.configure(lm=dspy.LM(
        model=f"azure/{deployment_name}",
        api_key=azure_key,
        api_base=azure_endpoint,
        api_version=api_version,
        max_tokens=500
    ))
    print("✓ Azure OpenAI が設定されました")
else:
    print("✗ Azure OpenAI キーが設定されていません")
    exit(1)

# ========== 2. プロンプトのロギングを有効にする ==========
print("\n" + "=" * 70)
print("2. DSPyロギングの有効化")
print("=" * 70)

# DSPyの内部ロギングを確認
try:
    dspy.enable_logging()
    print("✓ DSPyロギングを有効化しました")
except Exception as e:
    print(f"ロギング有効化: {e}")

# ========== 3. シンプルなPredictを実行 ==========
print("\n" + "=" * 70)
print("3. シンプルなPredict実行")
print("=" * 70)

# Predictの定義
simple_predict = dspy.Predict("question -> answer")

question = "Pythonとは何ですか？"
print(f"\n📝 入力質問: {question}")
print("-" * 70)

# 実行
result = simple_predict(question=question)

print(f"\n✅ 出力回答: {result.answer}")

# ========== 4. 実行履歴を確認 ==========
print("\n" + "=" * 70)
print("4. DSPy実行履歴の確認")
print("=" * 70)

# 設定からLMを取得して履歴を確認
lm = dspy.settings.lm

print(f"\n使用したLMモデル: {lm.model}")
print(f"API Base: {lm.api_base if hasattr(lm, 'api_base') else 'N/A'}")
print(f"Max Tokens: {lm.max_tokens}")

# ========== 5. 複雑なプロンプトフロー ==========
print("\n" + "=" * 70)
print("5. 複雑なプロンプトフロー（複数ステップ）")
print("=" * 70)

class MultiStepPipeline(dspy.Module):
    """複数ステップのパイプライン"""
    def __init__(self):
        super().__init__()
        # Step 1: キーポイント抽出
        self.extract_key_points = dspy.Predict("text -> key_points")
        # Step 2: 要約作成
        self.create_summary = dspy.Predict("key_points -> summary")
    
    def forward(self, text):
        print("\n[Step 1] キーポイント抽出")
        print(f"入力: {text[:50]}...")
        
        # Step 1実行
        result_step1 = self.extract_key_points(text=text)
        print(f"出力: {result_step1.key_points[:80]}...")
        
        print("\n[Step 2] 要約作成")
        print(f"入力: {result_step1.key_points[:50]}...")
        
        # Step 2実行
        result_step2 = self.create_summary(key_points=result_step1.key_points)
        print(f"出力: {result_step2.summary}")
        
        return dspy.Prediction(
            original_text=text,
            key_points=result_step1.key_points,
            summary=result_step2.summary
        )

# パイプラインを実行
pipeline = MultiStepPipeline()

text = "DSPyは言語モデルをプログラミングするためのフレームワークです。プロンプト工学を自動化し、モジュール化されたAIシステムを構築することができます。"
print(f"\n📝 入力テキスト: {text}\n")
print("-" * 70)

result = pipeline(text=text)

# ========== 6. プロンプトテンプレートの確認 ==========
print("\n" + "=" * 70)
print("6. プロンプトテンプレートの詳細")
print("=" * 70)

# Predictのシグネチャを確認
predict_sig = simple_predict.signature
print(f"\nシグネチャ: {predict_sig}")
print(f"入力フィールド: {list(predict_sig.input_fields.keys())}")
print(f"出力フィールド: {list(predict_sig.output_fields.keys())}")

# ========== 7. API呼び出しの詳細ログ ==========
print("\n" + "=" * 70)
print("7. LM設定の詳細")
print("=" * 70)

lm = dspy.settings.lm
print(f"\nLMクラス: {lm.__class__.__name__}")
print(f"モデル: {lm.model}")
print(f"Max Tokens: {lm.max_tokens}")

if hasattr(lm, 'api_base'):
    print(f"API Base: {lm.api_base}")
if hasattr(lm, 'api_version'):
    print(f"API Version: {lm.api_version}")

# ========== 8. プロンプト実行の流れを図示 ==========
print("\n" + "=" * 70)
print("8. プロンプト実行フロー")
print("=" * 70)

flow = """
┌─────────────────────────────────────────────────────────────┐
│          DSPy プロンプト実行フロー                          │
└─────────────────────────────────────────────────────────────┘

1. ユーザーコード
   ↓
   question = "Pythonとは何ですか？"
   result = predict(question=question)
   
2. Predict/ChainOfThoughtモジュール
   ↓
   • シグネチャを解析
   • プロンプトテンプレートを作成
   • 入力を形式化
   
3. プロンプト構築
   ↓
   ┌──────────────────────────────────┐
   │ システムプロンプト               │
   ├──────────────────────────────────┤
   │ You are an expert AI assistant   │
   │ that follows instructions...     │
   └──────────────────────────────────┘
   
   ┌──────────────────────────────────┐
   │ タスクプロンプト                 │
   ├──────────────────────────────────┤
   │ Question -> Answer                │
   │ ---                              │
   │ Question: Pythonとは...          │
   └──────────────────────────────────┘

4. LM API呼び出し
   ↓
   LM.forward(prompt)
   ↓
   • Azure OpenAI APIに送信
   • APIレスポンス受信
   
5. 応答解析
   ↓
   • LLMの出力を解析
   • フィールド抽出
   
6. 結果返却
   ↓
   Prediction(answer="Pythonは...プログラミング言語です")
"""

print(flow)

# ========== 9. デバッグ情報 ==========
print("\n" + "=" * 70)
print("9. デバッグ情報")
print("=" * 70)

# 現在の設定を表示
settings = dspy.settings
print(f"\n現在のDSPy設定:")
print(f"  LM: {settings.lm}")
print(f"  RM (Retriever): {settings.rm if hasattr(settings, 'rm') else 'N/A'}")

# ========== 10. プロンプト最適化の可能性 ==========
print("\n" + "=" * 70)
print("10. プロンプト最適化の機能")
print("=" * 70)

optimization_info = """
DSPyで利用可能な最適化機能:

1. BootstrapFewShot
   - 自動的に良い例（few-shot）を生成
   - プロンプトの効果を向上させる

2. COPRO (Chain-of-Thought Optimization)
   - 思考過程を自動最適化
   - プロンプトテンプレートの改善

3. MIPROv2
   - 複数メトリックの最適化
   - より複雑なタスク向け

4. GEPA (Gradual Evolution with Prompt Adaptation)
   - 段階的なプロンプト進化
   - 精度向上を重視

使用例:
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=4
    )
    optimized_program = optimizer.compile(program, trainset)
"""

print(optimization_info)

print("\n" + "=" * 70)
print("✅ プロンプト流れの確認が完了しました")
print("=" * 70)
