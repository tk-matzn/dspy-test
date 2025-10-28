# DSPy プロンプト内部フロー解説

## 概要

DSPyはプロンプトエンジニアリングを自動化するフレームワークです。内部でどのようにプロンプトが構築・実行されるかを理解することは、効果的なプログラム設計に不可欠です。

---

## 📊 プロンプト実行の7ステップ

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: ユーザーコード                                  │
│  predict = dspy.Predict("question -> answer")           │
│  result = predict(question="Pythonとは？")             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: シグネチャ解析                                  │
│  • 入力フィールド抽出: question                        │
│  • 出力フィールド抽出: answer                          │
│  • 自動プロンプトテンプレート生成                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: プロンプト構築                                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ システムプロンプト（自動生成）                    │ │
│  │ You are an expert at answering...                │ │
│  ├───────────────────────────────────────────────────┤ │
│  │ タスク指示                                        │ │
│  │ Follow the following format:                     │ │
│  │ Question: ${question}                           │ │
│  │ Answer: ${answer}                               │ │
│  ├───────────────────────────────────────────────────┤ │
│  │ 入力データ                                        │ │
│  │ Question: Pythonとは？                          │ │
│  │ Answer:                                         │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: LM API呼び出し                                  │
│  Azure OpenAI APIにリクエスト送信                       │
│  • 完全なプロンプトを含む                             │
│  • APIキー、バージョン、設定を適用                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: LLM処理                                          │
│  • プロンプトのトークン化                             │
│  • トークンシーケンスの予測                           │
│  • テキスト生成                                       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 6: 応答受信・解析                                  │
│  Answer: Python はオブジェクト指向の...               │
│  • LLMからの完全な応答を受信                         │
│  • 出力フィールド（answer）を抽出                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 7: 結果返却                                         │
│  Prediction(answer="Python は...")                      │
│  • 構造化された形式でユーザーに返却                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 実装詳細

### 1. シグネチャの構造

```python
# シンプルな例
predict = dspy.Predict("question -> answer")

# シグネチャの内容（自動解析）
# 入力フィールド:
#   - question: Field(description="質問内容")
# 出力フィールド:
#   - answer: Field(description="回答内容")
```

**シグネチャの表現形式:**
- `"input1,input2 -> output1,output2"`
- 入力と出力は`->`で分離
- 複数フィールドはカンマ区切り

### 2. プロンプトテンプレートの生成

DSPyは自動的に以下の構造でプロンプトを生成します：

```
[システムプロンプト]
You are an expert at answering questions in a clear, 
concise, and informative manner. Follow the provided 
instructions and format your responses according to 
the requirements.

[タスク定義]
Given the fields `question`, produce the fields `answer`.

[形式指定]
Follow the following format:

Question: ${question}
Answer: ${answer}

[実際の入力]
Question: Pythonとは何ですか？
Answer:
```

### 3. ChainOfThoughtの場合

```python
# 思考過程を含めたい場合
cot = dspy.ChainOfThought("question -> answer")

# 内部的には以下のフィールドを持つ：
# - question: 入力質問
# - reasoning: LLMの思考過程（自動追加）
# - answer: 最終回答

# プロンプトテンプレートは自動的に以下になる：
# Question: ...
# Reasoning: (LLMに思考させるプロンプト)
# Answer: (最終回答)
```

---

## 🔗 複数ステップのパイプライン

### パイプラインの構造

```python
class TextSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_keywords = dspy.Predict("text -> keywords")
        self.generate_summary = dspy.Predict("keywords -> summary")
    
    def forward(self, text):
        # Step 1
        step1 = self.extract_keywords(text=text)
        # Step 2
        step2 = self.generate_summary(keywords=step1.keywords)
        return step2
```

### 実行フロー

```
入力テキスト
    ↓
┌─────────────────────────────┐
│ Step 1: キーワード抽出       │
│ prompt="text -> keywords"   │
│ [API呼び出し 1回目]         │
│ Output: "keyword1, kw2,..." │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Step 2: 要約作成            │
│ prompt="keywords -> summary"│
│ [API呼び出し 2回目]         │
│ Output: "要約文..."         │
└─────────────────────────────┘
    ↓
最終出力
```

**重要:** 各ステップは独立したAPI呼び出しです。

---

## 🌐 Azure OpenAI APIレベルでの流れ

### HTTPリクエスト

```
POST https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview

Headers:
  api-key: ${AZURE_OPENAI_API_KEY}
  Content-Type: application/json

Body:
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert at answering questions..."
    },
    {
      "role": "user",
      "content": "[完全なプロンプト]\nQuestion: Pythonとは？\nAnswer:"
    }
  ],
  "temperature": 1,
  "max_tokens": 200,
  "top_p": 1
}
```

### HTTPレスポンス

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Python はオブジェクト指向のプログラミング言語です..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 150,
    "total_tokens": 235
  }
}
```

---

## 🔍 プロンプト検査とデバッグ

### 実行履歴の確認

```python
# ロギングを有効化
dspy.enable_logging()

# プログラム実行
result = predict(question="質問")

# 実行履歴を取得（特定のフレームワークで）
history = dspy.inspect_history()
```

### トレースの活用

```python
# 実行を追跡
with dspy.trace():
    result = predict(question="質問")

# 追跡情報から：
# - 入力値の確認
# - 生成されたプロンプト
# - LMからの応答
# - 処理時間
# などが取得可能
```

---

## ⚡ 最適化とキャッシング

### キャッシング機構

```
1回目実行:
  Input: "Pythonとは？"
  → API呼び出し
  → Result キャッシュ保存
  
2回目実行:
  Input: "Pythonとは？"
  → キャッシュから取得 ✓
  → API呼び出しなし
```

### プロンプト最適化

```python
# 初期プロンプト（シンプル）
program = QA()  # "question -> answer"

# 最適化前のプロンプトテンプレート
# → 生成される回答の質が低い

↓ BootstrapFewShot等で最適化

# 最適化後のプロンプト（改善）
# "You are an expert..."
# Example 1: Q1 → A1
# Example 2: Q2 → A2
# Question: ${input}
# Answer: """
```

---

## 🎯 実行可能なスクリプト

このプロジェクトには以下のスクリプトがあります：

1. **`prompt_flow_explanation.py`**
   - プロンプト実行フローの視覚化
   - 複数ステップの流れ
   - 最適化の概念

2. **`prompt_content_detail.py`**
   - 実際のシグネチャ構造
   - プロンプトテンプレートの詳細
   - API呼び出しの形式

3. **`sample_program.py`**
   - 実際に動作するサンプル
   - 複数ステップのパイプライン例

---

## 📚 参考資料

- **DSPy公式ドキュメント**: https://dspy.ai/
- **GitHub**: https://github.com/stanfordnlp/dspy
- **プロンプト最適化**: https://dspy.ai/docs/building-blocks/optimizers
- **Azure OpenAI API**: https://learn.microsoft.com/en-us/azure/ai-services/openai/

---

## 💡 Key Takeaways

1. **自動プロンプト構築**: DSPyは`Signature`から自動的にプロンプトテンプレートを生成
2. **複数ステップ**: パイプラインの各ステップは独立したAPI呼び出し
3. **構造化出力**: 出力は自動的に解析され、フィールド値として利用可能
4. **最適化可能**: DSPyは組み込み最適化器でプロンプトを改善可能
5. **Azure OpenAI対応**: DSPy 3.0.3では`LM`クラスで統一的に各プロバイダをサポート

---

**作成日**: 2025年10月28日  
**DSPyバージョン**: 3.0.3
