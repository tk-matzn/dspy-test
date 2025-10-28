# DSPy サンプルプログラム

このプロジェクトは、[DSPy](https://github.com/stanfordnlp/dspy)（Declarative Self-improving Python）の使用例を示すサンプルプログラム集です。

DSPyは言語モデルをプログラミングするためのフレームワークで、プロンプト工学を自動化し、モジュール化されたAIシステムを構築することができます。

## 📋 プロジェクト構成

```
dspy-test/
├── README.md              # このファイル
├── requirements.txt       # 依存関係
├── sample_program.py      # 基本的なサンプル
└── advanced_example.py    # 高度な使用例（RAG等）
```

## 🚀 クイックスタート

### 1. 環境設定

```bash
# 依存関係をインストール
pip install -r requirements.txt
```

### 2. API キーの設定

#### 方法1: `.env` ファイルで管理（推奨）

```bash
# .env.example をコピーして .env を作成
cp .env.example .env

# .env ファイルを編集して APIキーを設定
# または、以下のコマンドで直接設定
```

**Windows PowerShell でテキストエディタで編集:**
```powershell
notepad .env
```

**または、PowerShell で直接設定:**
```powershell
@"
OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
"@ | Out-File -Encoding UTF8 .env
```

#### 方法2: 環境変数で直接設定

**OpenAI APIの場合:**
```bash
# Windows PowerShell の場合
$env:OPENAI_API_KEY="your-api-key-here"
```

**Azure OpenAI APIの場合:**
```bash
# Windows PowerShell の場合
$env:AZURE_OPENAI_API_KEY="your-api-key-here"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```

#### 方法3: ローカルLLMを使う場合

[Ollama](https://ollama.ai/) をインストール後、コードで設定：

```python
dspy.configure(lm=dspy.OllamaLocal(model='mistral'))
```

### 3. サンプルプログラムを実行

```bash
# 基本的なサンプル
python sample_program.py

# 高度な例（RAG等）
python advanced_example.py
```

## 📋 .env ファイル設定

### `.env.example` の使用方法

1. `.env.example` をコピーして `.env` を作成
   ```bash
   cp .env.example .env
   ```

2. `.env` ファイルを編集して、実際の APIキーを設定
   ```bash
   # OpenAI
   OPENAI_API_KEY=sk-...

   # Azure OpenAI
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
   ```

3. プログラム実行時に自動で読み込まれます
   ```bash
   python sample_program.py
   ```

### ⚠️ 重要なセキュリティ注意

- `.env` ファイルを Git にコミットしないでください（`.gitignore` に含まれています）
- `.env.example` には実際のキーを入れないでください
- APIキーは厳重に管理してください

## 📚 サンプルプログラムの説明

### `sample_program.py` - 基本的な使い方

#### 1. ChainOfThought - 思考過程を含む回答
- LLMに「思考過程」を含めることで、より高品質な回答を生成

#### 2. 複数ステップのパイプライン
- 複数のモジュールを組み合わせて、複雑なタスクを実行

#### 3. LLMの設定
- OpenAI、ローカルLLM、その他のプロバイダーをサポート

### `advanced_example.py` - 高度な使い方

#### 1. RAG（Retrieval-Augmented Generation）
- ドキュメント検索と回答生成を組み合わせたパイプライン

#### 2. バリデーション
- 入力値の検証と出力値のチェック

#### 3. マルチターン会話
- 会話履歴を管理して、文脈を踏まえた応答を生成

## 💡 DSPyの主な特徴

| 機能 | 説明 |
|------|------|
| **ChainOfThought** | 思考過程を含めてLLMを構成 |
| **Prediction** | 構造化された出力を生成 |
| **Module** | 複数のLLM呼び出しを組み合わせる |
| **Optimizer** | プロンプトを自動最適化 |
| **Assertion** | LLM出力の制約を定義 |

## 🔧 LLMプロバイダーの選択肢

### OpenAI (推奨)
```python
import dspy
dspy.configure(lm=dspy.OpenAI(
    api_key="your-key",
    model="gpt-3.5-turbo",
    max_tokens=1000
))
```

### Azure OpenAI ⭐ 推奨
```python
import dspy
import os

# 環境変数から設定値を取得
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

dspy.configure(lm=dspy.AzureOpenAI(
    api_key=api_key,
    api_base=api_base,
    api_version="2024-02-15-preview",  # または最新のバージョン
    model="deployment-name"  # Azure上でのモデルのデプロイ名
))
```

**Azure OpenAIの設定に必要な情報:**
- `api_key`: Azure OpenAIのAPIキー
- `api_base`: エンドポイントURL（例：`https://your-resource-name.openai.azure.com/`）
- `api_version`: API バージョン（例：`2024-02-15-preview`）
- `model`: Azureにデプロイしたモデルのデプロイ名

**環境変数の設定例（Windows PowerShell）:**
```powershell
$env:AZURE_OPENAI_API_KEY="your-api-key"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```

### Ollama (ローカル)
```python
dspy.configure(lm=dspy.OllamaLocal(model='mistral'))
```

## 📖 学習リソース

- **公式ドキュメント**: https://dspy.ai/
- **GitHub リポジトリ**: https://github.com/stanfordnlp/dspy
- **研究論文**: https://arxiv.org/abs/2310.03714
- **Discord コミュニティ**: https://discord.gg/XCGy2WDCQB

## 🛠️ カスタマイズ例

### カスタムモジュールの作成

```python
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("text -> sentiment, confidence")
    
    def forward(self, text):
        result = self.analyze(text=text)
        return result
```

## ⚠️ トラブルシューティング

### `ModuleNotFoundError: No module named 'dspy'`

```bash
pip install dspy-ai
```

### `ModuleNotFoundError: No module named 'dotenv'`

```bash
pip install python-dotenv
```

または

```bash
pip install -r requirements.txt
```

### `.env` ファイルが読み込まれていない

1. `.env` ファイルが正しい場所にあるか確認
   ```bash
   # ファイルの確認
   ls -la .env
   ```

2. `.env` ファイルのフォーマットを確認
   ```
   KEY=VALUE  # OK
   KEY = VALUE  # スペースが入るとNG
   ```

3. Python スクリプトの最初に `load_dotenv()` が実行されているか確認

### `Error: API key not provided`

API キーが設定されていません。以下を確認してください：

**1. `.env` ファイルが存在するか**
```bash
# カレントディレクトリから実行していることを確認
cd C:\Users\4000268\work\dspy-test
```

**2. `.env` ファイルの内容**
```bash
cat .env
```

**3. 環境変数が正しく設定されているか**
```powershell
$env:AZURE_OPENAI_API_KEY
$env:OPENAI_API_KEY
```

### Azure OpenAI エラーのよくある原因

| エラー | 原因と対処法 |
|-------|-----------|
| `Invalid deployment name` | デプロイ名が正しくありません。Azureポータルで確認してください |
| `Invalid API version` | API バージョンが古い可能性があります。最新のものを使用してください |
| `Unauthorized` | APIキーが無効か、リソースのアクセス権がありません |
| `Resource not found` | エンドポイントが正しくありません。`AZURE_OPENAI_ENDPOINT` を確認してください |

## 📝 ファイル説明

- **sample_program.py**: 基本的な使い方（ChainOfThought、複数モジュールの組み合わせ）
- **advanced_example.py**: 高度な使い方（RAG、バリデーション、マルチターン会話）
- **requirements.txt**: 依存関係のリスト

## 📄 ライセンス

MITライセンス

---

**作成日**: 2025年10月28日  
**DSPyバージョン**: 1.0以上推奨  
**Python バージョン**: 3.8以上推奨
