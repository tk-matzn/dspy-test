"""
DSPy クイックスタート
最もシンプルな例から始めます
"""

import os
import sys
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


def setup_environment():
    """環境をセットアップしてDSPyをインポート"""
    try:
        import dspy
        return True, dspy
    except ImportError:
        print("❌ DSPyがインストールされていません")
        print("\n以下のコマンドを実行してください:")
        print("  pip install dspy-ai")
        return False, None


def check_api_key():
    """OpenAI/Azure OpenAI APIキーを確認"""
    import os
    
    # 優先順位: Azure OpenAI > OpenAI
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if azure_key and azure_endpoint:
        return True, "azure", (azure_key, azure_endpoint)
    elif openai_key:
        return True, "openai", openai_key
    else:
        print("⚠️  APIキーが設定されていません\n")
        return False, None, None


def simple_example(dspy):
    """最もシンプルな例"""
    print("\n" + "="*60)
    print("1️⃣  シンプルな質問応答")
    print("="*60)
    
    # ChainOfThoughtプログラムを作成
    qa = dspy.ChainOfThought("question -> answer")
    
    # 質問を実行
    question = "Python とは何ですか？"
    print(f"\n質問: {question}\n")
    
    try:
        result = qa(question=question)
        print(f"思考過程:\n{result.reasoning}\n")
        print(f"回答:\n{result.answer}")
        return True
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return False


def predict_example(dspy):
    """予測（入出力フィールドを定義）"""
    print("\n" + "="*60)
    print("2️⃣  予測（構造化出力）")
    print("="*60)
    
    # 入出力フィールドを定義したプログラム
    class SimplePrediction(dspy.ChainOfThought):
        input_text = dspy.InputField(desc="入力テキスト")
        output_text = dspy.OutputField(desc="出力テキスト")
    
    program = SimplePrediction()
    
    input_text = "リンゴは赤い果物です"
    print(f"\n入力: {input_text}\n")
    
    try:
        result = program(input_text=input_text)
        print(f"出力: {result.output_text}")
        return True
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return False


def custom_module_example(dspy):
    """カスタムモジュール"""
    print("\n" + "="*60)
    print("3️⃣  カスタムモジュール")
    print("="*60)
    
    # カスタムモジュールを定義
    class Summarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.summarize = dspy.ChainOfThought("text -> summary")
        
        def forward(self, text):
            return self.summarize(text=text)
    
    summarizer = Summarizer()
    
    text = "DSPyはプロンプト工学を自動化するフレームワークです。複雑なAIシステムを簡単に構築できます。"
    print(f"\n入力テキスト:\n{text}\n")
    
    try:
        result = summarizer(text=text)
        print(f"要約:\n{result.summary}")
        return True
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return False


def main():
    """メイン処理"""
    print("""
╔════════════════════════════════════╗
║     DSPy クイックスタート          ║
║   Declarative Self-improving       ║
║            Python                  ║
╚════════════════════════════════════╝
    """)
    
    # Step 1: DSPyをセットアップ
    print("📦 DSPyをセットアップ中...\n")
    success, dspy = setup_environment()
    if not success:
        sys.exit(1)
    
    print("✅ DSPy をインポートしました")
    
    # Step 2: APIキーを確認
    print("\n🔑 API キーを確認中...\n")
    has_key, provider, credentials = check_api_key()
    
    if not has_key:
        print("Azure OpenAI API キーを設定してください:")
        print("  $env:AZURE_OPENAI_API_KEY='your-api-key'")
        print("  $env:AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'\n")
        print("または OpenAI API キーを設定してください:")
        print("  $env:OPENAI_API_KEY='your-api-key'\n")
        print("ローカルLLM（Ollama）を使う場合は sample_program.py を参照してください")
        sys.exit(1)
    
    print(f"✅ {provider.upper()} キーが設定されています")
    
    # Step 3: DSPyを設定
    print("\n⚙️  DSPyを設定中...\n")
    try:
        if provider == "azure":
            api_key, api_base = credentials
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
            dspy.configure(lm=dspy.LM(
                model=f"azure/{deployment_name}",
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                max_tokens=500
            ))
        else:  # openai
            dspy.configure(lm=dspy.LM(
                model="openai/gpt-3.5-turbo",
                api_key=credentials,
                max_tokens=500
            ))
        print("✅ DSPy が設定されました")
    except Exception as e:
        print(f"❌ 設定エラー: {str(e)}")
        sys.exit(1)
    
    # Step 4: 例を実行
    print("\n🚀 サンプルを実行します\n")
    
    try:
        simple_example(dspy)
        predict_example(dspy)
        custom_module_example(dspy)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        sys.exit(1)
    
    # 完了
    print("\n" + "="*60)
    print("✅ すべてのサンプルが正常に実行されました！")
    print("="*60)
    print("\n📚 次のステップ:")
    print("  1. sample_program.py を実行して基本的な使い方を学ぶ")
    print("  2. advanced_example.py で高度な例を見る")
    print("  3. https://dspy.ai/ で公式ドキュメントを読む")
    print("\n🎉 Happy DSPying!\n")


if __name__ == "__main__":
    main()
