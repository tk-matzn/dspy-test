"""
DSPyの基本的なサンプルプログラム
質問に対して回答を生成する簡単なシステムです
"""

import dspy
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


# 3. ローカルLLMを使う場合の設定例（コメント化）
def setup_local_llm():
    """ローカルLLMを使う場合の設定例"""
    # ollama を使う場合:
    # dspy.configure(lm=dspy.OllamaLocal(model='mistral'))
    pass


# 4. OpenAI APIを使う場合の設定
def setup_openai():
    """OpenAI APIを使う場合の設定"""
    import os
    
    # 環境変数から APIキーを取得
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        dspy.configure(lm=dspy.LM(
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
            max_tokens=1000
        ))
        return True
    else:
        print("警告: OPENAI_API_KEY 環境変数が設定されていません")
        return False


# 5. Azure OpenAI APIを使う場合の設定
def setup_azure_openai():
    """Azure OpenAI APIを使う場合の設定"""
    import os
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
    
    if api_key and api_base:
        # DSPyの新しい形式でAzure OpenAIを設定
        dspy.configure(lm=dspy.LM(
            model=f"azure/{deployment_name}",
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_tokens=1000
        ))
        return True
    else:
        print("警告: Azure OpenAI 環境変数が設定されていません")
        print("  必要な環境変数:")
        print("    - AZURE_OPENAI_API_KEY")
        print("    - AZURE_OPENAI_ENDPOINT")
        return False


# 5. デモンストレーション用の簡単な例
def simple_demo():
    """シンプルなデモンストレーション"""
    print("=" * 60)
    print("DSPy シンプルデモ")
    print("=" * 60)
    
    # Predictを使用した簡単な質問応答
    qa = dspy.Predict("question -> answer")
    
    # 質問を実行
    questions = [
        "Pythonとは何ですか？",
        "機械学習の基礎を説明してください。",
        "DSPyの利点は何ですか？"
    ]
    
    for question in questions:
        print(f"\n質問: {question}")
        try:
            # 回答を生成
            result = qa(question=question)
            print(f"回答: {result.answer}")
        except Exception as e:
            print(f"エラー: {str(e)}")
            print("注: Azure OpenAI API キーが有効であることを確認してください")


# 6. より複雑なパイプラインの例
def pipeline_demo():
    """複数ステップのパイプラインデモ"""
    print("\n" + "=" * 60)
    print("DSPy パイプラインデモ")
    print("=" * 60)
    
    # シンプルなパイプライン
    class SimplePipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract_points = dspy.Predict("question -> key_points")
            self.generate_answer = dspy.Predict("key_points -> answer")
        
        def forward(self, question):
            # Step 1: 質問の要点を抽出
            points = self.extract_points(question=question)
            
            # Step 2: 要点から回答を生成
            result = self.generate_answer(key_points=points.key_points)
            
            return dspy.Prediction(
                question=question,
                key_points=points.key_points,
                answer=result.answer
            )
    
    pipeline = SimplePipeline()
    
    # 質問を実行
    question = "日本の首都はどこですか？その都市の特徴を説明してください。"
    
    print(f"\n質問: {question}")
    try:
        result = pipeline(question=question)
        print(f"要点: {result.key_points}")
        print(f"回答: {result.answer}")
    except Exception as e:
        print(f"エラー: {str(e)}")
        print("注: Azure OpenAI API キーが有効であることを確認してください")


# 7. トレーニングデータを使った最適化の例
def optimization_demo():
    """DSPyの最適化機能のデモ"""
    print("\n" + "=" * 60)
    print("DSPy 最適化デモ")
    print("=" * 60)
    
    # トレーニングデータセット（例）
    trainset = [
        dspy.Example(
            question="2+2は何ですか？",
            answer="4です"
        ).with_inputs("question"),
        dspy.Example(
            question="3×5は何ですか？",
            answer="15です"
        ).with_inputs("question"),
    ]
    
    print("\nトレーニングデータセット:")
    for example in trainset:
        print(f"  Q: {example.question}")
        print(f"  A: {example.answer}")
    
    print("\n最適化は実装例です。実際の使用には詳細な設定が必要です。")


if __name__ == "__main__":
    print("DSPy サンプルプログラムへようこそ！\n")
    
    # LLMの設定を試みる
    print("1. LLMの設定中...")
    
    # 優先順位: Azure OpenAI > OpenAI
    if setup_azure_openai():
        print("   ✓ Azure OpenAI が設定されました\n")
        
        # デモ実行
        simple_demo()
        pipeline_demo()
        optimization_demo()
    elif setup_openai():
        print("   ✓ OpenAI APIが設定されました\n")
        
        # デモ実行
        simple_demo()
        pipeline_demo()
        optimization_demo()
    else:
        print("   ✗ LLMが設定されていません\n")
        print("以下の方法でLLMを設定してください：\n")
        print("方法1: Azure OpenAI API")
        print("  - AZURE_OPENAI_API_KEY 環境変数を設定する")
        print("  - AZURE_OPENAI_ENDPOINT 環境変数を設定する\n")
        print("方法2: OpenAI API")
        print("  - OPENAI_API_KEY 環境変数を設定する")
        print("  - または、コード内で API キーを指定する\n")
        print("方法3: ローカルLLM（Ollama）")
        print("  - ollama をインストール")
        print("  - setup_local_llm() を呼び出す\n")
        print("詳細は以下を参照してください：")
        print("  - DSPy ドキュメント: https://dspy.ai/")
        print("  - OpenAI API ドキュメント: https://platform.openai.com/docs/")
        print("  - Azure OpenAI ドキュメント: https://learn.microsoft.com/en-us/azure/ai-services/openai/")
