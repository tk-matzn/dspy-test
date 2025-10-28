"""
DSPyの高度な使用例
RAG（Retrieval-Augmented Generation）パイプラインのサンプル
"""

import dspy
from typing import List
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


# 1. ドキュメント検索シミュレーション
class DocumentRetriever(dspy.Module):
    """ドキュメント検索モジュール（シミュレーション）"""
    
    def __init__(self, documents: List[str]):
        super().__init__()
        self.documents = documents
    
    def forward(self, query: str, k: int = 3) -> List[str]:
        """
        クエリに関連したドキュメントを取得
        実際の実装では、実連携やベクトル検索を使用
        """
        # シンプルなキーワードマッチング
        relevant_docs = []
        query_words = set(query.lower().split())
        
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            # 共通の単語数でスコアリング
            score = len(query_words & doc_words)
            if score > 0:
                relevant_docs.append((score, doc))
        
        # スコアでソートして上位k件を返す
        relevant_docs.sort(reverse=True)
        return [doc for _, doc in relevant_docs[:k]]


# 2. RAGパイプライン
class RAGPipeline(dspy.Module):
    """
    Retrieval-Augmented Generation パイプライン
    質問に基づいてドキュメントを検索し、回答を生成
    """
    
    def __init__(self, retriever: DocumentRetriever):
        super().__init__()
        self.retriever = retriever
        # 検索結果に基づいて回答を生成
        self.answer_generator = dspy.ChainOfThought(
            "context, question -> answer"
        )
    
    def forward(self, question: str):
        # Step 1: 関連ドキュメントを検索
        context_docs = self.retriever(question)
        context = "\n".join(context_docs)
        
        # Step 2: コンテキストを使用して回答を生成
        result = self.answer_generator(
            context=context,
            question=question
        )
        
        return dspy.Prediction(
            question=question,
            context=context,
            answer=result.answer
        )


# 3. エラーチェック付きのモジュール
class ValidatedAnswering(dspy.Module):
    """入力検証と出力チェックを行うモジュール"""
    
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str):
        # 入力検証
        if not question or len(question.strip()) == 0:
            raise ValueError("質問が空です")
        
        if len(question) > 1000:
            raise ValueError("質問が長すぎます（最大1000文字）")
        
        # 回答を生成
        result = self.qa(question=question)
        
        # 出力検証
        if not result.answer or len(result.answer.strip()) == 0:
            raise ValueError("回答を生成できませんでした")
        
        return result


# 4. マルチターン会話のモジュール
class ConversationManager(dspy.Module):
    """マルチターン会話を管理するモジュール"""
    
    def __init__(self):
        super().__init__()
        self.conversation_step = dspy.ChainOfThought(
            "history, new_question -> response"
        )
        self.history: List[dict] = []
    
    def add_turn(self, question: str, answer: str):
        """会話ターンを履歴に追加"""
        self.history.append({
            "question": question,
            "answer": answer
        })
    
    def forward(self, new_question: str):
        """新しい質問に対して、履歴を考慮した応答を生成"""
        # 履歴をテキスト形式にフォーマット
        history_text = "\n".join([
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in self.history[-3:]  # 直近3ターンのみ使用
        ])
        
        result = self.conversation_step(
            history=history_text if history_text else "（会話開始）",
            new_question=new_question
        )
        
        self.add_turn(new_question, result.response)
        return result


# 5. デモデータ
def get_sample_documents():
    """サンプルドキュメントを取得"""
    return [
        "Python は解釈型のプログラミング言語です。シンプルな構文で知られています。",
        "機械学習は、データからパターンを学習するコンピュータサイエンスの分野です。",
        "DSPy は言語モデルをプログラミングするフレームワークです。プロンプト工学を自動化します。",
        "RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。",
        "ベクトル検索は、埋め込みベクトル空間で類似のドキュメントを見つけます。",
    ]


# 6. RAGパイプラインのデモ
def rag_demo():
    """RAGパイプラインのデモンストレーション"""
    print("=" * 60)
    print("RAG パイプライン デモ")
    print("=" * 60)
    
    # ドキュメントとリトリーバーの作成
    documents = get_sample_documents()
    retriever = DocumentRetriever(documents)
    
    # RAGパイプラインの作成
    rag = RAGPipeline(retriever)
    
    # テスト質問
    questions = [
        "DSPyについて教えてください",
        "機械学習とは何ですか？",
        "RAGの仕組みについて説明してください"
    ]
    
    for question in questions:
        print(f"\n質問: {question}")
        print("-" * 40)
        try:
            result = rag(question=question)
            print(f"検索されたコンテキスト:\n{result.context}\n")
            print(f"回答: {result.answer}")
        except Exception as e:
            print(f"エラー: {str(e)}")
            print("（OpenAI API キーが必要です）")


# 7. エラーチェックのデモ
def validation_demo():
    """入力検証と出力チェックのデモ"""
    print("\n" + "=" * 60)
    print("バリデーション デモ")
    print("=" * 60)
    
    validator = ValidatedAnswering()
    
    # 有効な質問
    valid_question = "Pythonの特徴は何ですか？"
    print(f"\n✓ 有効な質問: {valid_question}")
    try:
        result = validator(question=valid_question)
        print(f"  結果: 正常に処理されました")
    except Exception as e:
        print(f"  エラー: {str(e)}")
    
    # 無効な質問（空文字列）
    print(f"\n✗ 無効な質問: （空文字列）")
    try:
        result = validator(question="")
        print(f"  結果: {result}")
    except ValueError as e:
        print(f"  エラー: {str(e)}")
    
    # 無効な質問（長すぎる）
    long_question = "a" * 1001
    print(f"\n✗ 無効な質問: （1001文字）")
    try:
        result = validator(question=long_question)
        print(f"  結果: {result}")
    except ValueError as e:
        print(f"  エラー: {str(e)}")


# 8. マルチターン会話のデモ
def conversation_demo():
    """マルチターン会話のデモンストレーション"""
    print("\n" + "=" * 60)
    print("マルチターン会話 デモ")
    print("=" * 60)
    
    manager = ConversationManager()
    
    questions = [
        "Pythonについて教えてください",
        "Pythonの利点は何ですか？",
        "機械学習に使えますか？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- ターン {i} ---")
        print(f"質問: {question}")
        try:
            result = manager(new_question=question)
            print(f"応答: {result.response}")
        except Exception as e:
            print(f"エラー: {str(e)}")
            print("（OpenAI API キーが必要です）")


if __name__ == "__main__":
    import os
    
    print("DSPy 高度なサンプルプログラムへようこそ！\n")
    
    # LLMの設定（優先順位: Azure OpenAI > OpenAI）
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if azure_key and azure_endpoint:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
        dspy.configure(lm=dspy.LM(
            model=f"azure/{deployment_name}",
            api_key=azure_key,
            api_base=azure_endpoint,
            api_version=api_version,
            max_tokens=1000
        ))
        print("✓ Azure OpenAI が設定されました\n")
        
        # 各デモを実行
        rag_demo()
        # validation_demo()  # これは実装例なので、OpenAI APIがあっても完全には動作しない場合がある
        # conversation_demo()  # マルチターン会話のデモ
    elif openai_key:
        dspy.configure(lm=dspy.LM(
            model="openai/gpt-3.5-turbo",
            api_key=openai_key,
            max_tokens=1000
        ))
        print("✓ OpenAI API が設定されました\n")
        
        # 各デモを実行
        rag_demo()
        # validation_demo()  # これは実装例なので、OpenAI APIがあっても完全には動作しない場合がある
        # conversation_demo()  # マルチターン会話のデモ
    else:
        print("✗ APIキーが設定されていません\n")
        print("以下のいずれかを設定してください：")
        print("  1. Azure OpenAI:")
        print("     - AZURE_OPENAI_API_KEY")
        print("     - AZURE_OPENAI_ENDPOINT")
        print("  2. OpenAI:")
        print("     - OPENAI_API_KEY\n")
        print("使用例：")
        print("  - sample_program.py: 基本的な使い方")
        print("  - advanced_example.py: 高度な使い方（このファイル）")
