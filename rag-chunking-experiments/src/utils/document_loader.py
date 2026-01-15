"""
Document Loader

다양한 형식의 문서를 로드하는 유틸리티
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class DocumentLoader:
    """
    문서 로더

    txt, md, json 등 다양한 형식의 문서를 로드합니다.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Args:
            data_dir: 문서 디렉토리 경로
        """
        self.data_dir = Path(data_dir)

    def load_all(self, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        디렉토리의 모든 문서 로드

        Args:
            extensions: 로드할 파일 확장자 리스트 (기본: [".txt", ".md"])

        Returns:
            문서 리스트 [{"id": str, "text": str, "metadata": dict}, ...]
        """
        extensions = extensions or [".txt", ".md"]
        documents = []

        if not self.data_dir.exists():
            print(f"Warning: Directory not found: {self.data_dir}")
            return documents

        for ext in extensions:
            for file_path in self.data_dir.glob(f"*{ext}"):
                try:
                    doc = self.load_file(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(documents)} documents from {self.data_dir}")
        return documents

    def load_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        단일 파일 로드

        Args:
            file_path: 파일 경로

        Returns:
            문서 딕셔너리
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        ext = file_path.suffix.lower()

        if ext == ".json":
            return self._load_json(file_path)
        elif ext in [".txt", ".md"]:
            return self._load_text(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return None

    def _load_text(self, file_path: Path) -> Dict[str, Any]:
        """텍스트/마크다운 파일 로드"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "id": file_path.stem,
            "text": text,
            "metadata": {
                "source": str(file_path),
                "file_type": file_path.suffix,
                "file_name": file_path.name,
            }
        }

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """JSON 파일 로드"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 단일 문서 또는 문서 리스트 처리
        if isinstance(data, list):
            # 첫 번째 문서만 반환 (리스트의 경우)
            if data:
                return data[0]
            return None
        else:
            return {
                "id": data.get("id", file_path.stem),
                "text": data.get("text", data.get("content", "")),
                "metadata": data.get("metadata", {}),
            }

    def load_processed(self, file_path: str = "data/processed/cleaned_documents.json") -> List[Dict[str, Any]]:
        """
        전처리된 문서 로드

        Args:
            file_path: JSON 파일 경로

        Returns:
            문서 리스트
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"Processed file not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        print(f"Loaded {len(documents)} processed documents")
        return documents

    def save_processed(
        self,
        documents: List[Dict[str, Any]],
        file_path: str = "data/processed/cleaned_documents.json"
    ):
        """
        전처리된 문서 저장

        Args:
            documents: 문서 리스트
            file_path: 저장 경로
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(documents)} documents to {file_path}")

    def get_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        문서 통계 계산

        Args:
            documents: 문서 리스트

        Returns:
            통계 딕셔너리
        """
        if not documents:
            return {"total_documents": 0}

        word_counts = []
        char_counts = []

        for doc in documents:
            text = doc.get("text", "")
            word_counts.append(len(text.split()))
            char_counts.append(len(text))

        return {
            "total_documents": len(documents),
            "total_words": sum(word_counts),
            "total_chars": sum(char_counts),
            "avg_words_per_doc": sum(word_counts) / len(documents),
            "avg_chars_per_doc": sum(char_counts) / len(documents),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }

    def load_test_queries(
        self,
        file_path: str = "data/evaluation/test_queries.json"
    ) -> List[Dict[str, Any]]:
        """
        테스트 쿼리 로드

        Args:
            file_path: 쿼리 파일 경로

        Returns:
            쿼리 리스트
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"Test queries file not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

        print(f"Loaded {len(queries)} test queries")
        return queries
