#!/usr/bin/env python
"""
RAG Chunking Experiments - 자동 실험 실행기

Usage:
    python run_experiments.py --all              # 모든 전략 실행
    python run_experiments.py --strategy fixed   # 특정 전략만 실행
    python run_experiments.py --evaluate         # 평가만 실행
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

from src.chunkers import get_chunker
from src.embedders import BGEEmbedder
from src.retrievers import VectorRetriever
from src.evaluators import RetrievalEvaluator
from src.utils.document_loader import DocumentLoader


def run_chunking_experiment(
    strategy: str,
    documents: List[Dict[str, Any]],
    config: dict,
    test_queries: List[Dict[str, Any]],
    output_dir: Path,
    embedder: BGEEmbedder
) -> Dict[str, Any]:
    """단일 청킹 전략 실험 실행"""

    print(f"\n{'='*60}")
    print(f"{strategy.upper()} Chunking 실험")
    print(f"{'='*60}")

    # 1. 청킹
    print("\n[1/4] 청킹 실행...")
    chunker = get_chunker(strategy, config['chunking'])

    start_time = time.time()
    chunks = chunker.chunk_documents(documents, verbose=True)
    chunking_time = time.time() - start_time

    print(f"총 {len(chunks)}개 청크 생성 (소요 시간: {chunking_time:.2f}초)")

    # 2. 통계
    stats = chunker.get_statistics(chunks)
    stats['chunking_time'] = chunking_time

    # 3. 임베딩 및 인덱싱
    print("\n[2/4] 임베딩 생성...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    print("\n[3/4] 벡터 인덱싱...")
    retriever = VectorRetriever(embedder)
    retriever.add_chunks(chunks, embeddings)

    # 4. 검색
    print("\n[4/4] 테스트 쿼리 검색...")
    retrieval_results = {}

    for query_data in test_queries:
        query_id = query_data['query_id']
        query_text = query_data['query']

        results = retriever.search(query_text, k=5)
        retrieval_results[query_id] = [r.chunk_id for r in results]

    print(f"총 {len(retrieval_results)}개 쿼리 검색 완료")

    # 5. 결과 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(output_dir / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'retrieval_results.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {output_dir}")

    return {
        'strategy': strategy,
        'stats': stats,
        'retrieval_results': retrieval_results
    }


def run_evaluation(results_dir: Path, test_queries: List[Dict[str, Any]]):
    """평가 실행"""

    print(f"\n{'='*60}")
    print("통합 평가")
    print(f"{'='*60}")

    strategies = ['fixed', 'recursive', 'semantic']
    all_stats = {}

    for strategy in strategies:
        stats_path = results_dir / f'{strategy}_chunking' / 'stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                all_stats[strategy] = json.load(f)

    if not all_stats:
        print("평가할 결과가 없습니다.")
        return

    # 비교 테이블 출력
    print("\n청킹 전략 비교:")
    print("-" * 70)
    print(f"{'Strategy':<12} {'Chunks':<10} {'Avg Size':<12} {'Std':<10} {'Time(s)':<10}")
    print("-" * 70)

    for strategy, stats in all_stats.items():
        print(f"{strategy.capitalize():<12} "
              f"{stats.get('total_chunks', 0):<10} "
              f"{stats.get('avg_size', 0):<12.1f} "
              f"{stats.get('std_size', 0):<10.1f} "
              f"{stats.get('chunking_time', 0):<10.2f}")

    # 비교 결과 저장
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    with open(comparison_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"\n평가 결과 저장: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description='RAG Chunking Experiments')
    parser.add_argument('--all', action='store_true', help='Run all strategies')
    parser.add_argument('--strategy', type=str, choices=['fixed', 'recursive', 'semantic'],
                       help='Run specific strategy')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')

    args = parser.parse_args()

    # 경로 설정
    base_dir = Path(__file__).parent
    data_dir = base_dir / args.data_dir
    results_dir = base_dir / args.results_dir
    configs_dir = base_dir / 'configs'

    # 데이터 로드
    print("데이터 로드 중...")
    loader = DocumentLoader(str(data_dir / 'raw'))

    # 전처리된 문서 로드 또는 생성
    processed_path = data_dir / 'processed' / 'cleaned_documents.json'
    if processed_path.exists():
        documents = loader.load_processed(str(processed_path))
    else:
        documents = loader.load_all()
        if documents:
            loader.save_processed(documents, str(processed_path))

    if not documents:
        print("문서를 찾을 수 없습니다. data/raw/ 디렉토리를 확인하세요.")
        return

    # 테스트 쿼리 로드
    test_queries = loader.load_test_queries(str(data_dir / 'evaluation' / 'test_queries.json'))

    # 평가만 실행
    if args.evaluate:
        run_evaluation(results_dir, test_queries)
        return

    # 실행할 전략 결정
    if args.all:
        strategies = ['fixed', 'recursive', 'semantic']
    elif args.strategy:
        strategies = [args.strategy]
    else:
        print("--all 또는 --strategy를 지정하세요.")
        parser.print_help()
        return

    # 임베더 초기화 (공유)
    print("\n임베딩 모델 로드 중...")
    embedder = BGEEmbedder()

    # 각 전략 실행
    all_results = []

    for strategy in strategies:
        config_path = configs_dir / f'{strategy}_config.yaml'

        if not config_path.exists():
            print(f"설정 파일 없음: {config_path}")
            continue

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        output_dir = results_dir / f'{strategy}_chunking'

        result = run_chunking_experiment(
            strategy=strategy,
            documents=documents,
            config=config,
            test_queries=test_queries,
            output_dir=output_dir,
            embedder=embedder
        )

        all_results.append(result)

    # 평가 실행
    if len(all_results) > 1:
        run_evaluation(results_dir, test_queries)

    print(f"\n{'='*60}")
    print("실험 완료!")
    print(f"{'='*60}")
    print(f"결과 디렉토리: {results_dir}")


if __name__ == '__main__':
    main()
