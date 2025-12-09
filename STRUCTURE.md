# HotpotQA Graph-RAG Pipeline - 파일 구조

## 파일 구조

```
hotpot_git/
├── main.py                              # 메인 실행 파일 (main 함수)
├── src/                                 # 핵심 모듈
│   ├── __init__.py
│   ├── config.py                       # 설정 상수 (GPU, 모델 파라미터)
│   ├── models.py                       # GAT 모델 (GATLayer, HierarchicalQueryAwareGATRetriever)
│   ├── losses.py                       # Loss 함수 (ContrastiveLoss, WeightedRankingLoss)
│   ├── cache.py                        # 캐시 구축 (build_hierarchical_graph_cache, build_labels)
│   ├── retriever.py                    # Retriever 학습 및 추론 (train, retrieve_supporting_facts)
│   ├── generator.py                     # Generator 학습 (train_t5_generator, FiD 관련)
│   └── evaluation.py                    # 평가 함수 (evaluate, normalize_answer, compute_answer_em_f1)
├── utils/                               # 유틸리티 모듈
│   ├── __init__.py
│   ├── graph_utils.py                  # 그래프 유틸리티
│   └── gpu_utils.py                    # GPU 자동 선택
├── requirements.txt                    # Python 의존성
├── README.md                           # 통합 문서
└── .gitignore                          # Git 제외 파일
```

## 파일 크기

- **총 Python 코드**: ~3,541 줄 (10개 파일)
- **가장 큰 파일**: generator.py (738줄)
- **가장 작은 파일**: config.py (23줄)

## 모듈 의존성

```
main.py (main)
    ├── src.config
    ├── src.models
    ├── src.cache
    ├── src.losses
    ├── src.retriever
    ├── src.generator
    └── src.evaluation

src.models
    └── src.config

src.cache
    ├── src.config
    ├── utils.graph_utils
    └── utils.gpu_utils

src.retriever
    ├── src.config
    ├── src.models
    ├── src.cache
    ├── src.losses
    └── utils.graph_utils

src.generator
    └── src.config

src.evaluation
    ├── src.config
    └── utils.graph_utils
```

## 실행 방법

```bash
python main.py --samples 100 --retriever_epochs 5 --generator_epochs 2
```

## 주요 변경사항

1. OK: **불필요한 파일 제거**: pipeline.py, query_sentence_pipeline/, 비교 문서 등
2. OK: **MD 파일 통합**: README.md, README_PUBLIC.md, QUICKSTART.md → README.md
3. OK: **Python 모듈화**: 2,498줄 단일 파일 → 10개 모듈로 분리
4. OK: **폴더 구조화**: src/ (핵심 모듈), utils/ (유틸리티)
5. OK: **특수기호 제거**: 모든 파일에서 이모지 제거
