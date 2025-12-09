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


## 실행 방법

```bash
python main.py --samples 100 --retriever_epochs 5 --generator_epochs 2
```
