"""Tiny keyword-based retrieval regression check; not an answer accuracy benchmark."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

CASES = [
    ('运单需要写哪些信息？', ['货物重量和尺寸']),
    ('ABC分类法如何管理库存？', ['A类物品']),
    ('经济订货批量的计算公式是什么？', ['2DS/H']),
    ('仓储出库包含什么流程？', ['拣选、复核、包装、发货']),
    ('WMS和TMS分别是什么？', ['TMS：运输管理系统', 'WMS：仓库管理系统']),
    ('绿色物流可以如何减少包装消耗？', ['包装减量']),
    ('选择第三方物流时看哪些条件？', ['3PL选择标准']),
    ('空运的体积重量怎么算？', ['6000']),
]

def main():
    # Dedicated evaluation collection/directory never overwrites the application index.
    options = config.model_copy(deep=True)
    options.vector_store.persist_directory = str(ROOT / 'chroma_db' / 'evaluation')
    options.vector_store.collection_name = 'retrieval_evaluation'
    manager = VectorStoreManager(options)
    chunks = DocumentProcessor(options).process_documents()
    manager.create_vectorstore(chunks)
    report = {'dataset': '8 bundled educational questions; keyword coverage, not correctness',
              'chunks': len(chunks), 'embedding_model': options.model.embedding, 'results': []}
    for k in [1, 3, 5]:
        options.vector_store.k = k
        cases = []
        for question, expected in CASES:
            matches = manager.similarity_search(question)
            context = '\n'.join(doc.page_content for doc, _ in matches)
            hit = all(phrase in context for phrase in expected)
            cases.append({'question': question, 'expected_keywords_found': hit,
                          'chunk_ids': [d.metadata['chunk_id'] for d, _ in matches]})
        report['results'].append({'k': k, 'hits': sum(c['expected_keywords_found'] for c in cases),
                                  'total': len(cases), 'cases': cases})
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
