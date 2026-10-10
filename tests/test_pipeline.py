from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from src.rag_pipeline import RAGPipeline, NO_EVIDENCE


def manager(matches=True):
    result = Mock()
    result.similarity_search.return_value = [(Document(page_content='运单包含货物信息。', metadata={'source': 'demo.txt', 'chunk_id': 0}), 0.8)] if matches else []
    return result


def test_query_retrieves_once_and_keeps_full_source():
    store = manager()
    pipeline = RAGPipeline(store, llm=FakeListLLM(responses=['货物信息 [1]']))
    result = pipeline.query('运单包含什么？')
    store.similarity_search.assert_called_once_with('运单包含什么？')
    assert result['sources'][0]['content'] == '运单包含货物信息。'
    assert result['sources'][0]['chunk_id'] == 0
    assert '[1]' in result['answer']


def test_stream_retrieves_once_and_emits_sources_before_tokens():
    store = manager()
    pipeline = RAGPipeline(store, llm=FakeListLLM(responses=['中文\n回答']))
    events = list(pipeline.stream_events('运单'))
    store.similarity_search.assert_called_once()
    assert events[0]['type'] == 'sources'
    assert events[-1]['type'] == 'done'
    assert ''.join(e['content'] for e in events if e['type'] == 'token') == '中文\n回答'


@pytest.mark.parametrize('streaming', [False, True])
def test_no_evidence_skips_generation(streaming):
    pipeline = RAGPipeline(manager(False), llm=FakeListLLM(responses=['not used']))
    pipeline.chain = Mock()
    if streaming:
        assert list(pipeline.stream_events('unknown'))[1]['content'] == NO_EVIDENCE
    else:
        assert pipeline.query('unknown')['answer'] == NO_EVIDENCE
    pipeline.chain.invoke.assert_not_called()
    pipeline.chain.stream.assert_not_called()


def test_context_is_the_returned_excerpt():
    pipeline = RAGPipeline(manager(), llm=FakeListLLM(responses=['answer']))
    pipeline.chain = Mock()
    pipeline.chain.invoke.return_value = 'answer'
    result = pipeline.query('运单')
    inputs = pipeline.chain.invoke.call_args.args[0]
    assert inputs['context'] == '[1] demo.txt\n' + result['sources'][0]['content']
