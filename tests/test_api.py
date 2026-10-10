import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from src.web_api import create_app


@pytest.fixture
def api():
    pipeline = Mock()
    pipeline.query.return_value = {'answer': '回答', 'sources': []}
    pipeline.vector_store_manager.get_collection_info.return_value = {'document_count': 1}
    app = create_app(pipeline=pipeline)
    with TestClient(app) as client:
        yield app, client, pipeline


@pytest.mark.parametrize('body', [{'question': ''}, {'question': '   '}, {'question': 'a' * 1001}, {'question': 'ok', 'secret': 'bad'}])
def test_invalid_request(api, body):
    assert api[1].post('/api/query', json=body).status_code == 422
    api[2].query.assert_not_called()


def test_query_and_health(api):
    app, client, pipeline = api
    assert client.get('/api/health').status_code == 200
    assert client.get('/').status_code == 200
    assert client.get('/api/status').json()['conversation_memory'] is False
    assert client.post('/api/query', json={'question': ' q '}).status_code == 200
    pipeline.query.assert_called_once_with('q')
    assert client.post('/api/query', json={'question': 'q', 'stream': True}).status_code == 400


def test_sse_preserves_newlines(api):
    _, client, pipeline = api
    events = [{'type': 'sources', 'sources': []}, {'type': 'token', 'content': '中文\n换行'}, {'type': 'done'}]
    pipeline.stream_events.return_value = iter(events)
    response = client.post('/api/query/stream', json={'question': 'q'})
    parsed = [json.loads(frame.removeprefix('data: ')) for frame in response.text.strip().split('\n\n')]
    assert parsed == events
    pipeline.stream_events.assert_called_once_with('q')


def test_errors_do_not_expose_internal_details(api):
    _, client, pipeline = api
    pipeline.query.side_effect = RuntimeError('PRIVATE_PATH_OR_SECRET')
    response = client.post('/api/query', json={'question': 'q'})
    assert response.status_code == 502
    assert 'PRIVATE_PATH_OR_SECRET' not in response.text
    pipeline.stream_events.side_effect = RuntimeError('PRIVATE_PATH_OR_SECRET')
    response = client.post('/api/query/stream', json={'question': 'q'})
    assert '"type":"error"' in response.text
    assert 'PRIVATE_PATH_OR_SECRET' not in response.text


def test_refresh_status_and_conflict(api):
    app, client, pipeline = api
    app.state.processor = Mock()
    app.state.processor.process_documents.return_value = ['doc']
    app.state.refresh_lock.acquire()
    assert client.post('/api/knowledge-base/refresh').status_code == 409
    app.state.refresh_lock.release()
    assert client.post('/api/knowledge-base/refresh').status_code == 202
    assert app.state.refresh['status'] == 'completed'
    pipeline.vector_store_manager.create_vectorstore.assert_called_once_with(['doc'])
    app.state.processor.process_documents.side_effect = ValueError('broken input')
    assert client.post('/api/knowledge-base/refresh').status_code == 202
    assert app.state.refresh['status'] == 'failed'
    assert not app.state.refresh_lock.locked()


def test_not_initialized_is_503():
    with TestClient(create_app(initialize=False)) as client:
        assert client.get('/api/health').status_code == 503
