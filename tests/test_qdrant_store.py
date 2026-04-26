import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib

# Dummy registry + embedder for safe module import
class DummyEmbedder:
    def encode(self, texts, batch_size=None, show_progress_bar=False):
        return [[0.1] * 384 for _ in texts]

class DummyModelRegistry:
    @staticmethod
    def get():
        return mock.Mock(embedder=DummyEmbedder())

class DummyClient:
    def __init__(self, *args, **kwargs):
        self._collections = []
        self.created = False
        self.upserts = []

    def get_collections(self):
        return mock.Mock(collections=self._collections)

    def create_collection(self, *args, **kwargs):
        self.created = True

    def count(self, collection_name):
        return mock.Mock(count=0)

    def upsert(self, collection_name, points):
        self.upserts.extend(points)

    def query_points(self, collection_name, query, limit, query_filter=None):
        point = mock.Mock()
        point.payload = {
            'chunk_id': 'c1',
            'doc_id': 'doc1',
            'text': 'dummy',
        }
        point.score = 0.85
        return mock.Mock(points=[point])

    def close(self):
        pass

qdrant_client_module = types.ModuleType('qdrant_client')
qdrant_client_module.QdrantClient = DummyClient

qdrant_models_module = types.ModuleType('qdrant_client.models')
qdrant_models_module.Distance = mock.Mock()
qdrant_models_module.VectorParams = mock.Mock()
qdrant_models_module.PointStruct = mock.Mock()
qdrant_models_module.Filter = mock.Mock()
qdrant_models_module.FieldCondition = mock.Mock()
qdrant_models_module.MatchValue = mock.Mock()

model_registry_module = types.ModuleType('src.retrieval.model_registry')
model_registry_module.ModelRegistry = DummyModelRegistry

with mock.patch.dict(sys.modules, {
    'qdrant_client': qdrant_client_module,
    'qdrant_client.models': qdrant_models_module,
    'src.retrieval.model_registry': model_registry_module,
}):
    if 'src.retrieval.qdrant_store' in sys.modules:
        del sys.modules['src.retrieval.qdrant_store']
    qdrant_store = importlib.import_module('src.retrieval.qdrant_store')


def test_qdrant_store_env_overrides():
    os.environ['QDRANT_PATH'] = 'tmp/qdrant_test'
    os.environ['QDRANT_COLLECTION'] = 'test_collection'
    os.environ.pop('QDRANT_URL', None)
    os.environ.pop('QDRANT_API_KEY', None)

    store = qdrant_store.QdrantVectorStore()
    assert store.collection_name == 'test_collection'
    assert store.vector_dim == 384
    store.close()


def test_qdrant_store_search_returns_results():
    os.environ['QDRANT_PATH'] = 'tmp/qdrant_test'
    os.environ['QDRANT_COLLECTION'] = 'test_collection'
    store = qdrant_store.QdrantVectorStore()
    results = store.search('hello world', k=1)
    assert len(results) == 1
    assert results[0].chunk_id == 'c1'
    assert results[0].score == 0.85
    store.close()

if __name__ == '__main__':
    test_qdrant_store_env_overrides()
    test_qdrant_store_search_returns_results()
    print('OK')