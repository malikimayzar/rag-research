import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, batch_size=None, show_progress_bar=False):
        return np.array([[0.0] * 384 for _ in texts])

class DummyRustChunker:
    @staticmethod
    def split_sentences_rs(text):
        return [s.strip() for s in text.split('.') if s.strip()]

    @staticmethod
    def find_breakpoints_rs(similarities, threshold):
        return [i for i, sim in enumerate(similarities) if sim < threshold]

    @staticmethod
    def assemble_chunks_rs(sentences, breakpoints, overlap):
        if not breakpoints:
            return [sentences]
        groups = []
        start = 0
        for bp in breakpoints:
            groups.append(sentences[start:bp + 1])
            start = bp + 1
        if start < len(sentences):
            groups.append(sentences[start:])
        return groups

with mock.patch.dict(sys.modules, {
    'sentence_transformers': mock.Mock(SentenceTransformer=DummySentenceTransformer),
    'semantic_chunker_rust': DummyRustChunker,
}):
    import importlib
    semantic_chunker = importlib.import_module('src.ingestion.semantic_chunker')


def test_semantic_chunker_creates_chunks():
    chunker = semantic_chunker.SemanticChunker()
    doc = {
        'doc_id': 'doc1',
        'text': 'Abstract. This is the first sentence. This is the second sentence. ' \
                'This is the third sentence. This is the fourth sentence. ' \
                'This is the fifth sentence. This is the sixth sentence. ' \
                'References. [1] Paper citation. Another reference sentence.',
        'metadata': {'source': 'test'}
    }

    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(hasattr(chunk, 'chunk_id') for chunk in chunks)
    assert all(chunk.metadata.get('section') in {'ABSTRACT', 'REFERENCES', 'Abstract', 'References'} or chunk.metadata.get('section') for chunk in chunks)

if __name__ == '__main__':
    test_semantic_chunker_creates_chunks()
    print('OK')