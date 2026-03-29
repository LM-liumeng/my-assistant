import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.orchestrator import IntentRecognizer
from context.evidence_store import EvidenceStore
from security import SafetyLayer
from tools.rag_management_tool import RAGManagementTool


def test_intent_recognizes_rag_status_command():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise('rag status')
    assert intent == 'rag_manage'
    assert params['action'] == 'status'


def test_intent_recognizes_rag_rebuild_command():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise(r'rebuild rag D:\knowledge')
    assert intent == 'rag_manage'
    assert params['action'] == 'rebuild'
    assert params['input_dir'] == r'D:\knowledge'


def test_rag_management_status_reads_manifest(tmp_path: Path):
    data_dir = tmp_path / 'data'
    rag_dir = data_dir / 'rag'
    rag_dir.mkdir(parents=True)
    manifest = {
        'generated_at': '2026-03-07T00:00:00+00:00',
        'embedding_model': 'BAAI/bge-m3',
        'documents': [{'doc_id': 'doc_1'}],
        'chunk_count': 3,
        'collection_name': 'rag_chunks',
    }
    (rag_dir / 'ingestion_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (rag_dir / 'chunks.jsonl').write_text('{}\n', encoding='utf-8')
    (rag_dir / 'embeddings.npy').write_bytes(b'fake')

    tool = RAGManagementTool(str(data_dir), EvidenceStore(str(data_dir)), SafetyLayer(EvidenceStore(str(data_dir))))
    result = tool.status()
    assert result['display_filename'] == 'rag_status.txt'
    assert 'Indexed documents: 1' in result['display_content']
    assert 'Chunk count: 3' in result['display_content']


def test_rag_management_ingest_uses_project_root_rag_pipeline(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / 'data'
    knowledge_dir = data_dir / 'knowledge'
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / 'note.txt').write_text('hello', encoding='utf-8')
    pipeline_dir = project_root / 'RAG'
    pipeline_dir.mkdir()
    (pipeline_dir / 'rag_multiformat_ingestion_pipeline.py').write_text('print("stub")', encoding='utf-8')
    manifest = {
        'generated_at': '2026-03-07T00:00:00+00:00',
        'embedding_model': 'hashing-fallback:256',
        'documents': [{'doc_id': 'doc_1'}],
        'chunk_count': 1,
        'collection_name': '',
    }

    tool = RAGManagementTool(str(data_dir), EvidenceStore(str(data_dir)), SafetyLayer(EvidenceStore(str(data_dir))))

    def fake_execute(kb, rebuild):
        output_dir = Path(kb['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'ingestion_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        (output_dir / 'chunks.jsonl').write_text('{}\n', encoding='utf-8')
        (output_dir / 'embeddings.npy').write_bytes(b'fake')
        return 0, 'pipeline ok', 'hashing-fallback:256'

    monkeypatch.setattr(tool, '_execute_pipeline', fake_execute)

    result = tool.run_ingestion(input_dir=str(knowledge_dir), rebuild=False)
    assert result['message'] == "RAG incremental ingestion completed for knowledge base 'default'."
    assert 'Effective embedding model: hashing-fallback:256' in result['display_content']
    assert str(project_root / 'RAG' / 'rag_multiformat_ingestion_pipeline.py') == str(tool.pipeline_script)


def test_rag_management_async_job_reports_completion(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / 'data'
    knowledge_dir = data_dir / 'knowledge'
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / 'note.txt').write_text('hello', encoding='utf-8')
    pipeline_dir = tmp_path / 'RAG'
    pipeline_dir.mkdir()
    (pipeline_dir / 'rag_multiformat_ingestion_pipeline.py').write_text('print("stub")', encoding='utf-8')

    tool = RAGManagementTool(str(data_dir), EvidenceStore(str(data_dir)), SafetyLayer(EvidenceStore(str(data_dir))))

    def fake_run_ingestion(**kwargs):
        return {
            'message': "RAG incremental ingestion completed for knowledge base 'default'.",
            'knowledge_base': tool.registry.get_base('default'),
            'active_knowledge_base': 'default',
            'display_content': 'done',
            'display_filename': 'rag_ingestion_report.txt',
        }

    monkeypatch.setattr(tool, 'run_ingestion', fake_run_ingestion)
    started = tool.start_ingestion(knowledge_base='default', input_dir=str(knowledge_dir), rebuild=False)
    assert started['status'] == 'queued'

    job = None
    for _ in range(20):
        job = tool.get_job(started['job_id'])
        if job['status'] == 'completed':
            break
        time.sleep(0.05)
    assert job is not None
    assert job['status'] == 'completed'
    assert job['result']['display_filename'] == 'rag_ingestion_report.txt'


def test_rag_management_can_save_and_select_multiple_knowledge_bases(tmp_path: Path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    tool = RAGManagementTool(str(data_dir), EvidenceStore(str(data_dir)), SafetyLayer(EvidenceStore(str(data_dir))))

    saved = tool.save_knowledge_base(
        knowledge_base='product',
        knowledge_base_name='产品知识库',
        input_dir=str(data_dir / 'knowledge' / 'product'),
        description='产品文档与FAQ',
    )
    assert saved['knowledge_base']['id'] == 'product'

    selected = tool.select_knowledge_base('product')
    assert selected['active_knowledge_base'] == 'product'

    listing = tool.list_knowledge_bases()
    assert any(item['id'] == 'product' for item in listing['knowledge_bases'])
