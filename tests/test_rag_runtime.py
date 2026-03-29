import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.orchestrator import Agent, IntentRecognizer
from context.knowledge_registry import KnowledgeRegistry
from context.rag_store import HybridRAGStore


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_fake_rag_index(base_dir: Path, kb_id: str = 'default') -> Path:
    registry = KnowledgeRegistry(str(base_dir))
    kb = registry.upsert_base(knowledge_base=kb_id, name=kb_id)
    rag_dir = Path(kb['output_dir'])
    rag_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'generated_at': '2026-03-07T00:00:00+00:00',
        'input_dir': str(base_dir / 'knowledge' / kb_id),
        'output_dir': str(rag_dir),
        'collection_name': f'rag_chunks_{kb_id}',
        'embedding_model': 'fake-model',
        'documents': [
            {
                'doc_id': 'doc_1',
                'source_path': str(base_dir / 'knowledge' / kb_id / 'policy.txt'),
                'source_name': 'policy.txt',
                'checksum': 'abc123',
                'doc_type': 'txt',
                'updated_at': '2026-03-07T00:00:00+00:00',
            }
        ],
        'chunk_count': 2,
    }
    (rag_dir / 'ingestion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    chunks = [
        {
            'chunk_id': 'doc_1_chunk_00000',
            'doc_id': 'doc_1',
            'source_path': str(base_dir / 'knowledge' / kb_id / 'policy.txt'),
            'source_name': 'policy.txt',
            'doc_type': 'txt',
            'title': 'Password Policy',
            'section': 'Account Security',
            'section_path': ['Password Policy', 'Account Security'],
            'page_start': None,
            'page_end': None,
            'chunk_index': 0,
            'content': 'Password reset links expire after 15 minutes and require MFA for administrators.',
            'content_for_embedding': 'Password reset links expire after 15 minutes and require MFA for administrators.',
            'summary': 'Password policy summary',
            'tags': ['txt', 'security'],
            'keywords': ['password', 'reset', 'mfa'],
            'language': 'en',
            'created_at': None,
            'updated_at': '2026-03-07T00:00:00+00:00',
            'metadata': {'local_section_chunk_index': 0},
        },
        {
            'chunk_id': 'doc_1_chunk_00001',
            'doc_id': 'doc_1',
            'source_path': str(base_dir / 'knowledge' / kb_id / 'policy.txt'),
            'source_name': 'policy.txt',
            'doc_type': 'txt',
            'title': 'Password Policy',
            'section': 'General',
            'section_path': ['Password Policy', 'General'],
            'page_start': None,
            'page_end': None,
            'chunk_index': 1,
            'content': 'General onboarding guidance does not mention reset expiration.',
            'content_for_embedding': 'General onboarding guidance does not mention reset expiration.',
            'summary': 'Password policy summary',
            'tags': ['txt', 'general'],
            'keywords': ['onboarding'],
            'language': 'en',
            'created_at': None,
            'updated_at': '2026-03-07T00:00:00+00:00',
            'metadata': {'local_section_chunk_index': 1},
        },
    ]
    _write_jsonl(rag_dir / 'chunks.jsonl', chunks)
    np.save(rag_dir / 'embeddings.npy', np.asarray([[1.0, 0.0], [0.2, 0.0]], dtype=np.float32))
    return rag_dir


def _build_fake_chinese_rag_index(base_dir: Path, kb_id: str = 'default') -> Path:
    registry = KnowledgeRegistry(str(base_dir))
    kb = registry.upsert_base(knowledge_base=kb_id, name=kb_id)
    rag_dir = Path(kb['output_dir'])
    rag_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'generated_at': '2026-03-08T00:00:00+00:00',
        'input_dir': str(base_dir / 'knowledge' / kb_id),
        'output_dir': str(rag_dir),
        'collection_name': f'rag_chunks_{kb_id}',
        'embedding_model': 'fake-model',
        'documents': [
            {
                'doc_id': 'brand_doc',
                'source_path': str(base_dir / 'knowledge' / kb_id / 'brand.txt'),
                'source_name': 'brand.txt',
                'checksum': 'brand123',
                'doc_type': 'txt',
                'updated_at': '2026-03-08T00:00:00+00:00',
            },
            {
                'doc_id': 'finance_doc',
                'source_path': str(base_dir / 'knowledge' / kb_id / 'finance.txt'),
                'source_name': 'finance.txt',
                'checksum': 'finance123',
                'doc_type': 'txt',
                'updated_at': '2026-03-08T00:00:00+00:00',
            },
        ],
        'chunk_count': 2,
    }
    (rag_dir / 'ingestion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    chunks = [
        {
            'chunk_id': 'brand_doc_chunk_00000',
            'doc_id': 'brand_doc',
            'source_path': str(base_dir / 'knowledge' / kb_id / 'brand.txt'),
            'source_name': 'brand.txt',
            'doc_type': 'txt',
            'title': '2025美国品牌增长',
            'section': '品牌趋势',
            'section_path': ['2025美国品牌增长', '品牌趋势'],
            'page_start': None,
            'page_end': None,
            'chunk_index': 0,
            'content': '2025年美国品牌增长整体呈现总量趋稳、结构分化的特征。平均购买考虑度仅增长0.73个百分点，DoorDash增长6.39个百分点。',
            'content_for_embedding': '2025年美国品牌增长整体呈现总量趋稳、结构分化的特征。平均购买考虑度仅增长0.73个百分点，DoorDash增长6.39个百分点。',
            'summary': '品牌增长总结',
            'tags': ['txt', 'brand'],
            'keywords': ['美国', '品牌', '增长', 'DoorDash'],
            'language': 'zh',
            'created_at': None,
            'updated_at': '2026-03-08T00:00:00+00:00',
            'metadata': {'local_section_chunk_index': 0},
        },
        {
            'chunk_id': 'finance_doc_chunk_00000',
            'doc_id': 'finance_doc',
            'source_path': str(base_dir / 'knowledge' / kb_id / 'finance.txt'),
            'source_name': 'finance.txt',
            'doc_type': 'txt',
            'title': '2025全球金融市场',
            'section': '金融市场',
            'section_path': ['2025全球金融市场', '金融市场'],
            'page_start': None,
            'page_end': None,
            'chunk_index': 0,
            'content': '2025年全球金融市场在多重冲击下表现出较强韧性。多数主要资产类别实现较强回报，韩国股市涨幅接近70%。',
            'content_for_embedding': '2025年全球金融市场在多重冲击下表现出较强韧性。多数主要资产类别实现较强回报，韩国股市涨幅接近70%。',
            'summary': '金融市场总结',
            'tags': ['txt', 'finance'],
            'keywords': ['全球', '金融市场', '韧性', '韩国股市'],
            'language': 'zh',
            'created_at': None,
            'updated_at': '2026-03-08T00:00:00+00:00',
            'metadata': {'local_section_chunk_index': 0},
        },
    ]
    _write_jsonl(rag_dir / 'chunks.jsonl', chunks)
    np.save(rag_dir / 'embeddings.npy', np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    return rag_dir


def test_hybrid_rag_store_returns_ranked_evidence(tmp_path: Path):
    _build_fake_rag_index(tmp_path)
    store = HybridRAGStore(str(tmp_path))
    result = store.query('password reset expiration', top_k=2)
    assert 'error' not in result
    assert result['evidence'][0]['section'] == 'Account Security'
    assert '15 minutes' in result['evidence'][0]['content']


def test_hybrid_rag_store_can_query_selected_knowledge_base(tmp_path: Path):
    _build_fake_rag_index(tmp_path, kb_id='product')
    store = HybridRAGStore(str(tmp_path))
    result = store.query('password reset expiration', top_k=2, knowledge_base='product')
    assert 'error' not in result
    assert result['knowledge_base']['id'] == 'product'


def test_hybrid_rag_store_separates_chinese_topics(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    store = HybridRAGStore(str(tmp_path))

    brand = store.query('2025年美国品牌增长整体趋势是什么', knowledge_base='default')
    finance = store.query('2025年全球金融市场的情况怎么样', knowledge_base='default')

    assert brand['answerable'] is True
    assert finance['answerable'] is True
    assert brand['evidence'][0]['source_name'] == 'brand.txt'
    assert finance['evidence'][0]['source_name'] == 'finance.txt'
    assert len(brand['evidence']) == 1
    assert len(finance['evidence']) == 1


def test_intent_recognizes_explicit_rag_query():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise('knowledge password reset expiration')
    assert intent == 'rag_query'
    assert params['query'] == 'password reset expiration'


def test_agent_answers_rag_query_and_remembers_evidence(tmp_path: Path):
    _build_fake_rag_index(tmp_path)
    agent = Agent(str(tmp_path))
    response = agent.handle('knowledge password reset expiration')
    assert 'policy.txt' in response['display_content']
    assert response['display_filename'] == 'rag_evidence.txt'

    follow_up = agent.handle('based on the evidence, summarize the answer')
    assert 'policy.txt' in follow_up['display_content']
    assert follow_up['display_filename'] == 'rag_evidence.txt'


def test_agent_uses_selected_knowledge_base_for_rag_query(tmp_path: Path):
    _build_fake_rag_index(tmp_path, kb_id='product')
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('product')
    response = agent.handle('knowledge password reset expiration', knowledge_base='product')
    assert response['active_knowledge_base'] == 'product'
    assert 'Knowledge base: product' in response['display_content']


def test_agent_automatically_uses_selected_knowledge_base_for_general_questions(tmp_path: Path):
    _build_fake_rag_index(tmp_path, kb_id='product')
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('product')
    response = agent.handle('What is the overall trend?', knowledge_base='product')
    assert response['active_knowledge_base'] == 'product'
    assert response['display_filename'] == 'rag_evidence.txt'
    assert 'policy.txt' in response['display_content']


def test_agent_uses_registry_active_knowledge_base_when_request_does_not_pass_one(tmp_path: Path):
    _build_fake_rag_index(tmp_path, kb_id='product')
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('product')
    response = agent.handle('What is the overall trend?')
    assert response['active_knowledge_base'] == 'product'
    assert response['display_filename'] == 'rag_evidence.txt'


def test_agent_automatically_routes_chinese_questions_to_rag(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('default')

    brand = agent.handle('2025年美国品牌增长整体趋势是什么')
    finance = agent.handle('2025年全球金融市场的情况怎么样')

    assert brand['display_filename'] == 'rag_evidence.txt'
    assert finance['display_filename'] == 'rag_evidence.txt'
    assert 'brand.txt' in brand['display_content']
    assert 'finance.txt' in finance['display_content']


def test_rag_answer_prefers_grounded_local_summary(tmp_path: Path):
    _build_fake_rag_index(tmp_path)
    agent = Agent(str(tmp_path))

    class StubChatTool:
        def is_configured(self):
            return True

        def complete(self, *args, **kwargs):
            return {'message': 'According to industry trends, growth will continue.'}

        def complete_messages(self, *args, **kwargs):
            return {'message': 'chat'}

    agent.chat_tool = StubChatTool()
    from agent.rag_answerer import RAGAnswerAgent
    agent.rag_answer_agent = RAGAnswerAgent(agent.chat_tool)
    agent.context_agent.rag_answer_agent = agent.rag_answer_agent

    response = agent.handle('knowledge password reset expiration')
    assert '15 minutes' in response['message']
    assert '[1]' in response['message']
    assert 'industry trends' not in response['message']


def test_rag_answer_uses_key_numbers_from_evidence(tmp_path: Path):
    agent = Agent(str(tmp_path))
    retrieval_result = {
        'knowledge_base': {'id': 'default', 'name': 'default'},
        'answerable': True,
        'evidence': [
            {
                'source_name': 'trend.txt',
                'section': 'trend',
                'content': '2025 growth stays stable overall. Average consideration rises by 0.73 percentage points. DoorDash grows by 6.39 percentage points.',
                'score': 2.1,
                'source_path': 'trend.txt',
            }
        ],
    }
    response = agent.rag_answer_agent.answer('What is the 2025 overall growth trend?', retrieval_result)
    assert '0.73' in response['message']
    assert '6.39' in response['message']
    assert '来源：' in response['message']


def test_rag_answer_stays_concise_for_single_source_summary(tmp_path: Path):
    agent = Agent(str(tmp_path))
    retrieval_result = {
        'knowledge_base': {'id': 'default', 'name': 'default'},
        'answerable': True,
        'evidence': [
            {
                'source_name': 'brand.txt',
                'section': '品牌趋势',
                'content': '2025年美国品牌增长整体呈现总量趋稳、结构分化的特征。平均购买考虑度仅增长0.73个百分点，DoorDash增长6.39个百分点。AI品牌在高收入群体中增长明显，但这不是问题主轴。',
                'score': 3.2,
                'source_path': 'brand.txt',
            }
        ],
    }
    response = agent.rag_answer_agent.answer('2025年美国品牌增长整体趋势是什么', retrieval_result)
    assert 'DoorDash增长6.39个百分点' in response['message']
    assert '来源：brand.txt[1]' in response['message']
    assert 'AI品牌在高收入群体中增长明显' not in response['message']


def test_intent_treats_email_capability_question_as_chat():
    recognizer = IntentRecognizer(intent_tool=None)
    query = '你可以发送邮件吗'
    intent, params = recognizer.recognise(query)
    assert intent == 'chat'
    assert params['prompt'] == query


def test_agent_answers_general_capability_question_without_rag(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('default')
    query = '你除了对话之外还有什么功能'
    response = agent.handle(query)
    assert response['display_filename'] is None
    assert 'SMTP' in response['message']
    assert '知识库' in response['message']
    assert 'brand.txt' not in response['message']


def test_agent_answers_email_capability_question_without_triggering_send(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('default')
    query = '你可以发送邮件吗'
    response = agent.handle(query)
    assert '可以' in response['message']
    assert 'SMTP' in response['message']
    assert 'Error:' not in response['message']


def test_agent_does_not_auto_route_generic_open_chat_to_rag(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    agent = Agent(str(tmp_path))
    agent.rag_management_tool.select_knowledge_base('default')
    response = agent.handle('Tell me something interesting')
    assert response['display_filename'] is None
    assert 'brand.txt' not in response['message']


def test_agent_capability_question_uses_chat_model_when_available(tmp_path: Path):
    _build_fake_chinese_rag_index(tmp_path)
    agent = Agent(str(tmp_path))

    class StubChatTool:
        def __init__(self):
            self.called = False

        def is_configured(self):
            return True

        def complete_messages(self, messages, system_prompt=None):
            self.called = True
            return {'message': 'Yes. I can draft and send email when SMTP is configured.'}

    stub = StubChatTool()
    agent.chat_tool = stub

    response = agent.handle('Can you send email?')
    assert stub.called is True
    assert response['message'] == 'Yes. I can draft and send email when SMTP is configured.'
    assert 'Error:' not in response['message']


def test_agent_does_not_route_large_model_research_question_to_model_tool(tmp_path: Path):
    agent = Agent(str(tmp_path))

    class StubChatTool:
        def is_configured(self):
            return True

        def complete_messages(self, messages, system_prompt=None):
            return {'message': 'By 2026, major research hotspots will likely include reasoning, agent systems, long context, multimodality, and training efficiency.'}

    agent.chat_tool = StubChatTool()
    response = agent.handle('2026\u5e74\u5927\u6a21\u578b\u8fd8\u6709\u54ea\u4e9b\u7814\u7a76\u70ed\u70b9')
    assert response['message'].startswith('By 2026')
    assert 'Sentiment:' not in response['message']
