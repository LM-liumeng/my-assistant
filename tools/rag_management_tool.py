"""Knowledge-base management commands for multi-knowledge-base RAG."""

from __future__ import annotations

import io
import importlib.util
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from context.evidence_store import EvidenceStore
from context.knowledge_registry import KnowledgeRegistry
from security import SafetyLayer


class RAGManagementTool:
    def __init__(self, base_dir: str, evidence_store: EvidenceStore, safety: SafetyLayer) -> None:
        self.base_dir = Path(base_dir)
        self.project_root = self.base_dir.parent
        self.evidence_store = evidence_store
        self.safety = safety
        self.registry = KnowledgeRegistry(base_dir)
        self.pipeline_script = self.project_root / 'RAG' / 'rag_multiformat_ingestion_pipeline.py'
        self._pipeline_module = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()

    def manage(
        self,
        action: str,
        input_dir: str = '',
        knowledge_base: str = '',
        knowledge_base_name: str = '',
        description: str = '',
    ) -> Dict[str, Any]:
        normalized_action = (action or '').strip().lower()
        if normalized_action == 'status':
            return self.status(knowledge_base=knowledge_base)
        if normalized_action == 'contents':
            return self.browse_contents(knowledge_base=knowledge_base)
        if normalized_action in {'ingest', 'rebuild'}:
            return self.run_ingestion(
                knowledge_base=knowledge_base,
                input_dir=input_dir,
                rebuild=(normalized_action == 'rebuild'),
            )
        if normalized_action in {'save', 'create', 'update'}:
            return self.save_knowledge_base(
                knowledge_base=knowledge_base,
                knowledge_base_name=knowledge_base_name,
                input_dir=input_dir,
                description=description,
            )
        if normalized_action == 'select':
            return self.select_knowledge_base(knowledge_base)
        if normalized_action in {'list', 'list_kbs'}:
            return self.list_knowledge_bases()
        return {'error': f'Unknown RAG management action: {action}'}

    def list_knowledge_bases(self) -> Dict[str, Any]:
        rows = self.registry.list_bases()
        active_id = self.registry.get_active_id()
        lines = [f'Active knowledge base: {active_id}', 'Knowledge bases:']
        for row in rows:
            marker = '*' if row.get('is_active') else '-'
            lines.append(
                f"{marker} {row.get('id')} | {row.get('name', '')} | input={row.get('input_dir', '')} | index={row.get('output_dir', '')}"
            )
        return {
            'message': 'Knowledge-base registry loaded.',
            'knowledge_bases': rows,
            'active_knowledge_base': active_id,
            'display_content': '\n'.join(lines),
            'display_filename': 'rag_registry.txt',
        }

    def save_knowledge_base(
        self,
        knowledge_base: str,
        knowledge_base_name: str = '',
        input_dir: str = '',
        description: str = '',
    ) -> Dict[str, Any]:
        if not (knowledge_base or knowledge_base_name):
            return {'error': 'Knowledge-base id or name is required.'}
        kb = self.registry.upsert_base(
            knowledge_base=knowledge_base or knowledge_base_name,
            name=knowledge_base_name or knowledge_base,
            input_dir=input_dir or None,
            description=description,
        )
        return {
            'message': f"Knowledge base '{kb['id']}' saved.",
            'knowledge_base': kb,
            'knowledge_bases': self.registry.list_bases(),
            'active_knowledge_base': self.registry.get_active_id(),
            'display_content': self._format_kb_detail(kb),
            'display_filename': 'rag_kb_config.txt',
        }

    def select_knowledge_base(self, knowledge_base: str) -> Dict[str, Any]:
        if not knowledge_base:
            return {'error': 'Knowledge-base id is required.'}
        kb = self.registry.set_active(knowledge_base)
        return {
            'message': f"Knowledge base '{kb['id']}' is now active.",
            'knowledge_base': kb,
            'knowledge_bases': self.registry.list_bases(),
            'active_knowledge_base': kb['id'],
            'display_content': self._format_kb_detail(kb),
            'display_filename': 'rag_kb_config.txt',
        }

    def status(self, knowledge_base: str = '') -> Dict[str, Any]:
        kb = self.registry.get_base(knowledge_base)
        output_dir = Path(kb['output_dir'])
        manifest_path = output_dir / 'ingestion_manifest.json'
        chunks_path = output_dir / 'chunks.jsonl'
        embeddings_path = output_dir / 'embeddings.npy'
        manifest = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except Exception as exc:
                return {'error': f'Failed to read RAG manifest: {exc}'}

        lines = [
            f"Knowledge base: {kb['id']} ({kb.get('name', '')})",
            f"Input dir: {kb['input_dir']}",
            f"Index dir: {kb['output_dir']}",
            f'Pipeline: {self.pipeline_script}',
            f'Manifest present: {manifest_path.exists()}',
            f'Chunks present: {chunks_path.exists()}',
            f'Embeddings present: {embeddings_path.exists()}',
        ]
        if manifest:
            lines.extend([
                f"Embedding model: {manifest.get('embedding_model', '')}",
                f"Indexed documents: {len(manifest.get('documents', []))}",
                f"Chunk count: {manifest.get('chunk_count', 0)}",
                f"Last generated: {manifest.get('generated_at', '')}",
                f"Collection: {manifest.get('collection_name', '')}",
            ])
        else:
            lines.append('Manifest not available. Run ingestion first.')

        return {
            'message': f"RAG status loaded for knowledge base '{kb['id']}'.",
            'knowledge_base': kb,
            'active_knowledge_base': self.registry.get_active_id(),
            'display_content': '\n'.join(lines),
            'display_filename': 'rag_status.txt',
        }

    def browse_contents(self, knowledge_base: str = '', limit: int = 12, offset: int = 0) -> Dict[str, Any]:
        kb = self.registry.get_base(knowledge_base)
        output_dir = Path(kb['output_dir'])
        manifest_path = output_dir / 'ingestion_manifest.json'
        chunks_path = output_dir / 'chunks.jsonl'
        if not manifest_path.exists() or not chunks_path.exists():
            return {'error': f"RAG index content is unavailable for knowledge base '{kb['id']}'. Run ingestion first."}

        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception as exc:
            return {'error': f'Failed to read RAG manifest: {exc}'}

        rows = []
        try:
            with chunks_path.open('r', encoding='utf-8') as handle:
                for idx, line in enumerate(handle):
                    if idx < offset:
                        continue
                    if len(rows) >= limit:
                        break
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as exc:
            return {'error': f'Failed to read chunk index: {exc}'}

        lines = [
            f"Knowledge base: {kb['id']} ({kb.get('name', '')})",
            f"Index dir: {kb['output_dir']}",
            f"Indexed documents: {len(manifest.get('documents', []))}",
            f"Chunk count: {manifest.get('chunk_count', 0)}",
            f'Preview range: {offset + 1}-{offset + len(rows)}',
            'Preview chunks:',
        ]
        for idx, row in enumerate(rows, start=offset + 1):
            lines.append(
                f"[{idx}] {row.get('source_name', '')} | {row.get('section', '')} | chunk_id={row.get('chunk_id', '')}"
            )
            lines.append(str(row.get('content', ''))[:320].strip())
            lines.append(f"Path: {row.get('source_path', '')}")
            lines.append('')

        if not rows:
            lines.append('No chunk preview available.')

        return {
            'message': f"RAG index preview loaded for knowledge base '{kb['id']}'.",
            'knowledge_base': kb,
            'active_knowledge_base': self.registry.get_active_id(),
            'rows': rows,
            'display_content': '\n'.join(lines).strip(),
            'display_filename': 'rag_contents.txt',
        }

    def start_ingestion(self, knowledge_base: str = '', input_dir: str = '', rebuild: bool = False) -> Dict[str, Any]:
        kb = self.registry.upsert_base(
            knowledge_base=knowledge_base or self.registry.get_active_id(),
            input_dir=input_dir or None,
            name=knowledge_base or self.registry.get_active_id(),
        )
        source_dir = Path(kb['input_dir'])
        if not self.pipeline_script.exists():
            return {'error': f'RAG ingestion script not found: {self.pipeline_script}'}
        if not source_dir.exists() or not source_dir.is_dir():
            return {'error': f'Knowledge directory does not exist: {source_dir}'}

        job_id = uuid.uuid4().hex[:12]
        job = {
            'job_id': job_id,
            'status': 'queued',
            'knowledge_base': kb,
            'active_knowledge_base': self.registry.get_active_id(),
            'message': f"Queued {'rebuild' if rebuild else 'incremental ingestion'} for knowledge base '{kb['id']}'.",
            'logs': '',
            'result': None,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._run_ingestion_job,
            args=(job_id, kb['id'], input_dir, rebuild),
            daemon=True,
        )
        thread.start()
        return {
            'message': f"Started {'rebuild' if rebuild else 'incremental ingestion'} for knowledge base '{kb['id']}'.",
            'job_id': job_id,
            'status': 'queued',
            'knowledge_base': kb,
            'active_knowledge_base': self.registry.get_active_id(),
        }

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return {'error': f'RAG job not found: {job_id}'}
            return dict(job)

    def run_ingestion(self, knowledge_base: str = '', input_dir: str = '', rebuild: bool = False) -> Dict[str, Any]:
        kb = self.registry.upsert_base(
            knowledge_base=knowledge_base or self.registry.get_active_id(),
            input_dir=input_dir or None,
            name=knowledge_base or self.registry.get_active_id(),
        )
        source_dir = Path(kb['input_dir'])
        output_dir = Path(kb['output_dir'])
        if not self.pipeline_script.exists():
            return {'error': f'RAG ingestion script not found: {self.pipeline_script}'}
        if not source_dir.exists() or not source_dir.is_dir():
            return {'error': f'Knowledge directory does not exist: {source_dir}'}

        self.safety.log_tool_call('rag_manage', {'action': 'rebuild' if rebuild else 'ingest', 'knowledge_base': kb['id'], 'input_dir': str(source_dir)})
        result_code, logs, embedding_model = self._execute_pipeline(kb, rebuild)
        if result_code != 0:
            detail = logs or 'Unknown ingestion error.'
            return {
                'error': f'RAG ingestion failed: {detail}',
                'knowledge_base': kb,
                'active_knowledge_base': self.registry.get_active_id(),
                'display_content': detail,
                'display_filename': 'rag_ingestion_report.txt',
            }

        status = self.status(knowledge_base=kb['id'])
        title = f"RAG rebuild completed for knowledge base '{kb['id']}'." if rebuild else f"RAG incremental ingestion completed for knowledge base '{kb['id']}'."
        report_parts = [title]
        if logs:
            report_parts.append(logs)
        if embedding_model:
            report_parts.append(f'Effective embedding model: {embedding_model}')
        if status.get('display_content'):
            report_parts.append(status['display_content'])
        report = '\n\n'.join(part for part in report_parts if part)
        return {
            'message': title,
            'knowledge_base': kb,
            'active_knowledge_base': self.registry.get_active_id(),
            'display_content': report,
            'display_filename': 'rag_ingestion_report.txt',
        }

    def _run_ingestion_job(self, job_id: str, knowledge_base: str, input_dir: str, rebuild: bool) -> None:
        self._update_job(job_id, status='running', message='正在入库，请稍候...')
        result = self.run_ingestion(knowledge_base=knowledge_base, input_dir=input_dir, rebuild=rebuild)
        if 'error' in result:
            self._update_job(
                job_id,
                status='failed',
                message=result['error'],
                logs=result.get('display_content', ''),
                result=result,
            )
            return
        self._update_job(
            job_id,
            status='completed',
            message=result.get('message', '入库完成'),
            logs=result.get('display_content', ''),
            result=result,
        )

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(updates)

    def _execute_pipeline(self, kb: Dict[str, Any], rebuild: bool) -> tuple[int, str, str]:
        module = self._load_pipeline_module()
        args_list = [
            '--input_dir', kb['input_dir'],
            '--output_dir', kb['output_dir'],
            '--log_level', 'INFO',
        ]
        qdrant_url = os.environ.get('QDRANT_URL', '').strip()
        qdrant_collection = os.environ.get('QDRANT_COLLECTION', '').strip()
        if qdrant_url and qdrant_collection:
            args_list.extend(['--qdrant_url', qdrant_url, '--qdrant_collection', f"{qdrant_collection}_{kb['id']}", '--sync_qdrant'])

        args = module.parse_args(args_list)
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        logger = logging.getLogger('rag_ingestion')
        previous_level = logger.level
        previous_handlers = list(logger.handlers)
        previous_propagate = logger.propagate
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            return_code = module.run_pipeline(args)
        except Exception as exc:
            logger.exception('Pipeline execution failed: %s', exc)
            return_code = 1
        finally:
            handler.flush()
            logs = log_buffer.getvalue().strip()
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
        embedding_model = ''
        manifest_path = Path(kb['output_dir']) / 'ingestion_manifest.json'
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                embedding_model = str(manifest.get('embedding_model', ''))
            except Exception:
                embedding_model = ''
        return return_code, logs, embedding_model

    def _load_pipeline_module(self):
        if self._pipeline_module is not None:
            return self._pipeline_module
        spec = importlib.util.spec_from_file_location('assistant_demo_rag_pipeline', self.pipeline_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load RAG pipeline from {self.pipeline_script}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._pipeline_module = module
        return module

    def _format_kb_detail(self, kb: Dict[str, Any]) -> str:
        return '\n'.join([
            f"Knowledge base: {kb.get('id', '')}",
            f"Name: {kb.get('name', '')}",
            f"Description: {kb.get('description', '')}",
            f"Input dir: {kb.get('input_dir', '')}",
            f"Index dir: {kb.get('output_dir', '')}",
            f"Active: {kb.get('is_active', False)}",
        ])
