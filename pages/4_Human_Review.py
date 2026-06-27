"""
Human-in-the-Loop Review Interface
Streamlit page for reviewing and correcting LLM-as-Judge evaluations
"""
import streamlit as st
import sys
import json
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ui.theme import inject_dark_theme

from utils.human_annotations import (
    AnnotationManager,
    HumanAnnotation,
    create_pending_from_results
)
from datetime import datetime
import json


def build_fallback_reasoning(item: dict) -> str:
    """Build a minimal fallback reasoning text when judge reasoning is missing."""
    reasoning = item.get("llm_judge_reasoning") or ""
    if reasoning.strip():
        return reasoning

    score = item.get("llm_judge_score")
    expected = item.get("metadata", {}).get("full_result", {}).get("expected_answer") or item.get("expected_answer")
    model_answer = item.get("model_response") or ""

    if isinstance(score, (int, float)):
        if expected and model_answer:
            return (
                f"Otomatik değerlendirme skoru {score:.2f}. Beklenen cevap ile model cevabı arasındaki uyuma göre puanlandı."
            )
        return f"Otomatik değerlendirme skoru {score:.2f}. Bu kayıt için detaylı judge gerekçesi üretilmemiş."

    return "Bu kayıt için detaylı judge gerekçesi üretilmemiş."


def initialize_session_state():
    """Initialize session state variables"""
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotator_id' not in st.session_state:
        st.session_state.annotator_id = "default_annotator"
    if 'annotations_saved' not in st.session_state:
        st.session_state.annotations_saved = 0
    if 'review_queue_items' not in st.session_state:
        st.session_state.review_queue_items = []
    if 'review_queue_report' not in st.session_state:
        st.session_state.review_queue_report = None
    if 'reviewed_annotations' not in st.session_state:
        st.session_state.reviewed_annotations = {}
    if 'selected_run_id' not in st.session_state:
        st.session_state.selected_run_id = None


def main():
    st.set_page_config(
        page_title="Human Review - LLM Eval",
        page_icon="✍️",
        layout="wide"
    )
    inject_dark_theme()
    
    initialize_session_state()
    
    st.title("✍️ Human-in-the-Loop Review Interface")
    st.markdown("### LLM-as-Judge Değerlendirmelerini İncele ve Düzelt")
    
    # Initialize annotation manager
    annotation_manager = AnnotationManager()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Annotator ID
        annotator_id = st.text_input(
            "Değerlendirici ID:",
            value=st.session_state.annotator_id,
            help="Kimin yaptığını takip için"
        )
        st.session_state.annotator_id = annotator_id
        
        st.divider()
        
        # Statistics
        st.header("📊 İstatistikler")
        stats = annotation_manager.get_statistics()
        
        st.metric("Tamamlanan", stats['total_completed'])
        st.metric("Bekleyen", stats['total_pending'])
        
        if stats['total_completed'] > 0:
            st.metric(
                "Ortalama Uyum",
                f"{stats['average_agreement']:.2%}",
                help="LLM-Judge ile insan skorları arası uyum"
            )
            
            st.subheader("Düzeltme Tipleri")
            corrections = stats.get('corrections_by_type', {})
            for correction_type, count in corrections.items():
                st.write(f"**{correction_type.title()}**: {count}")
        
        st.divider()
        
        # Load results
        st.header("📥 Sonuç Yükle")
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.json"))
            if report_files:
                selected_file = st.selectbox(
                    "Rapor Seçin:",
                    report_files,
                    format_func=lambda x: x.name
                )
                st.session_state.selected_report_file = selected_file.name

                # ── Run seçici: store dosyasında birden fazla run varsa göster ──
                store_runs_info = []
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        _report_preview = json.load(f)
                    if isinstance(_report_preview, dict) and isinstance(_report_preview.get('runs'), list):
                        for r in _report_preview['runs']:
                            _meta = r.get('run_metadata') or {}
                            _ts = r.get('timestamp', 'N/A')[:19].replace('T', ' ')
                            _suite = _meta.get('test_suite', 'unknown')
                            _rid = _meta.get('run_id', '?')
                            _models = list((r.get('models') or {}).keys())
                            _models_str = ', '.join(_models) if _models else '?'
                            store_runs_info.append({
                                'run_id': _rid,
                                'label': f"{_ts} | {_suite} | {_models_str}",
                                'models': _models,
                                'run_data': r,
                            })
                except Exception:
                    pass

                selected_run_id = None
                selected_run_data = None

                if store_runs_info:
                    st.markdown("**Run Seçin:**")
                    run_options = {info['run_id']: info['label'] for info in store_runs_info}

                    chosen_run_id = st.selectbox(
                        "Değerlendirme Turu:",
                        options=list(run_options.keys()),
                        format_func=lambda rid: run_options.get(rid, rid),
                        help="Hangi test çalıştırmasının sonuçlarını incelemek istediğinizi seçin",
                        key="sidebar_run_selector"
                    )
                    selected_run_id = chosen_run_id
                    st.session_state.selected_run_id = chosen_run_id

                    chosen_run_info = next((x for x in store_runs_info if x['run_id'] == chosen_run_id), None)
                    if chosen_run_info:
                        selected_run_data = chosen_run_info['run_data']
                        st.caption(f"🆔 Run: `{chosen_run_id}`")
                        st.caption(f"🤖 Modeller: {', '.join(chosen_run_info['models'])}")
                else:
                    st.session_state.selected_run_id = None

                # Calculate max samples from selected run (or selected file for legacy)
                max_samples = 10  # default
                try:
                    if selected_run_data:
                        _target = selected_run_data
                    else:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            _tmp = json.load(f)
                        if isinstance(_tmp, dict) and isinstance(_tmp.get('runs'), list):
                            _runs = sorted(_tmp['runs'], key=lambda r: r.get('timestamp', ''), reverse=True)
                            _target = _runs[0] if _runs else {}
                        else:
                            _target = _tmp

                    for model_data in _target.get('models', {}).values():
                        for test_data in model_data.get('tests', {}).values():
                            results_count = len(test_data.get('results', []))
                            if results_count > max_samples:
                                max_samples = results_count
                except Exception:
                    pass

                sample_per_test = st.slider(
                    "Test başına örnek:",
                    min_value=1,
                    max_value=max_samples,
                    value=min(3, max_samples),
                    help=f"Rapordaki maksimum test sonucu sayısı: {max_samples}"
                )

                if st.button("🔄 Bekleyen Öğeler Oluştur"):
                    added = create_pending_from_results(
                        str(selected_file),
                        annotation_manager,
                        sample_per_test,
                        run_id=selected_run_id
                    )
                    if added > 0:
                        st.success(f"✅ {added} yeni öğe eklendi!")
                    else:
                        st.info("ℹ️ Yeni öğe eklenmedi (bu run için mevcut kayıtlar zaten var).")
                    st.rerun()
        
        st.divider()
        
        # Export training data
        st.header("💾 Veri Dışa Aktar")
        if st.button("📤 Eğitim Verisi Oluştur"):
            if stats['total_completed'] > 0:
                output_path = annotation_manager.export_for_training()
                st.success(f"✅ Dışa aktarıldı: {output_path}")
                
                # Show download button
                with open(output_path, 'r', encoding='utf-8') as f:
                    training_data = f.read()
                
                st.download_button(
                    label="⬇️ İndir",
                    data=training_data,
                    file_name=Path(output_path).name,
                    mime="application/jsonl"
                )
            else:
                st.warning("Henüz tamamlanmış annotation yok!")
    
    # Main content - Review interface
    st.header("🔍 Değerlendirme Sayfası")

    selected_report = st.session_state.get('selected_report_file')
    selected_run_id  = st.session_state.get('selected_run_id')

    # Compute effective source_report: includes run_id when a specific run is selected
    # so that get_pending_items returns only that run's items.
    if selected_report and selected_run_id:
        effective_source_report = f"{selected_report}::{selected_run_id}"
    else:
        effective_source_report = selected_report

    if selected_report:
        if selected_run_id:
            st.caption(f"📁 Aktif rapor: {selected_report}  |  🆔 Run: `{selected_run_id}`")
        else:
            st.caption(f"📁 Aktif rapor: {selected_report}")

    # Get pending items from storage (filtered by effective_source_report)
    all_pending_items = annotation_manager.get_pending_items(source_report=effective_source_report)

    # Build a stable in-session queue so saved items stay visible/navigable.
    # Reset queue when the report OR the selected run changes.
    report_changed = st.session_state.review_queue_report != effective_source_report
    if report_changed:
        st.session_state.review_queue_items = list(all_pending_items)
        st.session_state.review_queue_report = effective_source_report
        st.session_state.reviewed_annotations = {}
        st.session_state.current_index = 0
    else:
        # Keep existing queue items and append only newly created pending items
        queued_ids = {
            item.get('item_id') for item in st.session_state.review_queue_items
            if isinstance(item, dict) and item.get('item_id')
        }
        for item in all_pending_items:
            item_id = item.get('item_id')
            if item_id and item_id not in queued_ids:
                st.session_state.review_queue_items.append(item)
                queued_ids.add(item_id)

    all_review_items = st.session_state.review_queue_items
    
    # Model and Test Type filters
    if all_review_items:
        available_models = sorted(list(set(item['model_name'] for item in all_review_items)))
        available_test_types = sorted(list(set(item['test_category'] for item in all_review_items)))
        
        # Filters in columns
        filter_col1, filter_col2, filter_col3 = st.columns([3, 3, 1])
        
        with filter_col1:
            selected_models = st.multiselect(
                "🤖 Model Filtresi:",
                options=available_models,
                default=available_models,
                help="İncelemek istediğiniz modelleri seçin",
                key="model_filter"
            )
        
        with filter_col2:
            selected_test_types = st.multiselect(
                "📋 Test Tipi Filtresi:",
                options=available_test_types,
                default=available_test_types,
                help="İncelemek istediğiniz test tiplerini seçin",
                key="test_type_filter"
            )
        
        with filter_col3:
            st.metric("Toplam Model", len(available_models))
            st.metric("Toplam Test Tipi", len(available_test_types))
        
        # Filter items by selected models and test types
        if selected_models and selected_test_types:
            pending_items = [
                item for item in all_review_items
                if item['model_name'] in selected_models and item['test_category'] in selected_test_types
            ]
        else:
            pending_items = []
        
        # Reset index if filter changed
        current_filter = (selected_models, selected_test_types)
        if 'last_filter' not in st.session_state or st.session_state.last_filter != current_filter:
            st.session_state.current_index = 0
            st.session_state.last_filter = current_filter

        reviewed_ids = set(st.session_state.reviewed_annotations.keys())
        reviewed_in_filter = sum(1 for item in pending_items if item.get('item_id') in reviewed_ids)
        st.info(
            f"📊 {len(pending_items)} öğe gösteriliyor "
            f"(Toplam kuyruk: {len(all_review_items)} | Bu filtrede değerlendirilen: {reviewed_in_filter})"
        )
    else:
        pending_items = []
    
    if not pending_items:
        st.info("👍 Bekleyen değerlendirme yok! Soldaki menüden sonuç dosyası yükleyin.")
        
        # Show completed annotations
        if stats['total_completed'] > 0:
            st.divider()
            st.subheader("📝 Tamamlanan Değerlendirmeler")
            
            completed = annotation_manager.load_all_annotations()
            
            # Create summary table
            import pandas as pd
            
            df_data = []
            for ann in completed[-20:]:  # Last 20
                _lbl = "TAM_DOGRU" if ann.llm_judge_score >= 0.9 else "KISMEN_DOGRU" if ann.llm_judge_score >= 0.4 else "YANLIS"
                df_data.append({
                    "Test ID": ann.test_id,
                    "Kategori": ann.test_category,
                    "Model": ann.model_name,
                    "LLM Kararı": _lbl,
                    "İnsan Skoru": f"{ann.human_score:.2f}",
                    "Fark": f"{abs(ann.llm_judge_score - ann.human_score):.2f}",
                    "Düzeltme": ann.correction_type,
                    "Tarih": ann.timestamp.split('T')[0]
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, width='stretch')
        
        return
    
    # Navigation
    total_items = len(pending_items)
    current_index = st.session_state.current_index
    
    if current_index >= total_items:
        st.session_state.current_index = 0
        current_index = 0
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.progress(
            (current_index + 1) / total_items,
            text=f"İlerleme: {current_index + 1} / {total_items}"
        )
    
    # Get current item
    current_item = pending_items[current_index]
    
    # Display item for review with prominent model name
    col_title1, col_title2 = st.columns([3, 1])
    with col_title1:
        st.subheader(f"📋 Test: {current_item['test_category']}")
    with col_title2:
        st.info(f"🤖 **{current_item['model_name']}**")

    current_item_id = current_item.get('item_id')
    reviewed_annotations = st.session_state.reviewed_annotations
    existing_review = reviewed_annotations.get(current_item_id)
    if existing_review:
        st.success(
            "✅ Bu öğe daha önce kaydedildi. Düzenleyip tekrar kaydedebilirsiniz; öğe listeden düşmez."
        )
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📝 Soru")
        st.info(current_item['question'])
        
        st.markdown("#### 🤖 Model Yanıtı")
        st.caption("👇 Bu yanıtı okuyup kalitesini değerlendirin (sağ taraftaki slider ile)")
        # Try multiple field names for model response (fallback)
        model_response = current_item.get('model_response') or ''
        
        # If empty, try to extract from full_result based on test type
        if not model_response:
            full_result = current_item.get('metadata', {}).get('full_result', {})
            test_category = current_item.get('test_category', '')
            
            # Self-consistency: check by_temperature
            if 'by_temperature' in full_result:
                responses = []
                for temp_key, temp_data in full_result.get('by_temperature', {}).items():
                    if isinstance(temp_data, dict):
                        sample_responses = temp_data.get('sample_responses', [])
                        if sample_responses:
                            responses.extend(sample_responses[:2])
                if responses:
                    unique_responses = list(dict.fromkeys(responses))[:3]
                    model_response = "\n---\n".join(unique_responses)
            
            # Multi-turn: check turns
            elif 'turns' in full_result:
                turn_responses = []
                for turn in full_result.get('turns', []):
                    if not isinstance(turn, dict):
                        continue
                    assistant_response = turn.get('assistant_response') or turn.get('response')
                    if assistant_response:
                        unresolved_count = turn.get('unresolved_intent_count', 0)
                        unresolved_suffix = (
                            f" | unresolved_intents={unresolved_count}"
                            if isinstance(unresolved_count, int) and unresolved_count > 0
                            else ""
                        )
                        turn_responses.append(
                            f"Turn {turn.get('turn', '?')}: {assistant_response}{unresolved_suffix}"
                        )
                if turn_responses:
                    model_response = "\n".join(turn_responses)
            
            # Function calling: check tool_calls or selected_tool
            elif 'tool_calls' in full_result or 'selected_tool' in full_result:
                import json as json_module
                tool_info = []
                if full_result.get('selected_tool'):
                    tool_info.append(f"Tool: {full_result['selected_tool']}")
                if full_result.get('parameters'):
                    tool_info.append(f"Parameters: {json_module.dumps(full_result['parameters'], ensure_ascii=False)}")
                if full_result.get('tool_calls'):
                    for tc in full_result.get('tool_calls', []):
                        if isinstance(tc, dict):
                            tool_info.append(f"Tool Call: {tc.get('name', 'unknown')}")
                if tool_info:
                    model_response = "\n".join(tool_info)
            
            # Standard fields
            else:
                for field in ['model_answer', 'response', 'output', 'predicted_value', 'answer', 'content']:
                    if full_result.get(field):
                        model_response = str(full_result[field])
                        break
        
        if model_response:
            # Show response in a nice box
            st.success(model_response)
        else:
            st.warning("⚠️ Model yanıtı bulunamadı")
            with st.expander("🔍 Kullanılabilir alanlar"):
                st.json(current_item)
        
        st.markdown("#### 🏷️ Metadata")
        metadata_payload = {
            "Model": current_item['model_name'],
            "Test ID": current_item['test_id'],
            "Latency": f"{current_item['metadata'].get('latency', 0):.2f}s"
        }

        full_result = current_item.get('metadata', {}).get('full_result', {}) or {}
        agent_evaluation = full_result.get('agent_evaluation', {}) if isinstance(full_result, dict) else {}
        trace_payload = full_result.get('trace', {}) if isinstance(full_result, dict) else {}
        unresolved_intent_summary = full_result.get('unresolved_intent_summary', {}) if isinstance(full_result, dict) else {}
        if isinstance(agent_evaluation, dict) and agent_evaluation:
            mode = agent_evaluation.get('mode')
            aggregate_score = agent_evaluation.get('aggregate_score')
            if mode:
                metadata_payload["Agent Eval Mode"] = mode
            if isinstance(aggregate_score, (int, float)):
                metadata_payload["Azure Agent Aggregate"] = round(float(aggregate_score), 4)

        if isinstance(trace_payload, dict) and trace_payload:
            trace_summary = trace_payload.get('summary', {}) or {}
            span_types = trace_summary.get('span_types', {}) or {}
            if isinstance(trace_summary.get('total_spans'), int):
                metadata_payload["Trace Spans"] = trace_summary['total_spans']
            if isinstance(trace_summary.get('failed_spans'), int):
                metadata_payload["Failed Trace Spans"] = trace_summary['failed_spans']
            if isinstance(span_types, dict) and span_types:
                metadata_payload["Trace Span Types"] = ", ".join(
                    f"{span_type}:{count}" for span_type, count in span_types.items()
                )

        if isinstance(unresolved_intent_summary, dict) and unresolved_intent_summary:
            unresolved_turns = unresolved_intent_summary.get('unresolved_turns')
            unresolved_rate = unresolved_intent_summary.get('unresolved_turn_rate')
            unresolved_total = unresolved_intent_summary.get('unresolved_intent_total')
            if isinstance(unresolved_turns, int):
                metadata_payload["Unresolved Intent Turns"] = unresolved_turns
            if isinstance(unresolved_total, int):
                metadata_payload["Unresolved Intent Count"] = unresolved_total
            if isinstance(unresolved_rate, (int, float)):
                metadata_payload["Unresolved Intent Rate"] = round(float(unresolved_rate), 4)

        st.json(metadata_payload)

        if isinstance(trace_payload, dict) and isinstance(trace_payload.get('spans'), list) and trace_payload.get('spans'):
            with st.expander("🧭 Agent Trace"):
                st.json(trace_payload.get('spans'))
    
    with col_right:
        st.markdown("#### 🤖 LLM-as-Judge Otomatik Puanlama")
        st.caption("⚠️ Bu, LLM Judge'ın SOLDAKİ model yanıtına verdiği otomatik puandır")
        
        llm_score = current_item['llm_judge_score']

        # Prefer stored label, fallback to deriving from score
        llm_judge_label = current_item.get('llm_judge_label')
        if not llm_judge_label:
            if llm_score >= 0.9:
                llm_judge_label = "TAM_DOGRU"
            elif llm_score >= 0.4:
                llm_judge_label = "KISMEN_DOGRU"
            else:
                llm_judge_label = "YANLIS"

        label_cfg = {
            "TAM_DOGRU":    {"emoji": "✅", "color": "green",  "text": "TAM DOĞRU"},
            "KISMEN_DOGRU": {"emoji": "🟡", "color": "orange", "text": "KISMİ DOĞRU"},
            "YANLIS":       {"emoji": "❌", "color": "red",    "text": "YANLIŞ"},
        }
        cfg = label_cfg.get(llm_judge_label, {"emoji": "⚪", "color": "gray", "text": llm_judge_label})

        st.markdown(
            f"**LLM Judge Kararı:** {cfg['emoji']} :{cfg['color']}[**{cfg['text']}**]"
        )

        effective_reasoning = build_fallback_reasoning(current_item)
        
        st.write("**Gerekçe:**")
        st.write(effective_reasoning or "_(Gerekçe mevcut değil)_")
        
        st.divider()
        
        st.markdown("#### ✍️ SİZİN Değerlendirmeniz")
        st.caption("👉 Soldaki MODEL YANITINI okuyup KALİTESİNİ değerlendirin")
        
        score_key = f"human_score_{current_item_id}"
        correction_key = f"correction_type_{current_item_id}"
        feedback_key = f"human_feedback_{current_item_id}"

        if existing_review:
            if score_key not in st.session_state:
                st.session_state[score_key] = float(existing_review.get('human_score', llm_score))
            if correction_key not in st.session_state:
                st.session_state[correction_key] = existing_review.get('correction_type', 'approve')
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = existing_review.get('human_feedback', '')
        else:
            if score_key not in st.session_state:
                st.session_state[score_key] = llm_score
            if correction_key not in st.session_state:
                st.session_state[correction_key] = 'approve'
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = ''

        # Human score input
        human_score = st.slider(
            "📊 Model Yanıtı için Puanınız (0-1):",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state[score_key]),
            step=0.05,
            key=score_key,
            help="SOLDAKİ model yanıtının kalitesini değerlendirin. LLM Judge'ın puanıyla karşılaştırılacak."
        )
        
        # Show comparison
        score_diff = abs(llm_score - human_score)
        if score_diff > 0:
            comparison_color = "red" if score_diff >= 0.3 else "orange" if score_diff >= 0.1 else "blue"
            st.markdown(f"🔍 **Fark:** :{comparison_color}[{score_diff:.2f}] (LLM Judge: {llm_score:.2f} ↔ Sizin: {human_score:.2f})")
        
        default_idx = {"approve": 0, "adjust": 1, "reject": 2}.get(
            st.session_state.get(correction_key, "approve"),
            0
        )
        
        # Correction type
        correction_type = st.radio(
            "🎯 LLM Judge'ın PUANLAMASINDAKİ Doğruluk:",
            options=["approve", "adjust", "reject"],
            index=default_idx,
            key=correction_key,
            format_func=lambda x: {
                "approve": "✅ DOĞRU - LLM Judge'ın puanı yerinde",
                "adjust": "🔧 YAKIN - LLM Judge'ın puanı kabul edilebilir ama düzeltilmeli",
                "reject": "❌ YANLIŞ - LLM Judge'ın puanlaması tamamen hatalı"
            }[x],
            help="LLM Judge'ın MODEL YANITI için verdiği puanın doğruluğunu belirtin"
        )
        
        # Human feedback
        human_feedback = st.text_area(
            "📝 Açıklama (Zorunlu):",
            placeholder="Örnek: 'LLM Judge hesaplama hatasını gözden kaçırmış, 0.8 yerine 0.3 olmalıydı çünkü...'",
            height=150,
            key=feedback_key,
            help="LLM Judge'ı iyileştirmek için: Neden bu puanı verdiniz? LLM Judge neyi gözden kaçırdı/yanlış değerlendirdi?"
        )
        
        # Submit button
        if st.button("💾 Kaydet ve Sonraki", type="primary", width='stretch'):
            if not human_feedback.strip():
                st.warning("⚠️ Lütfen geri bildirim ekleyin!")
            else:
                # Create annotation
                stable_key = f"{current_item.get('item_id', '')}|{st.session_state.annotator_id}"
                annotation_id = hashlib.md5(stable_key.encode()).hexdigest()[:12]
                
                annotation = HumanAnnotation(
                    annotation_id=annotation_id,
                    test_id=current_item['test_id'],
                    test_category=current_item['test_category'],
                    model_name=current_item['model_name'],
                    question=current_item['question'],
                    model_response=current_item.get('model_response', ''),
                    llm_judge_score=llm_score,
                    llm_judge_reasoning=current_item['llm_judge_reasoning'],
                    human_score=human_score,
                    human_feedback=human_feedback,
                    correction_type=correction_type,
                    annotator_id=st.session_state.annotator_id,
                    timestamp=datetime.now().isoformat(),
                    metadata=current_item['metadata']
                )
                
                # Save annotation
                annotation_manager.save_annotation(annotation, status="completed")
                annotation_manager.apply_annotation_to_report(annotation)

                st.session_state.reviewed_annotations[current_item_id] = {
                    "human_score": float(human_score),
                    "correction_type": correction_type,
                    "human_feedback": human_feedback,
                    "timestamp": annotation.timestamp
                }
                
                # Remove from pending
                annotation_manager.remove_pending_item(current_item['item_id'])
                
                # Update counter
                st.session_state.annotations_saved += 1
                
                # Move to next
                st.session_state.current_index = min(st.session_state.current_index + 1, total_items - 1)
                
                st.success(f"✅ Kaydedildi! ({st.session_state.annotations_saved} toplam)")
                st.rerun()
    
    # Navigation buttons
    st.divider()
    
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col1:
        if st.button("⬅️ Önceki", disabled=(current_index == 0)):
            st.session_state.current_index -= 1
            st.rerun()
    
    with nav_col2:
        st.write(f"**#{current_index + 1}** / {total_items}")
    
    with nav_col3:
        if st.button("➡️ Sonraki", disabled=(current_index >= total_items - 1)):
            st.session_state.current_index += 1
            st.rerun()
    
    # Quick actions
    st.divider()
    st.subheader("⚡ Hızlı Aksiyonlar")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("✅ Hızlı Onayla (Aynı Skor)", width='stretch'):
            # Auto-approve with same score
            stable_key = f"{current_item.get('item_id', '')}|{st.session_state.annotator_id}"
            annotation_id = hashlib.md5(stable_key.encode()).hexdigest()[:12]
            
            annotation = HumanAnnotation(
                annotation_id=annotation_id,
                test_id=current_item['test_id'],
                test_category=current_item['test_category'],
                model_name=current_item['model_name'],
                question=current_item['question'],
                model_response=current_item.get('model_response', ''),
                llm_judge_score=llm_score,
                llm_judge_reasoning=current_item['llm_judge_reasoning'],
                human_score=llm_score,
                human_feedback="Hızlı onay - LLM-Judge değerlendirmesi uygun",
                correction_type="approve",
                annotator_id=st.session_state.annotator_id,
                timestamp=datetime.now().isoformat(),
                metadata=current_item['metadata']
            )
            
            annotation_manager.save_annotation(annotation, status="completed")
            annotation_manager.apply_annotation_to_report(annotation)
            annotation_manager.remove_pending_item(current_item['item_id'])
            st.session_state.reviewed_annotations[current_item_id] = {
                "human_score": float(llm_score),
                "correction_type": "approve",
                "human_feedback": "Hızlı onay - LLM-Judge değerlendirmesi uygun",
                "timestamp": annotation.timestamp
            }
            st.session_state.current_index = min(st.session_state.current_index + 1, total_items - 1)
            st.rerun()
    
    with quick_col2:
        if st.button("⏭️ Atla (Sonra Değerlendir)", width='stretch'):
            st.session_state.current_index = min(st.session_state.current_index + 1, total_items - 1)
            st.rerun()
    
    with quick_col3:
        # Jump to next model
        current_model = current_item['model_name']
        
        # Find all unique models after current index
        remaining_items = pending_items[current_index+1:]
        remaining_models = []
        seen = set()
        for item in remaining_items:
            if item['model_name'] != current_model and item['model_name'] not in seen:
                remaining_models.append(item['model_name'])
                seen.add(item['model_name'])
        
        if remaining_models:
            # Find next occurrence of first different model
            next_model_items = [i for i, item in enumerate(pending_items[current_index+1:], start=current_index+1) 
                               if item['model_name'] == remaining_models[0]]
            
            if next_model_items:
                next_model_name = remaining_models[0]
                other_models_text = f" (+{len(remaining_models)-1})" if len(remaining_models) > 1 else ""
                if st.button(f"🔀 Sonraki Model: {next_model_name[:12]}{other_models_text}", width='stretch'):
                    st.session_state.current_index = next_model_items[0]
                    st.rerun()
            else:
                st.button("🔀 Sonraki Model", disabled=True, width='stretch')
        else:
            st.button("🔀 Sonraki Model", disabled=True, width='stretch')
    
    with quick_col4:
        if st.button("🔄 Başa Dön", width='stretch'):
            st.session_state.current_index = 0
            st.rerun()


if __name__ == "__main__":
    main()
