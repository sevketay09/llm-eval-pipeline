"""
🚀 LLM Evaluation Pipeline - Interactive Web Dashboard
========================================================

Gelişmiş web arayüzü ile model değerlendirme sistemi.

Kullanım:
    streamlit run app.py

Özellikler:
    - Model seçimi ve test çalıştırma
    - Test suite yönetimi
    - Real-time progress tracking
    - Sonuç görselleştirme
    - Model ve test konfigürasyonu
"""

import streamlit as st
import yaml
import json
import subprocess
import threading
import queue
import time
import requests
import signal
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.evaluation_store import get_time_series_data, DEFAULT_STORE_PATH
from datetime import datetime
import os
import sys

_stdout_log_lock = threading.Lock()


def _append_output_lines(output_list, text: str):
    """Normalize raw process text into clean lines for UI/analysis."""
    if not text:
        return

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for candidate in normalized.split("\n"):
        cleaned = candidate.strip()
        if cleaned:
            output_list.append(cleaned)


def _stream_eval_log_to_stdout(line: str):
    """Mirror evaluation logs to stdout for terminal/debug console visibility."""
    try:
        with _stdout_log_lock:
            print(f"[EVAL] {line}", flush=True)
    except Exception:
        pass


def _get_recent_error_excerpt(logs, max_lines: int = 120):
    """Extract the most relevant recent traceback/error lines from logs."""
    if not logs:
        return []

    tail = logs[-max_lines:]

    traceback_start = None
    for index, log_line in enumerate(tail):
        if "Traceback (most recent call last):" in log_line:
            traceback_start = index

    if traceback_start is not None:
        return tail[traceback_start:]

    error_markers = (" error", "exception", "failed", "❌")
    filtered = [line for line in tail if any(marker in line.lower() for marker in error_markers)]
    return filtered[-50:] if filtered else tail[-50:]

# Page config
st.set_page_config(
    page_title="LLM Evaluation Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== UTILITY FUNCTIONS ====================

@st.cache_data
def load_models_config():
    """Load models configuration from YAML"""
    try:
        with open("config/models.yaml", 'r') as f:
            config = yaml.safe_load(f)
            return config.get('models', {})
    except Exception as e:
        st.error(f"Failed to load models config: {e}")
        return {}


@st.cache_data
def load_tests_config():
    """Load tests configuration from YAML"""
    try:
        with open("config/tests.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load tests config: {e}")
        return {}


@st.cache_data
def load_embedding_models_config():
    """Load embedding models configuration from YAML"""
    try:
        with open("config/models.yaml", 'r') as f:
            config = yaml.safe_load(f)
            return config.get('embedding_models', {})
    except Exception as e:
        st.error(f"Failed to load embedding models config: {e}")
        return {}


def save_models_config(config):
    """Save models configuration to YAML"""
    try:
        import os
        import shutil
        
        # Check if config directory exists
        if not os.path.exists("config"):
            os.makedirs("config")
        
        # Check write permission
        config_path = "config/models.yaml"
        if os.path.exists(config_path) and not os.access(config_path, os.W_OK):
            st.error(f"❌ '{config_path}' dosyasına yazma izni yok!")
            return False
        
        # Save with backup
        if os.path.exists(config_path):
            backup_path = config_path + ".backup"
            shutil.copy2(config_path, backup_path)

        # Preserve full YAML structure (models, embedding_models, judge_model, etc.)
        full_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f) or {}

        full_config['models'] = config

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        st.cache_data.clear()
        return True
    
    except PermissionError as e:
        st.error(f"❌ İzin hatası: {e}")
        st.markdown("**Çözüm:** Terminal'de `chmod 664 config/models.yaml` komutunu çalıştırın")
        return False
    
    except yaml.YAMLError as e:
        st.error(f"❌ YAML format hatası: {e}")
        st.markdown("**Sorun:** Konfigürasyon YAML formatına uygun değil")
        return False
    
    except IOError as e:
        st.error(f"❌ Dosya yazma hatası: {e}")
        st.markdown("**Olası Nedenler:** Disk dolu, dosya kilitli, yol geçersiz")
        return False
    
    except Exception as e:
        st.error(f"❌ Beklenmeyen hata: {type(e).__name__}: {e}")
        import traceback
        with st.expander("🔧 Teknik Detay"):
            st.code(traceback.format_exc(), language="text")
        return False


def save_tests_config(config):
    """Save tests configuration to YAML"""
    try:
        with open("config/tests.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Failed to save tests config: {e}")
        return False


def get_available_datasets():
    """Scan eval_datasets directory for available test files"""
    datasets = []
    eval_dir = Path("eval_datasets")
    
    if eval_dir.exists():
        for json_file in eval_dir.rglob("*.json"):
            rel_path = json_file.relative_to(eval_dir)
            dataset_name = str(rel_path).replace('.json', '').replace('/', '_').replace('\\', '_')
            datasets.append({
                'name': dataset_name,
                'path': str(json_file),
                'category': rel_path.parts[0] if len(rel_path.parts) > 1 else 'other'
            })
    
    return datasets


def load_results(filepath):
    """Load evaluation results from JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None


def read_process_output_background(process, output_list):
    """Background thread to read process output without blocking UI"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                before_count = len(output_list)
                _append_output_lines(output_list, line)
                for appended_line in output_list[before_count:]:
                    _stream_eval_log_to_stdout(appended_line)

        # Drain any remaining buffered output after process exit
        if process.stdout:
            remaining = process.stdout.read()
            if remaining:
                before_count = len(output_list)
                _append_output_lines(output_list, remaining)
                for appended_line in output_list[before_count:]:
                    _stream_eval_log_to_stdout(appended_line)
    except Exception as exc:
        _stream_eval_log_to_stdout(f"Log reader error: {exc}")


# ==================== SESSION STATE INITIALIZATION ====================

if 'evaluation_running' not in st.session_state:
    st.session_state.evaluation_running = False

if 'evaluation_output' not in st.session_state:
    st.session_state.evaluation_output = []

if 'evaluation_process' not in st.session_state:
    st.session_state.evaluation_process = None

if 'last_result_file' not in st.session_state:
    st.session_state.last_result_file = None


# ==================== HEADER ====================

st.markdown('<div class="main-header">🤖 LLM Evaluation Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Model Değerlendirme Sistemi</div>', unsafe_allow_html=True)

st.markdown("---")


# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/200x100.png?text=LLM+Eval", width='stretch')
    
    st.markdown("### 📊 İstatistikler")
    
    models_config = load_models_config()
    tests_config = load_tests_config()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Model", len(models_config))
    with col2:
        test_suites = tests_config.get('test_suites', {})
        st.metric("Test Suite", len(test_suites))
    
    # Reports count
    reports_dir = Path("reports")
    report_count = len(list(reports_dir.glob("*.json"))) if reports_dir.exists() else 0
    st.metric("Toplam Rapor", report_count)
    
    st.markdown("---")
    st.markdown("**v1.0.0** | © 2026")


# ==================== MAIN TABS ====================

tab1, tab2, tab3 = st.tabs(["🚀 Evaluation Çalıştır", "📊 Sonuçları Görüntüle", "⚙️ Konfigürasyon"])


# ==================== TAB 1: RUN EVALUATION ====================

with tab1:
    st.header("🚀 Yeni Evaluation Başlat")
    
    st.markdown("""
    Model ve test seçimlerini yaparak yeni bir değerlendirme başlatın. 
    Sistem seçtiğiniz modelleri test edip detaylı rapor oluşturacak.
    """)
    
    st.markdown("---")
    
    # Model Selection Section
    st.subheader("1️⃣ Model Seçimi")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Test Edilecek Modeller** (birden fazla seçilebilir)")
        
        # Load embedding models
        embedding_models_config = load_embedding_models_config()
        
        # Decide model type
        model_type = st.radio(
            "Model Tipi:",
            ["LLM", "Embedding Models"],
            horizontal=True,
            help="LLM text generation için, Embedding modelleri semantic search için"
        )
        
        # Get models by provider based on selection
        if model_type == "LLM":
            models_to_show = models_config
        else:
            models_to_show = embedding_models_config
        
        models_by_provider = {}
        for model_key, model_config in models_to_show.items():
            provider = model_config.get('provider', 'other')
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model_key)
        
        selected_models = []
        
        for provider, model_keys in models_by_provider.items():
            st.markdown(f"**{provider.upper()}**")
            for model_key in model_keys:
                model_info = models_to_show[model_key]
                model_name = model_info.get('model_name', model_key)
                base_url = model_info.get('base_url', 'N/A')
                
                # Truncate long URLs
                if len(base_url) > 50:
                    base_url = base_url[:47] + "..."
                
                if st.checkbox(
                    f"{model_key}",
                    key=f"model_{model_key}",
                    help=f"Model: {model_name}\nEndpoint: {base_url}"
                ):
                    selected_models.append(model_key)
    
    with col2:
        # Judge only needed for LLM, not embedding models
        if model_type == "LLM":
            st.markdown("**Judge Model** (scoring için)")
            
            # Tüm modelleri göster (kullanıcı istediğini seçebilir)
            judge_models = list(models_config.keys())
            
            selected_judge = st.selectbox(
                "Judge model seç",
                judge_models,
                help="Değerlendirme için kullanılacak model (genellikle GPT-4o veya güçlü bir model önerilir)"
            )
            
            st.info(f"✅ Judge: **{selected_judge}**")
        else:
            # Embedding models don't need judge
            st.markdown("**Judge Model** (scoring için)")
            st.info("ℹ️ Embedding modelleri için judge gerekmez\n\n📊 Matematiksel metrikler kullanılır:\n- Spearman/Pearson correlation\n- NDCG@k, MRR, MAP\n- Clustering accuracy")
            selected_judge = None  # No judge needed for embeddings
    
    st.markdown("---")
    
    # Test Suite Selection
    st.subheader("2️⃣ Test Suite Seçimi")
    
    test_suites = tests_config.get('test_suites', {})
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        suite_mode = st.radio(
            "Test seçim modu:",
            ["Hazır Test Suite", "Custom Test Seçimi"],
            help="Hazır suite'ler veya özel test kombinasyonu seçin"
        )
    
    with col2:
        parallel_models = st.checkbox(
            "Parallel Model Execution",
            value=True,
            help="Modelleri paralel olarak test et (daha hızlı)"
        )
        
        max_samples = st.number_input(
            "Max Samples (test başına)",
            min_value=1,
            max_value=1000,
            value=100,
            help="Her test için maksimum örnek sayısı (all için 9999 girin)"
        )
    
    selected_tests = []
    selected_suite_name = None
    
    if suite_mode == "Hazır Test Suite":
        suite_names = list(test_suites.keys())
        selected_suite_name = st.selectbox(
            "Test Suite Seç",
            suite_names,
            help="Önceden tanımlı test grupları"
        )
        
        if selected_suite_name:
            suite_info = test_suites[selected_suite_name]
            selected_tests = [
                test_name for test_name in suite_info.get('tests', [])
                if isinstance(test_name, str) and test_name.strip()
            ]
            
            st.info(f"**{selected_suite_name}** suite'i {len(selected_tests)} test içeriyor:")
            
            # Show tests in expandable
            with st.expander("Testleri görüntüle"):
                cols = st.columns(3)
                for idx, test in enumerate(selected_tests):
                    with cols[idx % 3]:
                        st.markdown(f"✓ `{test}`")
    
    else:  # Custom selection
        st.markdown("**Available Tests**")
        
        # Get all possible tests from test_mapping
        all_possible_tests = set()
        for suite_info in test_suites.values():
            all_possible_tests.update(
                test_name
                for test_name in suite_info.get('tests', [])
                if isinstance(test_name, str) and test_name.strip()
            )
        
        all_possible_tests = sorted(list(all_possible_tests))
        
        # Group by category
        test_categories = {
            'embedding': [],
            'turkish': [],
            'fintech': [],
            'function': [],
            'benchmark': [],
            'other': []
        }
        
        for test in all_possible_tests:
            if 'embedding' in test:
                test_categories['embedding'].append(test)
            elif 'turkish' in test:
                test_categories['turkish'].append(test)
            elif 'fintech' in test:
                test_categories['fintech'].append(test)
            elif 'function' in test or 'tool' in test or 'agentic' in test:
                test_categories['function'].append(test)
            elif any(x in test for x in ['mmlu', 'hellaswag', 'truthful', 'humaneval', 'gsm8k']):
                test_categories['benchmark'].append(test)
            else:
                test_categories['other'].append(test)
        
        selected_tests = []
        
        for category, tests in test_categories.items():
            if tests:
                st.markdown(f"**{category.upper()}**")
                cols = st.columns(3)
                for idx, test in enumerate(tests):
                    with cols[idx % 3]:
                        if st.checkbox(test, key=f"test_{test}"):
                            selected_tests.append(test)
    
    st.markdown("---")
    
    # Test Parameters Configuration
    st.subheader("3️⃣ Test Parametreleri (İsteğe Bağlı)")
    
    # Initialize test parameters in session state
    if 'test_params' not in st.session_state:
        st.session_state.test_params = {}
    
    # Check if self_consistency is in selected tests
    if 'self_consistency' in selected_tests:
        with st.expander("🔄 Self-Consistency Test Parametreleri", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                num_runs = st.slider(
                    "Tekrar Sayısı (num_runs)",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Her soru için kaç kez yanıt üretilecek"
                )
                st.session_state.test_params['self_consistency_num_runs'] = num_runs
            
            with col2:
                temp_preset = st.selectbox(
                    "Temperature Preset",
                    ["Düşük (0.0, 0.3)", "Orta (0.0, 0.5, 0.7)", "Yüksek (0.0, 0.5, 1.0)", "Özel"],
                    index=1,
                    help="Test edilecek temperature değerleri"
                )
                
                if temp_preset == "Düşük (0.0, 0.3)":
                    temperatures = [0.0, 0.3]
                elif temp_preset == "Orta (0.0, 0.5, 0.7)":
                    temperatures = [0.0, 0.5, 0.7]
                elif temp_preset == "Yüksek (0.0, 0.5, 1.0)":
                    temperatures = [0.0, 0.5, 1.0]
                else:  # Özel
                    temp_input = st.text_input(
                        "Temperature değerleri (virgülle ayrılmış)",
                        value="0.0, 0.3, 0.7",
                        help="Örnek: 0.0, 0.3, 0.5, 0.7"
                    )
                    try:
                        temperatures = [float(t.strip()) for t in temp_input.split(',')]
                    except:
                        temperatures = [0.0, 0.3, 0.7]
                        st.warning("⚠️ Geçersiz format, varsayılan kullanılıyor")
                
                st.session_state.test_params['self_consistency_temperatures'] = temperatures
            
            st.info(f"ℹ️ Her soru {num_runs} kez × {len(temperatures)} temperature = **{num_runs * len(temperatures)} API çağrısı** yapılacak")
    
    # Add more test parameter configurations here for other tests
    if 'needle_haystack' in selected_tests:
        with st.expander("🔍 Needle in Haystack Parametreleri"):
            context_lengths = st.multiselect(
                "Context Length (tokens)",
                [1000, 2000, 4000, 8000, 16000],
                default=[2000, 4000],
                help="Test edilecek context uzunlukları"
            )
            st.session_state.test_params['needle_haystack_context_lengths'] = context_lengths
    
    st.markdown("---")

    # Global runtime model parameters
    st.subheader("4️⃣ Model Runtime Parametreleri (Global)")

    global_temperature = None
    global_top_p = None
    global_max_tokens = None

    if model_type == "LLM":
        enable_runtime_overrides = st.checkbox(
            "Seçili tüm modeller için aynı runtime parametrelerini uygula",
            value=False,
            help="Aynı test koşulları için temperature, top_p ve max_tokens değerlerini tüm seçili modellere uygular"
        )

        if enable_runtime_overrides:
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                global_temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.05,
                    help="Tüm seçili modeller için global temperature"
                )
            with pcol2:
                global_top_p = st.number_input(
                    "Top-p",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    help="Tüm seçili modeller için global top_p"
                )
            with pcol3:
                global_max_tokens = st.number_input(
                    "Max tokens",
                    min_value=1,
                    max_value=32768,
                    value=1024,
                    step=64,
                    help="Tüm seçili modeller için global max_tokens"
                )

            st.caption(
                f"Global override aktif: temperature={global_temperature}, "
                f"top_p={global_top_p}, max_tokens={int(global_max_tokens)}"
            )
    else:
        st.info("ℹ️ Bu runtime parametreleri yalnızca LLM için geçerlidir.")

    st.markdown("---")
    
    # Output Configuration
    st.subheader("5️⃣ Output Ayarları")
    st.info("📁 Rapor dosyası, değerlendirme başlatıldığında otomatik oluşturulur:\n`reports/eval_YYYYMMDD_HHMMSS.json`")
    
    st.markdown("---")
    
    # Summary and Run Button
    st.subheader("6️⃣ Çalıştır")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Seçili Model", len(selected_models))
    with col2:
        st.metric("Seçili Test", len(selected_tests))
    with col3:
        st.metric("Max Sample", max_samples if max_samples < 9999 else "All")
    with col4:
        st.metric("Parallel", "✅" if parallel_models else "❌")
    
    # Validation
    # For embedding models, judge is not required (uses mathematical metrics)
    if model_type == "Embedding Models":
        can_run = len(selected_models) > 0 and len(selected_tests) > 0
    else:
        can_run = len(selected_models) > 0 and len(selected_tests) > 0 and selected_judge
    
    if not can_run:
        if model_type == "LLM":
            st.warning("⚠️ En az 1 model, 1 test ve judge model seçmelisiniz!")
        else:
            st.warning("⚠️ En az 1 embedding model ve 1 test seçmelisiniz!")
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(
            "▶️ EVALUATION BAŞLAT",
            disabled=not can_run or st.session_state.evaluation_running,
            type="primary",
            width='stretch'
        ):
            # Handle custom test selection
            if suite_mode == "Custom Test Seçimi" and selected_tests:
                # Create temporary custom suite in tests.yaml
                tests_config = load_tests_config()
                
                # Add or update dashboard_custom suite
                if 'test_suites' not in tests_config:
                    tests_config['test_suites'] = {}
                
                tests_config['test_suites']['dashboard_custom'] = {
                    'enabled': True,
                    'tests': selected_tests,
                    'max_samples': 'all' if max_samples >= 9999 else max_samples
                }
                
                # Save test-specific parameters if configured
                if 'test_parameters' not in tests_config:
                    tests_config['test_parameters'] = {}
                
                # Add self-consistency parameters if configured
                if 'self_consistency' in selected_tests and 'test_params' in st.session_state:
                    if 'self_consistency_num_runs' in st.session_state.test_params:
                        tests_config['test_parameters']['self_consistency'] = {
                            'num_runs': st.session_state.test_params['self_consistency_num_runs'],
                            'temperatures': st.session_state.test_params.get('self_consistency_temperatures', [0.0, 0.3, 0.7])
                        }
                
                # Add needle_haystack parameters if configured
                if 'needle_haystack' in selected_tests and 'test_params' in st.session_state:
                    if 'needle_haystack_context_lengths' in st.session_state.test_params:
                        tests_config['test_parameters']['needle_haystack'] = {
                            'context_lengths': st.session_state.test_params['needle_haystack_context_lengths']
                        }
                
                # Save updated config
                save_tests_config(tests_config)
                
                # Use dashboard_custom suite
                selected_suite_name = 'dashboard_custom'
            
            # Build command
            output_file = f"reports/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cmd = [
                sys.executable,  # Python interpreter
                "-u",  # unbuffered output for real-time/error visibility
                "main.py",
                "--models", *selected_models,
                "--output", output_file
            ]
            
            # Add judge only for LLM (not for embeddings)
            if selected_judge:
                cmd.extend(["--judge", selected_judge])
            
            if selected_suite_name:
                cmd.extend(["--suite", selected_suite_name])

            if model_type == "LLM" and global_temperature is not None and global_top_p is not None and global_max_tokens is not None:
                cmd.extend(["--temperature", str(global_temperature)])
                cmd.extend(["--top-p", str(global_top_p)])
                cmd.extend(["--max-tokens", str(int(global_max_tokens))])
            
            if parallel_models:
                cmd.append("--parallel-models")
            
            # Start evaluation in background
            st.session_state.evaluation_running = True
            st.session_state.evaluation_output = []
            st.session_state.last_result_file = output_file
            
            # Run subprocess
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid  # Create new process group for clean termination
                )
                
                st.session_state.evaluation_process = process
                
                # Start background thread to read output (non-blocking)
                reader_thread = threading.Thread(
                    target=read_process_output_background,
                    args=(process, st.session_state.evaluation_output),
                    daemon=True
                )
                reader_thread.start()

                _stream_eval_log_to_stdout(f"Evaluation started | cmd: {' '.join(cmd)}")
                
                st.success("✅ Evaluation başlatıldı!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Evaluation başlatılamadı: {e}")
                st.session_state.evaluation_running = False
    
    # Stop button - Always visible when running (at top, sticky)
    if st.session_state.evaluation_running:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("⏹️ EVALUATION DURDUR", key="stop_eval_top", type="secondary", width='stretch'):
                process = st.session_state.evaluation_process
                if process:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                            process.wait()
                    except:
                        try:
                            process.kill()
                            process.wait()
                        except:
                            pass
                
                st.session_state.evaluation_running = False
                st.session_state.evaluation_process = None
                st.warning("⚠️ Evaluation durduruldu!")
                time.sleep(0.5)
                st.rerun()
    
    # Progress section
    if st.session_state.evaluation_running:
        st.subheader("📊 Test İlerlemesi")
        
        # Get process
        process = st.session_state.evaluation_process
        
        if process and process.poll() is None:
            # Still running - logs are read by background thread
            # Just read from session state (non-blocking)
            logs = st.session_state.evaluation_output
            
            # Parse current test name from logs
            current_test = "Başlatılıyor..."
            for line in reversed(logs[-30:]):
                if "# TEST:" in line:
                    current_test = line.split("# TEST:")[-1].strip()
                    break
                elif "Starting" in line and any(x in line for x in ["turkish_", "fintech_", "function_", "agentic_"]):
                    for word in line.split():
                        if "_" in word and any(x in word for x in ["turkish", "fintech", "function", "agentic"]):
                            current_test = word.strip(".:,")
                            break
                    break
            
            # Calculate progress
            total_lines = len(logs)
            estimated_progress = min(int((total_lines / 400) * 100), 99)
            
            # Display compact progress
            st.progress(estimated_progress / 100, text=f"İlerleme: {estimated_progress}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Log Satırı", total_lines)
            with col2:
                if current_test != "Başlatılıyor...":
                    st.info(f"🔍 **{current_test}**")
                else:
                    st.info("⏳ Çalışıyor...")
            
            # Minimal log view (collapsed)
            with st.expander("📋 Son Loglar (isteğe bağlı)", expanded=False):
                recent = logs[-15:] if len(logs) > 15 else logs
                if recent:
                    st.code("\n".join(recent), language="text")
            
            st.caption("💡 Detaylı loglar için VSCode terminaline bakabilirsiniz")
            with st.expander("🖥️ Terminal/Debug Log Komutları", expanded=False):
                st.code(
                    "Lokal:\n"
                    "  streamlit run app.py\n\n"
                    "Docker (debug compose):\n"
                    "  make tail-logs\n"
                    "  docker compose -f docker-compose.debug.yml logs -f llm-eval-dashboard\n",
                    language="bash"
                )
            
            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()
        
        else:
            # Finished
            st.session_state.evaluation_running = False
            st.session_state.evaluation_process = None

            if process:
                _stream_eval_log_to_stdout(f"Evaluation process exited with code {process.returncode}")
            
            if process and process.returncode == 0:
                st.success("✅ Evaluation tamamlandı!")
                st.balloons()
                
                st.info(f"📄 Sonuçlar kaydedildi: `{st.session_state.last_result_file}`")
                
                if st.button("📊 Sonuçları Görüntüle", key="view_results_btn"):
                    st.session_state.selected_result_file = st.session_state.last_result_file
                    st.rerun()
            else:
                st.error("❌ Evaluation başarısız oldu!")
                recent_error_lines = _get_recent_error_excerpt(st.session_state.evaluation_output)

                if recent_error_lines:
                    _stream_eval_log_to_stdout("---- Error summary (recent) ----")
                    for line in recent_error_lines[-40:]:
                        _stream_eval_log_to_stdout(line)
                else:
                    _stream_eval_log_to_stdout("No detailed error lines captured from subprocess output.")

                st.info(
                    "🧭 Hata detayları terminal/debug console'a da yazıldı. "
                    "Lokal çalışıyorsanız Streamlit terminalini, Docker'da `make tail-logs` çıktısını kontrol edin."
                )
                
                with st.expander("❌ Error Logs", expanded=True):
                    st.text_area(
                        "Errors",
                        value="\n".join(recent_error_lines),
                        height=300,
                        disabled=True
                    )

                st.code(
                    "# Docker debug loglarını canlı izle\n"
                    "make tail-logs\n\n"
                    "# Alternatif\n"
                    "docker compose -f docker-compose.debug.yml logs -f llm-eval-dashboard\n",
                    language="bash"
                )


# ==================== TAB 2: VIEW RESULTS ====================

with tab2:
    st.header("📊 Evaluation Sonuçlarını Görüntüle")
    
    # File selector
    reports_dir = Path("reports")
    
    if not reports_dir.exists():
        st.error("❌ Reports dizini bulunamadı!")
    else:
        report_files = sorted(list(reports_dir.glob("*.json")), reverse=True)
        
        if not report_files:
            st.warning("⚠️ Henüz rapor bulunamadı. Önce evaluation çalıştırın.")
        else:
            selected_file = st.selectbox(
                "📁 Rapor Dosyası Seçin:",
                report_files,
                format_func=lambda x: f"{x.name} ({x.stat().st_size // 1024} KB)",
                help="En son raporlar listenin başında"
            )
            store_runs = []
            
            # Load results
            results = load_results(selected_file)
            
            if results:
                # Unified store support: select one run from runs[]
                if isinstance(results, dict) and isinstance(results.get('runs'), list):
                    store_runs = sorted(
                        results.get('runs', []),
                        key=lambda r: r.get('timestamp', ''),
                        reverse=True
                    )

                    if not store_runs:
                        st.error("❌ Bu store dosyasında hiç run bulunamadı!")
                        st.stop()

                    latest_run = store_runs[0]
                    latest_ts = latest_run.get('timestamp', 'N/A')
                    latest_suite = latest_run.get('run_metadata', {}).get('test_suite', 'unknown')
                    latest_id = latest_run.get('run_metadata', {}).get('run_id', 'latest')

                    st.info(f"🆕 Varsayılan run: {latest_ts} | {latest_suite} | {latest_id}")

                    selected_run_index = 0
                    if st.checkbox("🧭 Geçmiş bir run seçmek istiyorum", value=False):
                        show_all_runs = st.checkbox("Tüm runları göster", value=False)

                        selectable_runs = store_runs if show_all_runs else store_runs[:20]
                        run_options = []
                        for idx, run in enumerate(selectable_runs):
                            ts = run.get('timestamp', 'N/A')
                            suite = run.get('run_metadata', {}).get('test_suite', 'unknown')
                            run_id = run.get('run_metadata', {}).get('run_id', f'run_{idx}')
                            run_options.append((idx, f"{ts} | {suite} | {run_id}"))

                        selected_run_index = st.selectbox(
                            "Store içinden run seçin:",
                            options=[x[0] for x in run_options],
                            format_func=lambda i: dict(run_options).get(i, str(i)),
                            help="Varsayılan olarak en güncel run seçilir"
                        )

                    results = store_runs[selected_run_index]

                # Display timestamp
                st.info(f"📅 **Evaluation Tarihi:** {results.get('timestamp', 'N/A')}")
                
                # Check if results have model comparison
                if 'models' not in results:
                    st.error("❌ Geçersiz rapor formatı!")
                else:
                    st.markdown("---")
                    
                    # Model comparison
                    st.subheader("📊 Model Karşılaştırması")
                    
                    model_keys = list(results['models'].keys())
                    
                    if len(model_keys) == 0:
                        st.warning("Model sonucu bulunamadı.")
                    else:
                        # Build comparison dataframe
                        comparison_data = []
                        
                        for model_key in model_keys:
                            model_data = results['models'][model_key]

                            summary_cmp = results.get('summary', {}).get('model_comparison', {}).get(model_key, {})
                            overall_metrics = model_data.get('overall_metrics', {})
                            tests_by_name = model_data.get('tests', {})

                            edge_cases_summary = tests_by_name.get('edge_cases', {}).get('summary', {})
                            behavior_score = edge_cases_summary.get('avg_scores', {}).get('behavior_score')
                            if behavior_score is None:
                                behavior_score = edge_cases_summary.get('overall_score')

                            # Fallback averages if summary comparison missing
                            total_score = 0
                            total_latency = 0
                            test_count = 0

                            for test_name, test_data in tests_by_name.items():
                                if 'summary' in test_data:
                                    summary = test_data['summary']
                                    total_score += summary.get('overall_score', 0)
                                    total_latency += summary.get('avg_latency', 0)
                                    test_count += 1

                            avg_score_fallback = (total_score / test_count) if test_count else 0
                            avg_latency_fallback = (total_latency / test_count) if test_count else 0

                            if test_count > 0 or summary_cmp:
                                comparison_data.append({
                                    'Model': model_key,
                                    'Avg Score': summary_cmp.get('overall_score', avg_score_fallback),
                                    'Avg Latency (s)': summary_cmp.get('avg_latency', avg_latency_fallback),
                                    'Tests': test_count,
                                    'Behavior Score': behavior_score,
                                    'Score Stability': summary_cmp.get('score_stability', overall_metrics.get('score_stability', 0)),
                                    'Schema Compliance': summary_cmp.get('schema_compliance_rate', overall_metrics.get('schema_compliance_rate', 0)),
                                    'Efficiency': summary_cmp.get('quality_latency_efficiency', overall_metrics.get('quality_latency_efficiency', 0)),
                                    'Error Rate': summary_cmp.get('error_rate', overall_metrics.get('error_rate', 0)),
                                    'Judge Agreement': summary_cmp.get('judge_agreement_rate', overall_metrics.get('judge_agreement_rate'))
                                })
                        
                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)
                            for numeric_col in [
                                'Avg Score', 'Avg Latency (s)', 'Behavior Score', 'Score Stability',
                                'Schema Compliance', 'Efficiency', 'Error Rate', 'Judge Agreement'
                            ]:
                                if numeric_col in df_comparison.columns:
                                    df_comparison[numeric_col] = pd.to_numeric(df_comparison[numeric_col], errors='coerce').fillna(0.0)
                            
                            executive_tab, diagnostics_tab = st.tabs(["🎯 Executive View", "🔬 Diagnostics View"])

                            with executive_tab:
                                col1, col2 = st.columns(2)

                                with col1:
                                    fig_score = px.bar(
                                        df_comparison,
                                        x='Model',
                                        y='Avg Score',
                                        title='Ortalama Skor Karşılaştırması',
                                        color='Avg Score',
                                        color_continuous_scale='Viridis'
                                    )
                                    st.plotly_chart(fig_score, width='stretch')

                                with col2:
                                    fig_latency = px.bar(
                                        df_comparison,
                                        x='Model',
                                        y='Avg Latency (s)',
                                        title='Ortalama Latency Karşılaştırması',
                                        color='Avg Latency (s)',
                                        color_continuous_scale='Reds'
                                    )
                                    st.plotly_chart(fig_latency, width='stretch')

                                st.dataframe(df_comparison, width='stretch')

                                st.subheader("🏆 Top 3 Öneri")
                                rec_col1, rec_col2, rec_col3 = st.columns(3)

                                with rec_col1:
                                    if not df_comparison.empty and 'Avg Score' in df_comparison.columns:
                                        best_score_idx = df_comparison['Avg Score'].idxmax()
                                        best_score_model = df_comparison.loc[best_score_idx, 'Model']
                                        best_score_value = df_comparison.loc[best_score_idx, 'Avg Score']
                                        st.metric("En İyi Model (Skor)", best_score_model, f"{best_score_value:.3f}")

                                with rec_col2:
                                    if not df_comparison.empty and 'Score Stability' in df_comparison.columns:
                                        stable_idx = df_comparison['Score Stability'].idxmax()
                                        stable_model = df_comparison.loc[stable_idx, 'Model']
                                        stable_value = df_comparison.loc[stable_idx, 'Score Stability']
                                        st.metric("En Stabil Model", stable_model, f"{stable_value:.3f}")

                                with rec_col3:
                                    if not df_comparison.empty and 'Efficiency' in df_comparison.columns:
                                        efficient_idx = df_comparison['Efficiency'].idxmax()
                                        efficient_model = df_comparison.loc[efficient_idx, 'Model']
                                        efficient_value = df_comparison.loc[efficient_idx, 'Efficiency']
                                        st.metric("En Verimli Model", efficient_model, f"{efficient_value:.3f}")

                                st.markdown("---")
                                st.subheader("🎯 Gelişmiş Karşılaştırma Analitiği")

                                adv_col1, adv_col2 = st.columns(2)

                                with adv_col1:
                                    fig_pareto = px.scatter(
                                        df_comparison,
                                        x='Avg Latency (s)',
                                        y='Avg Score',
                                        color='Score Stability',
                                        size='Efficiency',
                                        hover_name='Model',
                                        title='Pareto Görünümü (Skor vs Latency)'
                                    )
                                    fig_pareto.update_layout(xaxis_title='Latency (düşük daha iyi)', yaxis_title='Score (yüksek daha iyi)')
                                    st.plotly_chart(fig_pareto, width='stretch')

                                with adv_col2:
                                    if 'Judge Agreement' in df_comparison.columns and df_comparison['Judge Agreement'].notna().any():
                                        fig_judge = px.bar(
                                            df_comparison,
                                            x='Model',
                                            y='Judge Agreement',
                                            color='Judge Agreement',
                                            color_continuous_scale='Blues',
                                            title='Judge Agreement Rate'
                                        )
                                        st.plotly_chart(fig_judge, width='stretch')
                                    else:
                                        fig_schema = px.bar(
                                            df_comparison,
                                            x='Model',
                                            y='Schema Compliance',
                                            color='Schema Compliance',
                                            color_continuous_scale='Greens',
                                            title='Schema Compliance Rate'
                                        )
                                        st.plotly_chart(fig_schema, width='stretch')

                                trend_rows = []
                                for mk, tdata in results.get('trends', {}).items():
                                    trend_info = tdata.get('trend', {}) if isinstance(tdata, dict) else {}
                                    values = trend_info.get('values', [])
                                    timestamps = trend_info.get('timestamps', [])
                                    for idx, val in enumerate(values):
                                        ts = timestamps[idx] if idx < len(timestamps) else f'run_{idx+1}'
                                        trend_rows.append({'Model': mk, 'Run': ts, 'Score': val})

                                if trend_rows:
                                    trend_df = pd.DataFrame(trend_rows)
                                    fig_trend = px.line(
                                        trend_df,
                                        x='Run',
                                        y='Score',
                                        color='Model',
                                        markers=True,
                                        title='Model Score Trend (Historical Runs)'
                                    )
                                    st.plotly_chart(fig_trend, width='stretch')

                            with diagnostics_tab:
                                st.subheader("📈 Test Bazında Detay")

                                selected_model = st.selectbox(
                                    "Model seçin:",
                                    model_keys,
                                    key="diagnostics_model_selector"
                                )

                                model_tests = results['models'][selected_model].get('tests', {})

                                if model_tests:
                                    test_names = list(model_tests.keys())
                                    for test_name in test_names:
                                        test_data = model_tests[test_name]

                                        if 'summary' not in test_data:
                                            continue

                                        summary = test_data['summary']

                                        with st.expander(f"📋 {test_name}"):
                                            col1, col2, col3 = st.columns(3)

                                            with col1:
                                                st.metric("Score", f"{summary.get('overall_score', 0):.3f}")
                                            with col2:
                                                st.metric("Latency", f"{summary.get('avg_latency', 0):.2f}s")
                                            with col3:
                                                st.metric("Tests", summary.get('total_tests', 0))

                                            if 'avg_scores' in summary:
                                                st.markdown("**Detaylı Skorlar:**")
                                                scores_df = pd.DataFrame([summary['avg_scores']])
                                                st.dataframe(scores_df, width='stretch')

                                summary_rows = []
                                for test_name, test_data in model_tests.items():
                                    summary = test_data.get('summary') if isinstance(test_data, dict) else None
                                    if not isinstance(summary, dict):
                                        continue

                                    per_model_scores = []
                                    for mk in model_keys:
                                        mk_test_data = results['models'][mk].get('tests', {}).get(test_name, {})
                                        mk_summary = mk_test_data.get('summary') if isinstance(mk_test_data, dict) else None
                                        if not isinstance(mk_summary, dict):
                                            continue
                                        mk_score = mk_summary.get('overall_score')
                                        if isinstance(mk_score, (int, float)):
                                            per_model_scores.append((mk, mk_score))

                                    if len(per_model_scores) > 1:
                                        best_model = max(per_model_scores, key=lambda x: x[1])[0]
                                        worst_model = min(per_model_scores, key=lambda x: x[1])[0]
                                        # Eğer tüm modeller aynı skoru aldıysa fark yok
                                        if best_model == worst_model:
                                            worst_model = "—"
                                    elif len(per_model_scores) == 1:
                                        best_model = per_model_scores[0][0]
                                        worst_model = "—"
                                    else:
                                        best_model = "N/A"
                                        worst_model = "N/A"

                                    total_tests = summary.get('total_tests', 0)
                                    successful_tests = summary.get('successful_tests', total_tests)
                                    error_count = max(0, total_tests - successful_tests)

                                    summary_rows.append({
                                        'Test': test_name,
                                        'Score': summary.get('overall_score', 0),
                                        'Best Model': best_model,
                                        'Worst Model': worst_model,
                                        'Latency': summary.get('avg_latency', 0),
                                        'Errors': error_count,
                                        'Total': total_tests
                                    })

                                if summary_rows:
                                    st.markdown("---")
                                    st.subheader("🧭 Özet Panel")

                                    summary_df = pd.DataFrame(summary_rows)
                                    for col in ['Score', 'Latency', 'Errors', 'Total']:
                                        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').fillna(0.0)

                                    total_tests_count = int(summary_df['Total'].sum())
                                    total_error_count = int(summary_df['Errors'].sum())
                                    avg_score_over_tests = summary_df['Score'].mean() if not summary_df.empty else 0

                                    kcol1, kcol2, kcol3 = st.columns(3)
                                    with kcol1:
                                        st.metric("Ortalama Test Skoru", f"{avg_score_over_tests:.3f}")
                                    with kcol2:
                                        st.metric("Toplam Test Adedi", total_tests_count)
                                    with kcol3:
                                        st.metric("Toplam Hata Adedi", total_error_count)

                                    best_score_df = summary_df.sort_values('Score', ascending=False).head(3)
                                    worst_score_df = summary_df.sort_values('Score', ascending=True).head(3)

                                    pcol1, pcol2 = st.columns(2)
                                    with pcol1:
                                        st.markdown("**En İyi 3 Test**")
                                        st.dataframe(
                                            best_score_df[['Test', 'Score', 'Best Model', 'Worst Model', 'Errors']].round(3),
                                            width='stretch'
                                        )
                                    with pcol2:
                                        st.markdown("**En Kötü 3 Test**")
                                        st.dataframe(
                                            worst_score_df[['Test', 'Score', 'Best Model', 'Worst Model', 'Errors']].round(3),
                                            width='stretch'
                                        )

                                st.markdown("---")
                                st.subheader("💾 Export")

                                csv = df_comparison.to_csv(index=False)
                                st.download_button(
                                    label="📥 CSV olarak indir",
                                    data=csv,
                                    file_name=f"eval_comparison_{selected_file.stem}.csv",
                                    mime="text/csv"
                                )

                    # ── Time-series & latest-run analytics ──────────────────
                    st.markdown("---")
                    st.subheader("📈 Karşılaştırmalı Analitik")

                    ts_rows = get_time_series_data(str(selected_file))
                    if not ts_rows:
                        st.info("Grafik için yeterli veri yok. Birden fazla test çalıştırın.")
                    else:
                        ts_df = pd.DataFrame(ts_rows)
                        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'], errors='coerce')
                        ts_df = ts_df.dropna(subset=['timestamp']).sort_values('timestamp')

                        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
                            "📊 En Güncel Run: Model Karş.",
                            "🕒 Dataset × Zaman",
                            "🔄 Model × Zaman",
                        ])

                        # ── Chart 3: Latest run — per-dataset model bar chart ──
                        with chart_tab1:
                            latest_ts = ts_df['timestamp'].max()
                            latest_df = ts_df[ts_df['timestamp'] == latest_ts].copy()

                            available_tests = sorted(latest_df['test_name'].unique().tolist())
                            selected_test_latest = st.selectbox(
                                "Test / Dataset seçin:",
                                available_tests,
                                key="chart3_test"
                            )
                            chart3_df = latest_df[latest_df['test_name'] == selected_test_latest]

                            if chart3_df.empty:
                                st.warning("Seçili dataset için veri yok.")
                            else:
                                fig_latest = px.bar(
                                    chart3_df.sort_values('score', ascending=False),
                                    x='model',
                                    y='score',
                                    color='model',
                                    text='score',
                                    title=f"En Güncel Run ({latest_ts.strftime('%Y-%m-%d %H:%M')}) — {selected_test_latest}",
                                    labels={'score': 'Skor', 'model': 'Model'},
                                    range_y=[0, 1]
                                )
                                fig_latest.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                fig_latest.update_layout(showlegend=False)
                                st.plotly_chart(fig_latest, width='stretch')

                        # ── Chart 1: Dataset bazlı, model x zaman ─────────────
                        with chart_tab2:
                            all_tests = sorted(ts_df['test_name'].unique().tolist())
                            selected_test_trend = st.selectbox(
                                "Test / Dataset seçin:",
                                all_tests,
                                key="chart1_test"
                            )
                            chart1_df = ts_df[ts_df['test_name'] == selected_test_trend].copy()

                            n_runs_ds = chart1_df.groupby('model')['timestamp'].nunique().max() or 0
                            if n_runs_ds < 2:
                                st.info("⏳ Zaman serisi grafik için aynı testin en az 2 farklı run'da çalışması gerekiyor. Şu an tek run var.")
                                # Still show bar for current data
                                fig_c1_bar = px.bar(
                                    chart1_df.sort_values('score', ascending=False),
                                    x='model', y='score', color='model', text='score',
                                    title=f"{selected_test_trend} — Mevcut Run Model Karşılaştırması",
                                    range_y=[0, 1]
                                )
                                fig_c1_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                fig_c1_bar.update_layout(showlegend=False)
                                st.plotly_chart(fig_c1_bar, width='stretch')
                            else:
                                fig_c1 = px.line(
                                    chart1_df,
                                    x='timestamp', y='score', color='model',
                                    markers=True,
                                    title=f"{selected_test_trend} — Farklı Modeller Zaman Trendi",
                                    labels={'score': 'Skor', 'timestamp': 'Tarih', 'model': 'Model'},
                                    range_y=[0, 1]
                                )
                                fig_c1.update_layout(xaxis_title='Tarih', yaxis_title='Skor')
                                st.plotly_chart(fig_c1, width='stretch')

                        # ── Chart 2: Model bazlı, dataset x zaman ─────────────
                        with chart_tab3:
                            all_models = sorted(ts_df['model'].unique().tolist())
                            selected_model_trend = st.selectbox(
                                "Model seçin:",
                                all_models,
                                key="chart2_model"
                            )
                            chart2_df = ts_df[ts_df['model'] == selected_model_trend].copy()

                            n_runs_m = chart2_df.groupby('test_name')['timestamp'].nunique().max() or 0
                            if n_runs_m < 2:
                                st.info("⏳ Zaman serisi grafik için aynı modelin en az 2 farklı run'da çalışması gerekiyor. Şu an tek run var.")
                                fig_c2_bar = px.bar(
                                    chart2_df.sort_values('score', ascending=False),
                                    x='test_name', y='score', color='test_name', text='score',
                                    title=f"{selected_model_trend} — Mevcut Run Dataset Bazlı Skor",
                                    labels={'score': 'Skor', 'test_name': 'Dataset'},
                                    range_y=[0, 1]
                                )
                                fig_c2_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                fig_c2_bar.update_layout(showlegend=False)
                                st.plotly_chart(fig_c2_bar, width='stretch')
                            else:
                                fig_c2 = px.line(
                                    chart2_df,
                                    x='timestamp', y='score', color='test_name',
                                    markers=True,
                                    title=f"{selected_model_trend} — Dataset Bazlı Zaman Trendi",
                                    labels={'score': 'Skor', 'timestamp': 'Tarih', 'test_name': 'Dataset'},
                                    range_y=[0, 1]
                                )
                                fig_c2.update_layout(xaxis_title='Tarih', yaxis_title='Skor')
                                st.plotly_chart(fig_c2, width='stretch')


# ==================== TAB 3: CONFIGURATION ====================

with tab3:
    st.header("⚙️ Konfigürasyon Yönetimi")
    
    config_tab1, config_tab2 = st.tabs(["🤖 Modeller", "📝 Test Suites"])
    
    # Models Configuration
    with config_tab1:
        st.subheader("Model Konfigürasyonu")
        
        st.info("💡 Mevcut modelleri görüntüleyin veya aşağıdaki formu kullanarak yeni model ekleyin.")
        
        models_config = load_models_config()
        
        if models_config:
            st.markdown(f"**Toplam {len(models_config)} model tanımlı**")
            
            for model_key, model_config in models_config.items():
                with st.expander(f"🤖 {model_key}"):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.json(model_config)
                    
                    with col2:
                        if st.button("🗑️ Sil", key=f"delete_model_{model_key}", type="secondary"):
                            # Delete confirmation
                            if st.session_state.get(f'confirm_delete_{model_key}', False):
                                # Actually delete
                                del models_config[model_key]
                                if save_models_config(models_config):
                                    st.success(f"✅ Model '{model_key}' silindi!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Model silinemedi!")
                            else:
                                # Ask for confirmation
                                st.session_state[f'confirm_delete_{model_key}'] = True
                                st.warning(f"⚠️ '{model_key}' modelini silmek istediğinizden emin misiniz? Tekrar 'Sil' butonuna tıklayın.")
                                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ➕ Yeni Model Ekle")
        
        with st.form("add_model_form"):
            st.markdown("**Model Bilgileri**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_model_key = st.text_input(
                    "Model Key (Unique ID)*",
                    placeholder="örn: my-model-name",
                    help="Model için benzersiz tanımlayıcı (kebab-case önerilir)"
                )
                
                provider = st.selectbox(
                    "Provider*",
                    ["azure", "openai", "anthropic", "ollama", "lmstudio", "vllm", "custom"],
                    help="API provider tipi"
                )
                
                # Set provider-specific placeholders
                if provider == "azure":
                    base_url_placeholder = "https://your-resource.openai.azure.com/"
                    model_name_placeholder = "gpt-4o-pr (deployment name)"
                    api_key_default = ""
                elif provider == "ollama":
                    base_url_placeholder = "http://localhost:11434/v1"
                    model_name_placeholder = "llama3"
                    api_key_default = "dummy"
                elif provider == "lmstudio":
                    base_url_placeholder = "http://localhost:1234/v1"
                    model_name_placeholder = "local-model"
                    api_key_default = "dummy"
                elif provider == "vllm":
                    base_url_placeholder = "http://localhost:8000/v1"
                    model_name_placeholder = "default veya model path"
                    api_key_default = "dummy"
                elif provider == "openai":
                    base_url_placeholder = "https://api.openai.com/v1"
                    model_name_placeholder = "gpt-4o veya gpt-4o-mini"
                    api_key_default = ""
                else:
                    base_url_placeholder = "https://api.example.com/v1"
                    model_name_placeholder = "model-adı"
                    api_key_default = ""
                
                if provider == "azure":
                    base_url = st.text_input(
                        "Azure Endpoint (Base URL)*",
                        placeholder=base_url_placeholder,
                        help="Azure OpenAI resource endpoint (https://<name>.openai.azure.com/)"
                    )
                    
                    model_name = st.text_input(
                        "Deployment Name (Model Name)*",
                        placeholder=model_name_placeholder,
                        help="Azure'da oluşturduğunuz deployment adı"
                    )
                else:
                    base_url = st.text_input(
                        "Base URL*",
                        placeholder=base_url_placeholder,
                        help="API endpoint adresi"
                    )
                    
                    model_name = st.text_input(
                        "Model Name*",
                        placeholder=model_name_placeholder,
                        help="Provider'daki model adı"
                    )
            
            with col2:
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=api_key_default,
                    placeholder="dummy veya gerçek key",
                    help="API anahtarı (Ollama/LM Studio/vLLM için 'dummy' yeterli)"
                )
                
                temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                    help="Sampling temperature"
                )
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1,
                    max_value=128000,
                    value=4096,
                    step=1024,
                    help="Maksimum token sayısı"
                )
                
                supports_function_calling = st.checkbox(
                    "Function Calling Destekliyor",
                    value=True,
                    help="Model function calling'i destekliyor mu?"
                )
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                test_endpoint = st.form_submit_button("🔍 Endpoint'i Test Et", type="secondary")
            
            with col2:
                submit = st.form_submit_button("💾 Kaydet", type="primary")
            
            with col3:
                if st.form_submit_button("🔄 Formu Temizle", type="secondary"):
                    st.rerun()
        
        # Handle form submission
        if submit:
            # Validation
            errors = []
            warnings = []
            
            # Model Key validation
            if not new_model_key:
                errors.append("Model Key zorunludur")
            elif new_model_key in models_config:
                errors.append(f"Model Key '{new_model_key}' zaten mevcut")
            elif not new_model_key.replace("-", "").replace("_", "").isalnum():
                errors.append("Model Key sadece harf, rakam, tire (-) ve alt çizgi (_) içerebilir")
            elif " " in new_model_key:
                errors.append("Model Key boşluk içeremez (tire veya alt çizgi kullanın)")
            
            # Base URL validation
            if not base_url:
                errors.append("Base URL zorunludur")
            elif not (base_url.startswith("http://") or base_url.startswith("https://")):
                errors.append("Base URL 'http://' veya 'https://' ile başlamalı")
            else:
                # Check Azure OpenAI endpoint format
                if provider == "azure":
                    if ".openai.azure.com" not in base_url:
                        warnings.append("Azure OpenAI endpoint genelde '.openai.azure.com' içerir. Örnek: https://your-resource.openai.azure.com/")
                    if base_url.endswith("/"):
                        # Azure endpoint should end with /
                        pass
                    else:
                        warnings.append("Azure endpoint genelde '/' ile biter: " + base_url + "/")
                
                # Check if /v1 is included for local providers
                elif provider in ["ollama", "lmstudio", "vllm"]:
                    if "/v1" not in base_url:
                        warnings.append(f"{provider.capitalize()} için genelde '/v1' endpoint kullanılır. Örnek: {base_url.rstrip('/')}/v1")
            
            # Model Name validation
            if not model_name:
                errors.append("Model Name zorunludur")
            
            # API Key validation
            if provider in ["azure", "openai", "anthropic"]:
                if not api_key or api_key == "dummy":
                    warnings.append(f"{provider.capitalize()} için gerçek API Key gereklidir")
            
            # Display errors and warnings
            if errors:
                st.error("### ❌ Hatalı Alanlar:")
                for idx, error in enumerate(errors, 1):
                    st.error(f"{idx}. {error}")
                
                st.info("💡 Lütfen yukarıdaki hataları düzeltin ve tekrar deneyin.")
            
            if warnings and not errors:
                st.warning("### ⚠️ Uyarılar:")
                for idx, warning in enumerate(warnings, 1):
                    st.warning(f"{idx}. {warning}")
            
            # Save if no errors
            if not errors:
                with st.spinner("💾 Model kaydediliyor..."):
                    # Determine actual provider for config
                    # Note: Azure models are saved as "provider: openai" with api_version field
                    config_provider = "openai" if (provider == "azure" or "azure" in base_url.lower()) else provider
                    
                    # Create new model config
                    new_model_config = {
                        "provider": config_provider,
                        "base_url": base_url,
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "supports_function_calling": supports_function_calling,
                        "supports_streaming": True
                    }
                    
                    if api_key:
                        new_model_config["api_key"] = api_key
                    
                    # Add Azure-specific fields
                    if provider == "azure" or "azure" in base_url.lower():
                        new_model_config["api_version"] = "2025-01-01-preview"
                        # Note: For Azure, model_name should be the deployment name
                    
                    # Add to models config
                    models_config[new_model_key] = new_model_config
                    
                    # Save to file
                    try:
                        if save_models_config(models_config):
                            st.success(f"✅ Model '{new_model_key}' başarıyla eklendi!")
                            
                            # Show saved config
                            with st.expander("📋 Kaydedilen Konfigürasyon"):
                                st.code(yaml.dump({new_model_key: new_model_config}, default_flow_style=False), language="yaml")
                            
                            st.info("🔄 Sayfa yenileniyor...")
                            st.balloons()
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("❌ Model config/models.yaml dosyasına kaydedilemedi!")
                            st.markdown("""
**Olası Nedenler:**
- Dosya yazma izni yok
- Disk dolu
- YAML formatı bozuk

**Çözüm:** Terminal'de şu komutu çalıştırın:
```bash
ls -la config/models.yaml
chmod 664 config/models.yaml
```
""")
                    except Exception as e:
                        st.error(f"❌ Kaydetme sırasında hata oluştu!")
                        st.markdown(f"""
**Hata Tipi:** `{type(e).__name__}`  
**Hata Mesajı:** `{str(e)}`

**İpucu:** Bu bir uygulama hatası olabilir. Loglara bakın.
""")
                        
                        with st.expander("🔧 Teknik Detay"):
                            import traceback
                            st.code(traceback.format_exc(), language="text")
        
        if test_endpoint:
            # Validation for test
            errors = []
            if not base_url:
                errors.append("Base URL gerekli")
            elif not (base_url.startswith("http://") or base_url.startswith("https://")):
                errors.append("Base URL 'http://' veya 'https://' ile başlamalı")
            
            if not model_name:
                errors.append("Model Name gerekli")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner("🔍 Endpoint test ediliyor..."):
                    import requests
                    from datetime import datetime
                    
                    # Prepare test info based on provider
                    headers = {}
                    
                    if provider == "azure" or "azure" in base_url.lower():
                        # Azure OpenAI - test with chat completion
                        api_version = "2025-01-01-preview"
                        test_url = base_url.rstrip('/') + f'/openai/deployments/{model_name}/chat/completions?api-version={api_version}'
                        test_method = "POST"  # Azure needs POST for chat completion
                        
                        if api_key and api_key != "dummy":
                            headers['api-key'] = api_key  # Azure uses 'api-key' header
                            headers['Content-Type'] = 'application/json'
                            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                        else:
                            masked_key = "(yok)"
                        
                        # Minimal test payload for Azure
                        test_payload = {
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1
                        }
                    else:
                        # Standard OpenAI-compatible endpoint
                        test_url = base_url.rstrip('/') + '/models'
                        test_method = "GET"
                        test_payload = None
                        
                        if api_key and api_key != "dummy":
                            headers['Authorization'] = f'Bearer {api_key}'
                            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                        else:
                            masked_key = "dummy" if api_key == "dummy" else "(yok)"
                    
                    # Display test details
                    with st.expander("🔍 Test Detayları", expanded=True):
                        test_info = f"""
**Test Zamanı:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Provider:** `{provider}`  
**Model Name:** `{model_name}`  
**Base URL:** `{base_url}`  
**Test URL:** `{test_url}`
**Test Method:** `{test_method}`
**API Key:** `{masked_key}`
**Timeout:** 10 saniye
"""
                        if provider == "azure" or "azure" in base_url.lower():
                            test_info += f"\n**API Version:** `{api_version}`\n**Test:** Minimal chat completion (max_tokens=1)"
                        
                        st.markdown(test_info)
                    
                    try:
                        # Perform endpoint test
                        if test_method == "POST":
                            response = requests.post(test_url, headers=headers, json=test_payload, timeout=10)
                        else:
                            response = requests.get(test_url, headers=headers, timeout=10)
                        
                        # Display response details
                        st.markdown(f"**HTTP Status:** `{response.status_code} {response.reason}`")
                        
                        if response.status_code == 200:
                            st.success("✅ Endpoint erişilebilir ve çalışıyor!")
                            
                            # Try to show available models or response
                            try:
                                data = response.json()
                                
                                if provider == "azure" or "azure" in base_url.lower():
                                    # Azure chat completion response
                                    if "choices" in data and len(data["choices"]) > 0:
                                        st.info("✅ Deployment erişilebilir ve yanıt veriyor")
                                        with st.expander("📋 Test Response"):
                                            st.json(data)
                                    elif "id" in data:
                                        st.info("✅ API çalışıyor")
                                else:
                                    # Standard OpenAI response
                                    if "data" in data and isinstance(data["data"], list):
                                        st.info(f"📋 {len(data['data'])} model bulundu")
                                        
                                        with st.expander("Bulunan Modeller"):
                                            for model in data["data"][:10]:  # Show first 10
                                                model_id = model.get("id", "unknown")
                                                st.code(model_id, language=None)
                            except Exception as e:
                                # Response OK but JSON parsing failed - that's fine
                                pass
                        
                        elif response.status_code == 404:
                            st.warning("⚠️ Endpoint bulunamadı (404)")
                            
                            if provider == "azure" or "azure" in base_url.lower():
                                st.markdown("""
**Azure OpenAI için uyarı:**
- **Deployment adı doğru mu?** Model Name: deployment adı olmalı (örn: `gpt-4o-pr`)
- **Endpoint doğru mu?** Format: `https://<resource-name>.openai.azure.com/`
- **Resource name** doğruluğunu Azure Portal'dan kontrol edin

**Azure Portal kontrol:**
1. Azure OpenAI resource → "Model deployments" sayfasına gidin
2. Deployment adını (Deployment name) kopyalayın
3. Endpoint'i "Keys and Endpoint" bölümünden alın

**Yaygın hatalar:**
- ❌ Model adı yerine deployment adı: `gpt-4o` yerine `gpt-4o-pr` 
- ❌ Yanlış resource name: endpoint'te resource adı yanlış
- ❌ `/v1` eklenmesi: Azure endpoint'ine `/v1` eklenMEZ
""")
                            else:
                                st.markdown("""
**Olası Nedenler:**
- `/models` endpoint'i desteklenmiyor (bazı provider'larda normal)
- Base URL yanlış veya eksik (örn: `/v1` eksik olabilir)
- Provider farklı endpoint yapısı kullanıyor

**Öneriler:**
""")
                                
                                if provider == "ollama":
                                    st.info("💡 Ollama için `/v1` eklenmiş URL kullanın: `http://localhost:11434/v1`")
                                elif provider == "lmstudio":
                                    st.info("💡 LM Studio için `/v1` eklenmiş URL kullanın: `http://localhost:1234/v1`")
                                elif provider == "vllm":
                                    st.info("💡 vLLM için `/v1` eklenmiş URL kullanın: `http://localhost:8000/v1`")
                            
                            # Show response body
                            try:
                                response_text = response.text[:500]
                                if response_text:
                                    with st.expander("📄 Response Body"):
                                        st.code(response_text, language="text")
                            except Exception:
                                pass
                        
                        elif response.status_code == 401:
                            st.error("❌ Yetkilendirme hatası (401 Unauthorized)")
                            
                            if provider == "azure" or "azure" in base_url.lower():
                                st.markdown("""
**Azure OpenAI için:**
- **API Key doğru mu?** Azure Portal'dan "Keys and Endpoint" → KEY 1 veya KEY 2
- **Doğru resource'a ait mi?** Key, base URL'deki resource ile eşleşmeli
- Azure OpenAI `api-key` header'ı kullanır (Bearer token değil)

**Azure Portal kontrol adımları:**
1. Azure OpenAI resource sayfasına gidin
2. Sol menüden "Keys and Endpoint" seçin
3. KEY 1 veya KEY 2'yi kopyalayın (ikisi de çalışır)
4. Endpoint URL'in doğru olduğunu kontrol edin
""")
                            else:
                                st.markdown("""
**Sorun:** API Key geçersiz veya eksik

**Çözümler:**
- API Key'in doğru girildiğinden emin olun
- Bearer token formatını kontrol edin
- Ollama/LM Studio/vLLM için 'dummy' yazın
- Provider console'da key'in aktif olduğunu kontrol edin
""")
                        
                        elif response.status_code == 403:
                            st.error("❌ Erişim engellendi (403 Forbidden)")
                            st.markdown("""
**Sorun:** API Key geçerli ama bu endpoint'e erişim yok

**Çözümler:**
- API Key'in yeterli izinlere sahip olduğunu kontrol edin
- Provider'da model erişim ayarlarını kontrol edin
""")
                        
                        elif response.status_code == 429:
                            st.error("❌ Rate limit aşıldı (429 Too Many Requests)")
                            st.markdown("**Sorun:** Çok fazla istek gönderildi. Birkaç dakika bekleyip tekrar deneyin.")
                        
                        elif response.status_code == 500:
                            st.error("❌ Server hatası (500 Internal Server Error)")
                            st.markdown("**Sorun:** Provider tarafında bir hata var. Loglara bakın veya biraz bekleyip tekrar deneyin.")
                        
                        elif response.status_code == 503:
                            st.error("❌ Servis kullanılamıyor (503 Service Unavailable)")
                            st.markdown("**Sorun:** Provider servisi şu anda çalışmıyor. Daha sonra tekrar deneyin.")
                        
                        else:
                            st.warning(f"⚠️ Beklenmeyen HTTP status: {response.status_code}")
                            
                            # Show response body for debugging
                            try:
                                response_text = response.text[:1000]
                                if response_text:
                                    with st.expander("📄 Response Body (İlk 1000 karakter)"):
                                        st.code(response_text, language="text")
                            except Exception:
                                pass
                        
                        # Show response headers
                        if response.headers:
                            with st.expander("📋 Response Headers"):
                                for key, value in response.headers.items():
                                    st.text(f"{key}: {value}")
                    
                    except requests.exceptions.Timeout:
                        st.error("❌ Timeout: Endpoint'e erişilemiyor (10 saniye)")
                        st.markdown(f"""
**Sorun:** Endpoint 10 saniye içinde cevap vermedi

**Olası Nedenler:**
- Server çalışmıyor veya dondu
- Ağ bağlantısı yavaş
- Firewall kuralları engelleme yapıyor
- URL yanlış

**Kontrol Edin:**
- `{base_url}` adresine erişilebiliyor mu?
- Ollama/LM Studio çalışıyor mu? (terminal'de kontrol edin)
- Port doğru mu? (Ollama: 11434, LM Studio: 1234, vLLM: 8000)
""")
                    
                    except requests.exceptions.ConnectionError as e:
                        st.error("❌ Bağlantı hatası: Endpoint'e ulaşılamıyor")
                        
                        if provider == "azure" or "azure" in base_url.lower():
                            st.markdown("""
**Azure OpenAI için:**
- Azure OpenAI servisiniz çalışıyor mu?
- Endpoint URL'i doğru mu? Format: `https://<resource>.openai.azure.com/`
- Network/firewall kuralları Azure OpenAI'ya erişimi engelliyor mu?
- Resource adı Azure Portal'dakiyle aynı mı?

**İnternet bağlantısı kontrolü:**
```bash
curl https://your-resource.openai.azure.com/
```
""")
                        else:
                            st.markdown("""
**Sorun:** TCP bağlantısı kurulamadı

**Olası Nedenler:**
- Server çalışmıyor
- Port yanlış
- Firewall engelleme yapıyor
- URL'de yazım hatası var

**Kontrol Edin:**
""")
                            
                            if provider == "ollama":
                                st.code("# Ollama çalışıyor mu?\nollama list", language="bash")
                            elif provider == "lmstudio":
                                st.info("💡 LM Studio'yu açın ve 'Local Server' sekmesinde 'Start Server' butonuna basın")
                            elif provider == "vllm":
                                st.code("# vLLM çalışıyor mu?\ncurl http://localhost:8000/health", language="bash")
                        
                        # Show detailed error
                        with st.expander("🔧 Teknik Detay"):
                            st.code(str(e), language="text")
                    
                    except requests.exceptions.SSLError as e:
                        st.error("❌ SSL/TLS hatası")
                        st.markdown("""
**Sorun:** HTTPS sertifikası doğrulanamadı

**Çözümler:**
- Self-signed sertifika kullanıyorsanız HTTP kullanın (http://)
- Sertifika geçerli ve güncel mi kontrol edin
""")
                        with st.expander("🔧 Teknik Detay"):
                            st.code(str(e), language="text")
                    
                    except Exception as e:
                        st.error("❌ Beklenmeyen hata")
                        st.markdown(f"""
**Hata Mesajı:** `{str(e)}`

**İpucu:** Bu bilgileri kopyalayıp destek ekibine gönderin.
""")
                        
                        with st.expander("🔧 Tam Stack Trace"):
                            import traceback
                            st.code(traceback.format_exc(), language="text")
        
        st.markdown("---")
        
        st.markdown("### 📄 models.yaml Dosyasını Düzenle")
        
        if st.button("📝 Config dosyasını aç", key="open_models_config"):
            st.code(open("config/models.yaml").read(), language="yaml")
    
    # Test Suites Configuration
    with config_tab2:
        st.subheader("Test Suite Konfigürasyonu")
        
        st.info("💡 Mevcut test suite'lerini görüntüleyin veya config/tests.yaml dosyasını düzenleyerek yeni suite ekleyin.")
        
        tests_config = load_tests_config()
        
        if 'test_suites' in tests_config:
            for suite_name, suite_config in tests_config['test_suites'].items():
                with st.expander(f"📦 {suite_name}"):
                    st.json(suite_config)
        
        st.markdown("---")
        
        st.markdown("### 📄 tests.yaml Dosyasını Düzenle")
        
        if st.button("📝 Config dosyasını aç", key="open_tests_config"):
            st.code(open("config/tests.yaml").read(), language="yaml")


# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>LLM Evaluation Pipeline v1.0</strong></p>
    <p>Model değerlendirme sistemi</p>
    <p>© 2026 | MIT License</p>
</div>
""", unsafe_allow_html=True)
