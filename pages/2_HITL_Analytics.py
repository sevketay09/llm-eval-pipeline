"""
Human-in-the-Loop Analytics Dashboard
Detailed statistics and insights from human annotations
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.human_annotations import AnnotationManager
from evaluators import HumanFeedbackEvaluator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.set_page_config(
        page_title="HITL Analytics",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Human-in-the-Loop Analytics")
    st.markdown("### İnsan Geri Bildirimi Analizi ve İçgörüler")
    
    # Initialize
    annotation_manager = AnnotationManager()
    evaluator = HumanFeedbackEvaluator(annotation_manager)
    
    # Get statistics
    stats = annotation_manager.get_statistics()
    
    if stats['total_completed'] == 0:
        st.warning("⚠️ Henüz tamamlanmış annotation yok. Önce değerlendirme yapın!")
        st.info("👉 Sol menüden **1_Human_Review** sayfasına gidin.")
        return
    
    # Overview metrics
    st.header("📊 Genel Bakış")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Toplam Değerlendirme",
            stats['total_completed'],
            delta=f"+{stats['total_pending']} bekliyor"
        )
    
    with col2:
        st.metric(
            "Ortalama Uyum",
            f"{stats['average_agreement']:.1%}",
            help="LLM-Judge ile insan skorları arası uyum"
        )
    
    with col3:
        approval_rate = stats['corrections_by_type']['approve'] / stats['total_completed']
        st.metric(
            "Onay Oranı",
            f"{approval_rate:.1%}",
            help="LLM-Judge'ın doğru değerlendirme oranı"
        )
    
    with col4:
        st.metric(
            "Değerlendiriciler",
            len(stats['annotators'])
        )
    
    # Correction Type Breakdown
    st.header("🔧 Düzeltme Tipi Dağılımı")
    
    corrections = stats['corrections_by_type']
    
    fig_corrections = go.Figure(data=[
        go.Pie(
            labels=['Approve', 'Adjust', 'Reject'],
            values=[
                corrections['approve'],
                corrections['adjust'],
                corrections['reject']
            ],
            marker=dict(colors=['#00cc88', '#ffaa00', '#ff4444']),
            textinfo='label+percent+value'
        )
    ])
    
    fig_corrections.update_layout(
        title="Değerlendirme Sonuçları Dağılımı",
        height=400
    )
    
    st.plotly_chart(fig_corrections, width='stretch')
    
    # Category Breakdown
    st.header("📁 Kategori Bazlı Analiz")
    
    category_data = []
    for category, cat_stats in stats['by_category'].items():
        category_data.append({
            "Kategori": category,
            "Değerlendirme Sayısı": cat_stats['count'],
            "Ortalama İnsan Skoru": cat_stats['avg_human_score']
        })
    
    df_categories = pd.DataFrame(category_data)
    df_categories = df_categories.sort_values('Ortalama İnsan Skoru', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cat_scores = px.bar(
            df_categories,
            x='Kategori',
            y='Ortalama İnsan Skoru',
            title='Kategorilere Göre Ortalama İnsan Skoru',
            color='Ortalama İnsan Skoru',
            color_continuous_scale='RdYlGn'
        )
        fig_cat_scores.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cat_scores, width='stretch')
    
    with col2:
        fig_cat_count = px.bar(
            df_categories,
            x='Kategori',
            y='Değerlendirme Sayısı',
            title='Kategorilere Göre Değerlendirme Sayısı',
            color='Değerlendirme Sayısı',
            color_continuous_scale='Blues'
        )
        fig_cat_count.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cat_count, width='stretch')
    
    st.dataframe(df_categories, width='stretch')
    
    # Judge Accuracy Analysis
    st.header("⚖️ LLM-as-Judge Doğruluğu")
    
    judge_metrics = evaluator.evaluate_judge_accuracy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Ortalama Mutlak Hata",
            f"{judge_metrics['mean_absolute_error']:.3f}",
            help="Lower is better (0 = perfect)"
        )
    
    with col2:
        st.metric(
            "Medyan Mutlak Hata",
            f"{judge_metrics['median_absolute_error']:.3f}"
        )
    
    with col3:
        bias = judge_metrics['judge_bias']
        bias_color = "normal" if abs(bias) < 0.1 else "inverse"
        st.metric(
            "Judge Bias",
            f"{bias:+.3f}",
            delta=judge_metrics['bias_interpretation'],
            delta_color=bias_color,
            help="Positive = scores too high, Negative = scores too low"
        )
    
    # Error by correction type
    st.subheader("Düzeltme Tipine Göre Hata")
    
    error_data = []
    for correction_type, error_stats in judge_metrics['error_by_correction_type'].items():
        if error_stats['count'] > 0:
            error_data.append({
                "Düzeltme Tipi": correction_type.title(),
                "Örnek Sayısı": error_stats['count'],
                "Ortalama Hata": error_stats['mean_error']
            })
    
    df_errors = pd.DataFrame(error_data)
    
    fig_errors = px.bar(
        df_errors,
        x='Düzeltme Tipi',
        y='Ortalama Hata',
        title='Düzeltme Tiplerine Göre Ortalama Hata',
        color='Ortalama Hata',
        color_continuous_scale='Reds',
        text='Örnek Sayısı'
    )
    st.plotly_chart(fig_errors, width='stretch')
    
    # High disagreement cases
    if judge_metrics['high_disagreement_cases']:
        st.subheader("⚠️ Yüksek Uyuşmazlık Durumları")
        st.write("LLM-Judge ve insan arasında büyük fark olan durumlar (>0.3):")
        
        for idx, case in enumerate(judge_metrics['high_disagreement_cases'][:5], 1):
            with st.expander(f"#{idx} - {case['test_category']} (Fark: {case['difference']:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**LLM-Judge:**")
                    st.metric("Skor", f"{case['llm_score']:.2f}")
                
                with col2:
                    st.write("**İnsan:**")
                    st.metric("Skor", f"{case['human_score']:.2f}")
                
                st.write("**İnsan Geri Bildirimi:**")
                st.info(case['human_feedback'])
                
                st.write(f"**Test ID:** {case['test_id']}")
        
        # Export button
        if st.button("📥 Tüm Uyuşmazlıkları Dışa Aktar"):
            output_path = evaluator.export_disagreement_cases(threshold=0.3)
            st.success(f"✅ Dışa aktarıldı: {output_path}")
    
    # Calibration Insights
    st.header("🎯 Kalibrasyon İçgörüleri")
    
    insights = evaluator.get_calibration_insights()
    
    if insights['recommendations']:
        st.subheader("💡 Öneriler")
        for idx, recommendation in enumerate(insights['recommendations'], 1):
            st.warning(f"**{idx}. {recommendation['issue']}**")
            st.write(f"✅ **Çözüm:** {recommendation['recommendation']}")
    else:
        st.success("✅ LLM-Judge iyi kalibre edilmiş görünüyor!")
    
    # Fine-tuning readiness
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Eğitim Verisi",
            insights['training_data_available'],
            help="Fine-tuning için kullanılabilir örnek sayısı"
        )
    
    with col2:
        if insights['ready_for_finetuning']:
            st.success("✅ Fine-tuning için hazır! (>50 örnek)")
        else:
            needed = 50 - insights['training_data_available']
            st.warning(f"⚠️ Fine-tuning için {needed} örnek daha gerekli")
    
    # Model comparison with human feedback
    st.header("🤖 Model Karşılaştırması (İnsan Skorları)")
    
    # Get all unique models
    all_annotations = annotation_manager.load_all_annotations()
    models = list(set(ann.model_name for ann in all_annotations))
    
    if len(models) > 1:
        model_comparison = []
        for model in models:
            model_result = evaluator.evaluate_model_with_human_feedback(model)
            if 'error' not in model_result:
                model_comparison.append({
                    "Model": model,
                    "İnsan Onaylı Skor": model_result['human_validated_score'],
                    "Onay Oranı": model_result['correction_breakdown']['approval_rate'],
                    "Red Oranı": model_result['correction_breakdown']['rejection_rate'],
                    "Toplam Değerlendirme": model_result['total_annotations']
                })
        
        if model_comparison:
            df_models = pd.DataFrame(model_comparison)
            df_models = df_models.sort_values('İnsan Onaylı Skor', ascending=False)
            
            # Bar chart
            fig_model_comp = px.bar(
                df_models,
                x='Model',
                y='İnsan Onaylı Skor',
                title='Modellerin İnsan Onaylı Skorları',
                color='İnsan Onaylı Skor',
                color_continuous_scale='Viridis',
                text='İnsan Onaylı Skor'
            )
            fig_model_comp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_model_comp, width='stretch')
            
            # Detailed table
            st.dataframe(df_models, width='stretch')
    else:
        st.info("Birden fazla model değerlendirildiğinde karşılaştırma görünecek.")
    
    # Timeline analysis
    st.header("📅 Zaman Serisi Analizi")
    
    timeline_data = []
    for ann in all_annotations:
        date = ann.timestamp.split('T')[0]
        timeline_data.append({
            "Tarih": date,
            "İnsan Skoru": ann.human_score,
            "LLM Skoru": ann.llm_judge_score,
            "Uyum": 1 - abs(ann.llm_judge_score - ann.human_score)
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Group by date
    df_timeline_grouped = df_timeline.groupby('Tarih').agg({
        'İnsan Skoru': 'mean',
        'LLM Skoru': 'mean',
        'Uyum': 'mean'
    }).reset_index()
    
    fig_timeline = px.line(
        df_timeline_grouped,
        x='Tarih',
        y=['İnsan Skoru', 'LLM Skoru', 'Uyum'],
        title='Günlük Ortalama Skorlar ve Uyum',
        markers=True
    )
    st.plotly_chart(fig_timeline, width='stretch')

    # Export section
    st.header("💾 Veri Dışa Aktarma")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Eğitim Verisi Oluştur", width='stretch'):
            output_path = annotation_manager.export_for_training()
            st.success(f"✅ Oluşturuldu: {output_path}")
            
            with open(output_path, 'r', encoding='utf-8') as f:
                training_data = f.read()
            
            st.download_button(
                label="⬇️ Eğitim Verisini İndir",
                data=training_data,
                file_name=Path(output_path).name,
                mime="application/jsonl"
            )
    
    with col2:
        if st.button("📊cısv Olarak Dışa Aktar", width='stretch'):
            # Convert all annotations to CSV
            csv_data = []
            for ann in all_annotations:
                csv_data.append({
                    "annotation_id": ann.annotation_id,
                    "test_id": ann.test_id,
                    "test_category": ann.test_category,
                    "model_name": ann.model_name,
                    "llm_judge_score": ann.llm_judge_score,
                    "human_score": ann.human_score,
                    "correction_type": ann.correction_type,
                    "annotator_id": ann.annotator_id,
                    "timestamp": ann.timestamp,
                    "score_difference": abs(ann.llm_judge_score - ann.human_score)
                })
            
            df_csv = pd.DataFrame(csv_data)
            csv = df_csv.to_csv(index=False)
            
            st.download_button(
                label="⬇️ CSV İndir",
                data=csv,
                file_name=f"human_annotations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
