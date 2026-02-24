"""
Streamlit Dashboard for Evaluation Results
"""
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse


def load_results(filepath):
    """Load evaluation results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    st.set_page_config(
        page_title="LLM Evaluation Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 LLM Evaluation Dashboard")
    st.markdown("### Türkçe ve Fintech Odaklı Model Değerlendirme")
    
    # File selector
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.json"))
        if report_files:
            selected_file = st.selectbox(
                "Rapor Seçin:",
                report_files,
                format_func=lambda x: x.name
            )
        else:
            st.error("Henüz rapor bulunamadı. Önce evaluation çalıştırın.")
            return
    else:
        st.error("Reports dizini bulunamadı.")
        return
    
    # Load results
    results = load_results(selected_file)
    
    # Display timestamp
    st.info(f"📅 Evaluation Date: {results['timestamp']}")
    
    # Overall Comparison
    st.header("📊 Model Karşılaştırması")
    
    comparison_data = []
    for model_key, data in results['summary']['model_comparison'].items():
        row = {
            "Model": model_key,
            "Overall Score": data['overall_score'],
            "Avg Latency (s)": data['avg_latency'],
            "P95 Latency (s)": data['latency_p95']
        }
        
        # Add score distribution if available
        if 'score_distribution_percentages' in data:
            row.update({
                "Good %": data['score_distribution_percentages'].get('good', 0),
                "Moderate %": data['score_distribution_percentages'].get('moderate', 0),
                "Poor %": data['score_distribution_percentages'].get('poor', 0)
            })
        
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score comparison
        fig_score = px.bar(
            df_comparison,
            x="Model",
            y="Overall Score",
            title="Overall Score Comparison",
            color="Overall Score",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_score, width='stretch')
    
    with col2:
        # Latency comparison
        fig_latency = px.bar(
            df_comparison,
            x="Model",
            y=["Avg Latency (s)", "P95 Latency (s)"],
            title="Latency Comparison",
            barmode="group"
        )
        st.plotly_chart(fig_latency, width='stretch')
    
    # Score distribution visualization
    if 'Good %' in df_comparison.columns:
        st.subheader("📈 Skor Dağılımı (Good/Moderate/Poor)")
        
        # Create stacked bar chart
        fig_distribution = go.Figure()
        
        fig_distribution.add_trace(go.Bar(
            name='Good (0.7-1.0)',
            x=df_comparison['Model'],
            y=df_comparison['Good %'],
            marker_color='#2ecc71'
        ))
        fig_distribution.add_trace(go.Bar(
            name='Moderate (0.3-0.7)',
            x=df_comparison['Model'],
            y=df_comparison['Moderate %'],
            marker_color='#f39c12'
        ))
        fig_distribution.add_trace(go.Bar(
            name='Poor (0-0.3)',
            x=df_comparison['Model'],
            y=df_comparison['Poor %'],
            marker_color='#e74c3c'
        ))
        
        fig_distribution.update_layout(
            barmode='stack',
            title='Score Distribution by Model',
            xaxis_title='Model',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_distribution, width='stretch')
    
    # Detailed metrics table
    st.dataframe(df_comparison, width='stretch')
    
    # Best Performers
    st.header("🏆 En İyi Performans Gösterenler")
    
    best_performers = results['summary'].get('best_performers', {})
    if best_performers:
        cols = st.columns(len(best_performers))
        for idx, (category, data) in enumerate(best_performers.items()):
            with cols[idx]:
                st.metric(
                    label=category.replace('_', ' ').title(),
                    value=data['model'],
                    delta=f"{data['score']:.3f}"
                )
    
    # Per-test breakdown
    st.header("📈 Test Kategorileri Detayı")
    
    selected_model = st.selectbox(
        "Model Seçin:",
        list(results['models'].keys())
    )
    
    model_data = results['models'][selected_model]
    
    # Create tabs for each test
    test_tabs = st.tabs(list(model_data['tests'].keys()))
    
    for idx, (test_name, test_tab) in enumerate(zip(model_data['tests'].keys(), test_tabs)):
        with test_tab:
            test_data = model_data['tests'][test_name]
            
            if 'error' in test_data:
                st.error(f"Error: {test_data['error']}")
                continue
            
            if 'summary' not in test_data:
                st.warning("No summary available")
                continue
            
            summary = test_data['summary']
            
            # Metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Overall Score", f"{summary.get('overall_score', 0):.3f}")
            with metric_cols[1]:
                st.metric("Avg Latency", f"{summary.get('avg_latency', 0):.2f}s")
            with metric_cols[2]:
                st.metric("Total Tests", summary.get('total_tests', 0))
            
            # Detailed scores
            if 'avg_scores' in summary:
                st.subheader("Detailed Scores")
                scores_df = pd.DataFrame([summary['avg_scores']])
                st.dataframe(scores_df, width='stretch')
                
                # Radar chart for scores
                categories = list(summary['avg_scores'].keys())
                values = list(summary['avg_scores'].values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_model
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title=f"{test_name} - Score Breakdown"
                )
                st.plotly_chart(fig_radar, width='stretch')
            
            # Sample results
            if 'results' in test_data and test_data['results']:
                st.subheader("Sample Results")
                with st.expander("Show detailed results"):
                    for result in test_data['results'][:5]:  # Show first 5
                        st.json(result)
    
    # Model comparison radar
    st.header("🎯 Multi-Model Comparison")
    
    # Collect scores for each model
    test_categories = list(model_data['tests'].keys())
    
    fig_multi = go.Figure()
    
    for model_key in results['models'].keys():
        scores = []
        for test_name in test_categories:
            test_data = results['models'][model_key]['tests'].get(test_name, {})
            if 'summary' in test_data:
                scores.append(test_data['summary'].get('overall_score', 0))
            else:
                scores.append(0)
        
        fig_multi.add_trace(go.Scatterpolar(
            r=scores,
            theta=test_categories,
            fill='toself',
            name=model_key
        ))
    
    fig_multi.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="All Models - All Tests"
    )
    st.plotly_chart(fig_multi, width='stretch')
    
    # Download results
    st.header("💾 Export")
    
    # Convert to CSV
    all_data = []
    for model_key, model_data in results['models'].items():
        for test_name, test_data in model_data['tests'].items():
            if 'summary' in test_data:
                row = {
                    'model': model_key,
                    'test': test_name,
                    'overall_score': test_data['summary'].get('overall_score', 0),
                    'avg_latency': test_data['summary'].get('avg_latency', 0),
                }
                if 'avg_scores' in test_data['summary']:
                    row.update(test_data['summary']['avg_scores'])
                all_data.append(row)
    
    df_export = pd.DataFrame(all_data)
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"eval_export_{selected_file.stem}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
