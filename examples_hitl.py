"""
Quick Start Example for Human-in-the-Loop System
Demonstrates basic usage of the HITL features
"""
from utils.human_annotations import (
    AnnotationManager,
    HumanAnnotation,
    create_pending_from_results
)
from evaluators import HumanFeedbackEvaluator
from datetime import datetime


def example_1_create_pending_items():
    """Example 1: Create pending review items from evaluation results"""
    print("=" * 60)
    print("Example 1: Creating Pending Review Items")
    print("=" * 60)
    
    annotation_manager = AnnotationManager()
    
    # Load latest evaluation results
    import glob
    reports = glob.glob("reports/*.json")
    if not reports:
        print("⚠️  No evaluation reports found. Run evaluation first.")
        return
    
    latest_report = max(reports, key=lambda x: x.split('_')[-1])
    print(f"📄 Loading report: {latest_report}")
    
    # Create pending items
    added_count = create_pending_from_results(
        latest_report,
        annotation_manager,
        sample_per_test=3
    )
    
    print(f"✅ Added {added_count} items for human review")


def example_2_programmatic_annotation():
    """Example 2: Create annotations programmatically"""
    print("\n" + "=" * 60)
    print("Example 2: Creating Annotations Programmatically")
    print("=" * 60)
    
    annotation_manager = AnnotationManager()
    
    # Create a sample annotation
    annotation = HumanAnnotation(
        annotation_id=annotation_manager.generate_annotation_id("test_001", "gpt35_turbo"),
        test_id="test_001",
        test_category="turkish_grammar",
        model_name="gpt35_turbo",
        question="Türkiye'nin başkenti nedir?",
        model_response="Türkiye'nin başkenti Ankara'dır.",
        llm_judge_score=0.95,
        llm_judge_reasoning="Doğru ve eksiksiz cevap",
        human_score=1.0,
        human_feedback="Mükemmel cevap, gereksiz bilgi yok, doğru format",
        correction_type="adjust",
        annotator_id="demo_user",
        timestamp=datetime.now().isoformat(),
        metadata={"latency": 0.5, "cost": 0.0001}
    )
    
    # Save annotation
    annotation_manager.save_annotation(annotation)
    print(f"✅ Saved annotation: {annotation.annotation_id}")


def example_3_load_and_analyze():
    """Example 3: Load annotations and get statistics"""
    print("\n" + "=" * 60)
    print("Example 3: Loading and Analyzing Annotations")
    print("=" * 60)
    
    annotation_manager = AnnotationManager()
    
    # Get statistics
    stats = annotation_manager.get_statistics()
    
    print(f"\n📊 Statistics:")
    print(f"   Total completed: {stats['total_completed']}")
    print(f"   Total pending: {stats['total_pending']}")
    
    if stats['total_completed'] > 0:
        print(f"   Average agreement: {stats['average_agreement']:.2%}")
        print(f"\n   Corrections by type:")
        for corr_type, count in stats['corrections_by_type'].items():
            print(f"      {corr_type}: {count}")
        
        print(f"\n   By category:")
        for category, cat_stats in stats['by_category'].items():
            print(f"      {category}: {cat_stats['count']} annotations, "
                  f"avg score: {cat_stats['avg_human_score']:.2f}")


def example_4_evaluate_model():
    """Example 4: Evaluate model using human feedback"""
    print("\n" + "=" * 60)
    print("Example 4: Evaluating Model with Human Feedback")
    print("=" * 60)
    
    evaluator = HumanFeedbackEvaluator()
    
    # Evaluate a model
    result = evaluator.evaluate_model_with_human_feedback(
        model_name="gpt35_turbo",
        test_category=None  # All categories
    )
    
    if 'error' in result:
        print(f"⚠️  {result['error']}")
        return
    
    print(f"\n🤖 Model: {result['model_name']}")
    print(f"   Human-validated score: {result['human_validated_score']:.3f}")
    print(f"   Total annotations: {result['total_annotations']}")
    print(f"\n   Correction breakdown:")
    print(f"      Approved: {result['correction_breakdown']['approved']} "
          f"({result['correction_breakdown']['approval_rate']:.1%})")
    print(f"      Adjusted: {result['correction_breakdown']['adjusted']} "
          f"({result['correction_breakdown']['adjustment_rate']:.1%})")
    print(f"      Rejected: {result['correction_breakdown']['rejected']} "
          f"({result['correction_breakdown']['rejection_rate']:.1%})")


def example_5_judge_accuracy():
    """Example 5: Evaluate LLM-as-Judge accuracy"""
    print("\n" + "=" * 60)
    print("Example 5: Evaluating LLM-as-Judge Accuracy")
    print("=" * 60)
    
    evaluator = HumanFeedbackEvaluator()
    
    # Evaluate judge
    result = evaluator.evaluate_judge_accuracy()
    
    if 'error' in result:
        print(f"⚠️  {result['error']}")
        return
    
    print(f"\n⚖️  LLM-as-Judge Metrics:")
    print(f"   Total comparisons: {result['total_comparisons']}")
    print(f"   Average agreement: {result['average_agreement']:.2%}")
    print(f"   Mean absolute error: {result['mean_absolute_error']:.3f}")
    print(f"   Median absolute error: {result['median_absolute_error']:.3f}")
    print(f"   Judge bias: {result['judge_bias']:+.3f}")
    print(f"   Interpretation: {result['bias_interpretation']}")
    
    print(f"\n   Error by correction type:")
    for corr_type, stats in result['error_by_correction_type'].items():
        if stats['count'] > 0:
            print(f"      {corr_type}: count={stats['count']}, "
                  f"mean_error={stats['mean_error']:.3f}")
    
    if result['high_disagreement_cases']:
        print(f"\n   ⚠️  {len(result['high_disagreement_cases'])} high disagreement cases found")


def example_6_calibration_insights():
    """Example 6: Get calibration insights and recommendations"""
    print("\n" + "=" * 60)
    print("Example 6: Getting Calibration Insights")
    print("=" * 60)
    
    evaluator = HumanFeedbackEvaluator()
    
    # Get insights
    insights = evaluator.get_calibration_insights()
    
    if 'error' in insights:
        print(f"⚠️  {insights['error']}")
        return
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Average agreement: {insights['overall_metrics']['average_agreement']:.2%}")
    print(f"   Mean absolute error: {insights['overall_metrics']['mean_absolute_error']:.3f}")
    print(f"   Judge bias: {insights['overall_metrics']['judge_bias']:+.3f}")
    
    print(f"\n💾 Training Data:")
    print(f"   Available: {insights['training_data_available']} examples")
    print(f"   Ready for fine-tuning: {'✅ Yes' if insights['ready_for_finetuning'] else '❌ No (need 50+)'}")
    
    if insights['recommendations']:
        print(f"\n💡 Recommendations:")
        for idx, rec in enumerate(insights['recommendations'], 1):
            print(f"   {idx}. Issue: {rec['issue']}")
            print(f"      Fix: {rec['recommendation']}")
    else:
        print("\n✅ No issues found - Judge is well calibrated!")


def example_7_export_training_data():
    """Example 7: Export training data for fine-tuning"""
    print("\n" + "=" * 60)
    print("Example 7: Exporting Training Data")
    print("=" * 60)
    
    annotation_manager = AnnotationManager()
    
    # Check if enough data
    stats = annotation_manager.get_statistics()
    if stats['total_completed'] < 10:
        print(f"⚠️  Only {stats['total_completed']} annotations. Need at least 10 for export.")
        return
    
    # Export
    output_path = annotation_manager.export_for_training(
        min_agreement_threshold=0.2  # Export cases with <0.8 agreement
    )
    
    print(f"✅ Training data exported to: {output_path}")
    
    # Count examples
    import json
    with open(output_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"   Total examples: {len(examples)}")
    print(f"   Ready for OpenAI fine-tuning format")


def example_8_export_disagreements():
    """Example 8: Export high disagreement cases for analysis"""
    print("\n" + "=" * 60)
    print("Example 8: Exporting Disagreement Cases")
    print("=" * 60)
    
    evaluator = HumanFeedbackEvaluator()
    
    # Export disagreements
    output_path = evaluator.export_disagreement_cases(
        threshold=0.3,  # 0.3+ difference
        output_file="disagreement_analysis.json"
    )
    
    print(f"✅ Disagreement cases exported to: {output_path}")
    
    # Load and show summary
    import json
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Total disagreements: {data['total_disagreements']}")
    print(f"   Threshold: {data['threshold']}")
    
    if data['cases']:
        print(f"\n   Sample case:")
        case = data['cases'][0]
        print(f"      Test: {case['test_category']}")
        print(f"      LLM score: {case['llm_judge']['score']:.2f}")
        print(f"      Human score: {case['human']['score']:.2f}")
        print(f"      Direction: {case['direction']}")


def main():
    """Run all examples"""
    print("\n" + "🚀" * 30)
    print("Human-in-the-Loop Quick Start Examples")
    print("🚀" * 30 + "\n")
    
    try:
        # Run examples
        example_1_create_pending_items()
        example_2_programmatic_annotation()
        example_3_load_and_analyze()
        example_4_evaluate_model()
        example_5_judge_accuracy()
        example_6_calibration_insights()
        example_7_export_training_data()
        example_8_export_disagreements()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
        print("\n📚 Next Steps:")
        print("   1. Open the Streamlit dashboard: streamlit run dashboard.py")
        print("   2. Go to 'Human Review' page to annotate")
        print("   3. Check 'HITL Analytics' page for insights")
        print("   4. Export training data when you have 50+ annotations")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
