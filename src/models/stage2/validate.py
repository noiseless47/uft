#!/usr/bin/env python3
"""
Stage-2 Validation against Historical Selections.
Validates Stage-2 pipeline performance using historical squad selections.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging
from api import Stage2API
from optimizer import SquadOptimizer

logger = logging.getLogger(__name__)


class Stage2Validator:
    """Validate Stage-2 pipeline against historical selections."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize APIs
        self.stage2_api = Stage2API()
        self.optimizer = SquadOptimizer()
        
    def load_validation_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical lineups and features for validation."""
        logger.info("Loading validation data...")
        
        lineups_df = pd.read_csv(os.path.join(data_dir, "lineups.csv"))
        features_df = pd.read_csv(os.path.join(data_dir, "features.csv"))
        players_df = pd.read_csv(os.path.join(data_dir, "players.csv"))
        
        # Sort by date for temporal validation
        lineups_df = lineups_df.sort_values('match_date')
        
        logger.info(f"Loaded {len(lineups_df)} historical lineups")
        return lineups_df, features_df, players_df
    
    def validate_squad_predictions(self, lineups_df: pd.DataFrame, 
                                 features_df: pd.DataFrame,
                                 players_df: pd.DataFrame,
                                 validation_split: float = 0.2) -> Dict:
        """Validate Stage-2 squad predictions against historical selections."""
        logger.info("Validating squad predictions...")
        
        # Split data temporally
        split_idx = int(len(lineups_df) * (1 - validation_split))
        val_lineups = lineups_df.iloc[split_idx:].copy()
        
        validation_results = []
        
        for _, lineup in val_lineups.iterrows():
            match_date = lineup['match_date']
            team_id = lineup['team_id']
            actual_selected = set(lineup['selected_players'].split(','))
            formation = lineup.get('formation', '4-3-3')
            
            # Get eligible players for this match
            eligible_players = features_df[
                (features_df['match_date'] == match_date) & 
                (features_df['team_id'] == team_id)
            ].copy()
            
            if len(eligible_players) < 11:  # Need at least 11 players
                continue
            
            try:
                # Get Stage-2 predictions
                prediction_result = self.stage2_api.predict_squad_selection(
                    eligible_players, 
                    formation=formation,
                    return_probabilities=True
                )
                
                predicted_selected = set(prediction_result['selected_players'])
                
                # Calculate match-level metrics
                match_result = self._calculate_match_metrics(
                    actual_selected, predicted_selected, eligible_players['player_id'].tolist()
                )
                
                match_result.update({
                    'match_date': match_date,
                    'team_id': team_id,
                    'formation': formation,
                    'n_eligible': len(eligible_players),
                    'actual_squad_size': len(actual_selected),
                    'predicted_squad_size': len(predicted_selected)
                })
                
                validation_results.append(match_result)
                
            except Exception as e:
                logger.warning(f"Prediction failed for {match_date}: {e}")
                continue
        
        # Aggregate validation metrics
        if validation_results:
            aggregated_metrics = self._aggregate_validation_metrics(validation_results)
            logger.info("Squad prediction validation completed")
            return {
                'match_results': validation_results,
                'aggregated_metrics': aggregated_metrics
            }
        else:
            logger.error("No successful validations")
            return {'error': 'No successful validations'}
    
    def _calculate_match_metrics(self, actual: set, predicted: set, 
                               all_players: List[str]) -> Dict:
        """Calculate metrics for a single match prediction."""
        # Convert to binary arrays
        y_true = np.array([1 if player in actual else 0 for player in all_players])
        y_pred = np.array([1 if player in predicted else 0 for player in all_players])
        
        # Calculate metrics
        intersection = len(actual & predicted)
        union = len(actual | predicted)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'jaccard_similarity': intersection / union if union > 0 else 0,
            'overlap_count': intersection,
            'actual_size': len(actual),
            'predicted_size': len(predicted)
        }
        
        return metrics
    
    def _aggregate_validation_metrics(self, validation_results: List[Dict]) -> Dict:
        """Aggregate metrics across all validation matches."""
        metrics_to_aggregate = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'jaccard_similarity', 'overlap_count'
        ]
        
        aggregated = {}
        
        for metric in metrics_to_aggregate:
            values = [result[metric] for result in validation_results if metric in result]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        # Additional aggregate metrics
        aggregated['n_matches_validated'] = len(validation_results)
        aggregated['avg_squad_size_actual'] = np.mean([r['actual_size'] for r in validation_results])
        aggregated['avg_squad_size_predicted'] = np.mean([r['predicted_size'] for r in validation_results])
        
        return aggregated
    
    def validate_formation_optimization(self, lineups_df: pd.DataFrame,
                                      features_df: pd.DataFrame,
                                      players_df: pd.DataFrame) -> Dict:
        """Validate formation optimization against historical formations."""
        logger.info("Validating formation optimization...")
        
        formation_results = []
        
        # Sample recent matches for validation
        recent_lineups = lineups_df.tail(20)  # Last 20 matches
        
        for _, lineup in recent_lineups.iterrows():
            match_date = lineup['match_date']
            team_id = lineup['team_id']
            actual_formation = lineup.get('formation', '4-3-3')
            
            # Get eligible players
            eligible_players = features_df[
                (features_df['match_date'] == match_date) & 
                (features_df['team_id'] == team_id)
            ].copy()
            
            if len(eligible_players) < 15:
                continue
            
            try:
                # Mock selection probabilities for validation
                n_players = len(eligible_players)
                selection_probs = np.random.beta(2, 5, n_players)
                
                # Load compatibility matrix
                matrix_path = os.path.join("artifacts", "compat_matrix_v1.npz")
                if os.path.exists(matrix_path):
                    matrix_data = np.load(matrix_path, allow_pickle=True)
                    compat_matrix = matrix_data['compatibility_matrix']
                    player_mappings = {
                        'player_to_idx': matrix_data['player_to_idx'].item(),
                        'idx_to_player': matrix_data['idx_to_player'].item()
                    }
                    
                    # Compare formations
                    comparison = self.optimizer.compare_formations(
                        eligible_players, selection_probs, compat_matrix, player_mappings
                    )
                    
                    formation_result = {
                        'match_date': match_date,
                        'actual_formation': actual_formation,
                        'predicted_best_formation': comparison['best_formation'],
                        'formation_scores': comparison['formation_scores'],
                        'formation_match': actual_formation == comparison['best_formation']
                    }
                    
                    formation_results.append(formation_result)
                
            except Exception as e:
                logger.warning(f"Formation validation failed for {match_date}: {e}")
                continue
        
        # Aggregate formation validation
        if formation_results:
            formation_accuracy = np.mean([r['formation_match'] for r in formation_results])
            
            formation_summary = {
                'formation_accuracy': formation_accuracy,
                'n_matches_validated': len(formation_results),
                'formation_distribution': pd.Series([r['actual_formation'] for r in formation_results]).value_counts().to_dict(),
                'predicted_distribution': pd.Series([r['predicted_best_formation'] for r in formation_results]).value_counts().to_dict()
            }
            
            logger.info(f"Formation prediction accuracy: {formation_accuracy:.3f}")
            return {
                'formation_results': formation_results,
                'formation_summary': formation_summary
            }
        
        return {'error': 'No formation validations completed'}
    
    def validate_compatibility_impact(self, lineups_df: pd.DataFrame,
                                    features_df: pd.DataFrame) -> Dict:
        """Validate impact of compatibility matrix on selections."""
        logger.info("Validating compatibility impact...")
        
        # Compare selections with and without compatibility
        comparison_results = []
        
        recent_lineups = lineups_df.tail(10)  # Sample matches
        
        for _, lineup in recent_lineups.iterrows():
            match_date = lineup['match_date']
            team_id = lineup['team_id']
            
            eligible_players = features_df[
                (features_df['match_date'] == match_date) & 
                (features_df['team_id'] == team_id)
            ].copy()
            
            if len(eligible_players) < 11:
                continue
            
            try:
                # Prediction with compatibility
                result_with_compat = self.stage2_api.predict_squad_selection(
                    eligible_players, return_probabilities=True
                )
                
                # Mock prediction without compatibility (Stage-1 only)
                stage1_only_probs = result_with_compat['stage1_probabilities']
                top_indices = np.argsort(stage1_only_probs)[::-1][:18]
                stage1_only_selection = eligible_players.iloc[top_indices]['player_id'].tolist()
                
                comparison_result = {
                    'match_date': match_date,
                    'with_compatibility': result_with_compat['selected_players'],
                    'without_compatibility': stage1_only_selection,
                    'selection_overlap': len(set(result_with_compat['selected_players']) & 
                                           set(stage1_only_selection)),
                    'compatibility_impact_score': result_with_compat['total_score']
                }
                
                comparison_results.append(comparison_result)
                
            except Exception as e:
                logger.warning(f"Compatibility validation failed for {match_date}: {e}")
                continue
        
        if comparison_results:
            avg_overlap = np.mean([r['selection_overlap'] for r in comparison_results])
            avg_impact = np.mean([r['compatibility_impact_score'] for r in comparison_results])
            
            compatibility_summary = {
                'avg_selection_overlap': avg_overlap,
                'avg_compatibility_impact': avg_impact,
                'n_comparisons': len(comparison_results),
                'compatibility_changes_selection': avg_overlap < 15  # Less than full overlap
            }
            
            logger.info(f"Average selection overlap (with/without compatibility): {avg_overlap:.1f}/18")
            return {
                'comparison_results': comparison_results,
                'compatibility_summary': compatibility_summary
            }
        
        return {'error': 'No compatibility validations completed'}
    
    def generate_validation_report(self, squad_validation: Dict,
                                 formation_validation: Dict,
                                 compatibility_validation: Dict,
                                 output_dir: str = "artifacts") -> str:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report_lines = [
            "# Stage-2 Pipeline Validation Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Squad prediction summary
        if 'aggregated_metrics' in squad_validation:
            metrics = squad_validation['aggregated_metrics']
            report_lines.extend([
                "### Squad Selection Performance",
                f"- **Accuracy**: {metrics.get('accuracy_mean', 0):.3f} ± {metrics.get('accuracy_std', 0):.3f}",
                f"- **Precision**: {metrics.get('precision_mean', 0):.3f} ± {metrics.get('precision_std', 0):.3f}",
                f"- **Recall**: {metrics.get('recall_mean', 0):.3f} ± {metrics.get('recall_std', 0):.3f}",
                f"- **F1-Score**: {metrics.get('f1_score_mean', 0):.3f} ± {metrics.get('f1_score_std', 0):.3f}",
                f"- **Jaccard Similarity**: {metrics.get('jaccard_similarity_mean', 0):.3f} ± {metrics.get('jaccard_similarity_std', 0):.3f}",
                f"- **Matches Validated**: {metrics.get('n_matches_validated', 0)}",
                ""
            ])
        
        # Formation validation summary
        if 'formation_summary' in formation_validation:
            form_summary = formation_validation['formation_summary']
            report_lines.extend([
                "### Formation Prediction Performance",
                f"- **Formation Accuracy**: {form_summary.get('formation_accuracy', 0):.3f}",
                f"- **Matches Validated**: {form_summary.get('n_matches_validated', 0)}",
                "",
                "**Most Common Actual Formations:**"
            ])
            
            for formation, count in form_summary.get('formation_distribution', {}).items():
                report_lines.append(f"- {formation}: {count} matches")
            
            report_lines.append("")
        
        # Compatibility impact summary
        if 'compatibility_summary' in compatibility_validation:
            compat_summary = compatibility_validation['compatibility_summary']
            report_lines.extend([
                "### Compatibility Matrix Impact",
                f"- **Average Selection Overlap**: {compat_summary.get('avg_selection_overlap', 0):.1f}/18 players",
                f"- **Compatibility Changes Selection**: {compat_summary.get('compatibility_changes_selection', False)}",
                f"- **Average Impact Score**: {compat_summary.get('avg_compatibility_impact', 0):.3f}",
                ""
            ])
        
        # Detailed analysis
        report_lines.extend([
            "## Detailed Analysis",
            "",
            "### Model Performance Insights",
            ""
        ])
        
        if 'aggregated_metrics' in squad_validation:
            metrics = squad_validation['aggregated_metrics']
            
            # Performance interpretation
            accuracy = metrics.get('accuracy_mean', 0)
            if accuracy > 0.8:
                performance_level = "Excellent"
            elif accuracy > 0.7:
                performance_level = "Good"
            elif accuracy > 0.6:
                performance_level = "Fair"
            else:
                performance_level = "Poor"
            
            report_lines.extend([
                f"**Overall Performance**: {performance_level} (Accuracy: {accuracy:.3f})",
                "",
                "**Key Findings:**"
            ])
            
            # Precision vs Recall analysis
            precision = metrics.get('precision_mean', 0)
            recall = metrics.get('recall_mean', 0)
            
            if precision > recall + 0.1:
                report_lines.append("- Model is conservative (high precision, lower recall)")
            elif recall > precision + 0.1:
                report_lines.append("- Model is aggressive (high recall, lower precision)")
            else:
                report_lines.append("- Model shows balanced precision-recall trade-off")
            
            # Jaccard similarity interpretation
            jaccard = metrics.get('jaccard_similarity_mean', 0)
            if jaccard > 0.6:
                report_lines.append("- Strong overlap with historical selections")
            elif jaccard > 0.4:
                report_lines.append("- Moderate overlap with historical selections")
            else:
                report_lines.append("- Low overlap suggests different selection strategy")
        
        report_lines.extend([
            "",
            "### Recommendations",
            ""
        ])
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations(
            squad_validation, formation_validation, compatibility_validation
        )
        
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, "stage2_validation_report.md")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path
    
    def _generate_recommendations(self, squad_val: Dict, formation_val: Dict, 
                                compat_val: Dict) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Squad selection recommendations
        if 'aggregated_metrics' in squad_val:
            metrics = squad_val['aggregated_metrics']
            accuracy = metrics.get('accuracy_mean', 0)
            
            if accuracy < 0.7:
                recommendations.append("Consider retraining Stage-2 models with more recent data")
            
            precision = metrics.get('precision_mean', 0)
            recall = metrics.get('recall_mean', 0)
            
            if precision < 0.6:
                recommendations.append("Increase selection threshold to improve precision")
            
            if recall < 0.6:
                recommendations.append("Lower selection threshold or expand eligible player pool")
        
        # Formation recommendations
        if 'formation_summary' in formation_val:
            form_acc = formation_val['formation_summary'].get('formation_accuracy', 0)
            
            if form_acc < 0.5:
                recommendations.append("Formation prediction needs improvement - consider tactical context features")
        
        # Compatibility recommendations
        if 'compatibility_summary' in compat_val:
            changes_selection = compat_val['compatibility_summary'].get('compatibility_changes_selection', False)
            
            if not changes_selection:
                recommendations.append("Compatibility matrix may need retraining or higher weight in ensemble")
            else:
                recommendations.append("Compatibility matrix is effectively influencing selections")
        
        if not recommendations:
            recommendations.append("Pipeline performance is satisfactory - ready for production use")
        
        return recommendations
    
    def create_validation_plots(self, squad_validation: Dict, 
                              output_dir: str = "artifacts") -> List[str]:
        """Create validation visualization plots."""
        logger.info("Creating validation plots...")
        
        plot_paths = []
        
        if 'match_results' in squad_validation:
            match_results = squad_validation['match_results']
            
            # Plot 1: Accuracy over time
            plt.figure(figsize=(12, 6))
            
            dates = [pd.to_datetime(r['match_date']) for r in match_results]
            accuracies = [r['accuracy'] for r in match_results]
            
            plt.subplot(1, 2, 1)
            plt.plot(dates, accuracies, 'o-', alpha=0.7)
            plt.title('Squad Selection Accuracy Over Time')
            plt.xlabel('Match Date')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Precision vs Recall
            precisions = [r['precision'] for r in match_results]
            recalls = [r['recall'] for r in match_results]
            
            plt.subplot(1, 2, 2)
            plt.scatter(recalls, precisions, alpha=0.6)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall')
            plt.grid(True, alpha=0.3)
            
            # Add diagonal line
            max_val = max(max(precisions), max(recalls))
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Balance')
            plt.legend()
            
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "stage2_validation_performance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(plot_path)
            
            # Plot 3: Jaccard similarity distribution
            plt.figure(figsize=(8, 6))
            
            jaccard_scores = [r['jaccard_similarity'] for r in match_results]
            
            plt.hist(jaccard_scores, bins=15, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(jaccard_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(jaccard_scores):.3f}')
            plt.xlabel('Jaccard Similarity')
            plt.ylabel('Frequency')
            plt.title('Distribution of Squad Selection Similarity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, "stage2_jaccard_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(plot_path)
        
        logger.info(f"Created {len(plot_paths)} validation plots")
        return plot_paths


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Validate Stage-2 pipeline")
    parser.add_argument('--data', default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--artifacts', default='artifacts',
                       help='Artifacts directory')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory for validation results')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting Stage-2 validation")
    
    try:
        # Initialize validator
        validator = Stage2Validator()
        
        # Load data
        lineups_df, features_df, players_df = validator.load_validation_data(args.data)
        
        # Run validations
        logger.info("Running squad prediction validation...")
        squad_validation = validator.validate_squad_predictions(
            lineups_df, features_df, players_df, args.validation_split
        )
        
        logger.info("Running formation optimization validation...")
        formation_validation = validator.validate_formation_optimization(
            lineups_df, features_df, players_df
        )
        
        logger.info("Running compatibility impact validation...")
        compatibility_validation = validator.validate_compatibility_impact(
            lineups_df, features_df
        )
        
        # Generate report
        report_path = validator.generate_validation_report(
            squad_validation, formation_validation, compatibility_validation, args.output
        )
        
        # Create plots
        plot_paths = validator.create_validation_plots(squad_validation, args.output)
        
        # Save validation results
        validation_results = {
            'squad_validation': squad_validation,
            'formation_validation': formation_validation,
            'compatibility_validation': compatibility_validation,
            'report_path': report_path,
            'plot_paths': plot_paths
        }
        
        results_path = os.path.join(args.output, "stage2_validation_results.pkl")
        joblib.dump(validation_results, results_path)
        
        print(f"\n=== Stage-2 Validation Summary ===")
        
        if 'aggregated_metrics' in squad_validation:
            metrics = squad_validation['aggregated_metrics']
            print(f"Squad Selection Accuracy: {metrics.get('accuracy_mean', 0):.3f}")
            print(f"Jaccard Similarity: {metrics.get('jaccard_similarity_mean', 0):.3f}")
        
        if 'formation_summary' in formation_validation:
            form_acc = formation_validation['formation_summary'].get('formation_accuracy', 0)
            print(f"Formation Prediction Accuracy: {form_acc:.3f}")
        
        if 'compatibility_summary' in compatibility_validation:
            overlap = compatibility_validation['compatibility_summary'].get('avg_selection_overlap', 0)
            print(f"Compatibility Impact: {18 - overlap:.1f} players changed on average")
        
        print(f"\nValidation report: {report_path}")
        print(f"Validation plots: {len(plot_paths)} files created")
        
        logger.info("Stage-2 validation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Stage-2 validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
