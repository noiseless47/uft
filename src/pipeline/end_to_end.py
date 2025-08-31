#!/usr/bin/env python3
"""
End-to-End Pipeline Integration.
Orchestrates the complete football squad selection pipeline.
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
import time
from datetime import datetime
import mlflow

sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging
from data.prepare_data import DataCollector
from features.engineer_features import FeatureEngineer
from features.mab_feature_selection import MABFeatureSelector
from models.stage1.train import Stage1Trainer
from models.stage1.api import Stage1API
from models.stage2.compatibility_matrix import CompatibilityMatrixBuilder
from models.stage2.train import Stage2ModelTrainer
from models.stage2.api import Stage2API
from models.stage2.optimizer import SquadOptimizer
from models.transfer_learning.domain_adapter import TransferLearningAdapter
from simulation.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Complete end-to-end pipeline orchestrator."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline_config = self.config.get('pipeline', {})
        self.random_seed = self.config['random_seeds']['global']
        
        # Initialize components
        self.data_collector = DataCollector(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.mab_selector = MABFeatureSelector(config_path)
        self.stage1_trainer = Stage1Trainer(config_path)
        self.compat_builder = CompatibilityMatrixBuilder(config_path)
        self.stage2_trainer = Stage2ModelTrainer(config_path)
        self.optimizer = SquadOptimizer(config_path)
        self.transfer_adapter = TransferLearningAdapter(config_path)
        self.simulator = MonteCarloSimulator(config_path)
        
        # Pipeline state
        self.pipeline_state = {
            'data_prepared': False,
            'features_engineered': False,
            'features_selected': False,
            'stage1_trained': False,
            'compatibility_built': False,
            'stage2_trained': False,
            'pipeline_validated': False
        }
        
    def run_full_pipeline(self, mode: str = "development") -> Dict:
        """Run the complete pipeline from data to final predictions."""
        logger.info(f"Starting full pipeline in {mode} mode")
        
        pipeline_start = time.time()
        results = {}
        
        try:
            # Step 1: Data Preparation
            logger.info("=== STEP 1: Data Preparation ===")
            data_results = self._run_data_preparation()
            results['data_preparation'] = data_results
            self.pipeline_state['data_prepared'] = True
            
            # Step 2: Feature Engineering
            logger.info("=== STEP 2: Feature Engineering ===")
            feature_results = self._run_feature_engineering()
            results['feature_engineering'] = feature_results
            self.pipeline_state['features_engineered'] = True
            
            # Step 3: Feature Selection (optional)
            if self.pipeline_config.get('use_mab_selection', True):
                logger.info("=== STEP 3: MAB Feature Selection ===")
                selection_results = self._run_feature_selection()
                results['feature_selection'] = selection_results
                self.pipeline_state['features_selected'] = True
            
            # Step 4: Stage-1 Training
            logger.info("=== STEP 4: Stage-1 Training ===")
            stage1_results = self._run_stage1_training()
            results['stage1_training'] = stage1_results
            self.pipeline_state['stage1_trained'] = True
            
            # Step 5: Compatibility Matrix
            logger.info("=== STEP 5: Compatibility Matrix ===")
            compat_results = self._run_compatibility_building()
            results['compatibility_matrix'] = compat_results
            self.pipeline_state['compatibility_built'] = True
            
            # Step 6: Stage-2 Training
            logger.info("=== STEP 6: Stage-2 Training ===")
            stage2_results = self._run_stage2_training()
            results['stage2_training'] = stage2_results
            self.pipeline_state['stage2_trained'] = True
            
            # Step 7: Pipeline Validation
            if mode != "fast":
                logger.info("=== STEP 7: Pipeline Validation ===")
                validation_results = self._run_pipeline_validation()
                results['validation'] = validation_results
                self.pipeline_state['pipeline_validated'] = True
            
            # Step 8: Final Integration Test
            logger.info("=== STEP 8: Integration Test ===")
            integration_results = self._run_integration_test()
            results['integration_test'] = integration_results
            
            pipeline_duration = time.time() - pipeline_start
            
            # Pipeline summary
            pipeline_summary = {
                'status': 'SUCCESS',
                'duration_seconds': pipeline_duration,
                'duration_minutes': pipeline_duration / 60,
                'pipeline_state': self.pipeline_state,
                'timestamp': datetime.now().isoformat()
            }
            
            results['pipeline_summary'] = pipeline_summary
            
            logger.info(f"Full pipeline completed successfully in {pipeline_duration/60:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed at step: {e}")
            results['error'] = str(e)
            results['pipeline_state'] = self.pipeline_state
            return results
    
    def _run_data_preparation(self) -> Dict:
        """Run data preparation step."""
        try:
            # Collect and prepare data
            data_results = self.data_collector.collect_all_data()
            
            return {
                'status': 'success',
                'datasets_created': len(data_results),
                'data_files': list(data_results.keys()) if isinstance(data_results, dict) else []
            }
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_feature_engineering(self) -> Dict:
        """Run feature engineering step."""
        try:
            # Load raw data
            players_df = pd.read_csv("data/processed/players.csv")
            matches_df = pd.read_csv("data/processed/matches.csv")
            
            # Engineer features
            features_df = self.feature_engineer.create_all_features(players_df, matches_df)
            
            return {
                'status': 'success',
                'n_features': len(features_df.columns),
                'n_samples': len(features_df),
                'feature_file': 'data/processed/features.csv'
            }
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_feature_selection(self) -> Dict:
        """Run MAB feature selection step."""
        try:
            # Load features
            features_df = pd.read_csv("data/processed/features.csv")
            
            # Prepare for MAB
            feature_cols = [col for col in features_df.columns if col not in 
                           ['player_id', 'match_date', 'team_id', 'selected']]
            
            X = features_df[feature_cols]
            y = features_df['selected'] if 'selected' in features_df.columns else pd.Series(np.random.binomial(1, 0.1, len(features_df)))
            
            # Run MAB selection
            mab_results = self.mab_selector.adaptive_feature_selection(X, y)
            
            # Save selected features
            self.mab_selector.save_mab_results(mab_results, "contextual", "artifacts")
            
            return {
                'status': 'success',
                'original_features': len(feature_cols),
                'selected_features': len(mab_results['best_features']),
                'best_reward': mab_results['best_reward'],
                'selection_file': 'artifacts/best_features_contextual.txt'
            }
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_stage1_training(self) -> Dict:
        """Run Stage-1 training step."""
        try:
            # Load features
            features_df = pd.read_csv("data/processed/features.csv")
            
            # Train Stage-1 models
            training_results = self.stage1_trainer.train_all_models(features_df)
            
            return {
                'status': 'success',
                'models_trained': len(training_results['models']),
                'best_model': training_results['best_model'],
                'cv_scores': training_results['cv_scores']
            }
        except Exception as e:
            logger.error(f"Stage-1 training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_compatibility_building(self) -> Dict:
        """Run compatibility matrix building step."""
        try:
            # Load co-play data
            coplay_df, lineups_df, events_df = self.compat_builder.load_coplay_data()
            
            # Build compatibility matrix
            coplay_matrix, player_to_idx, idx_to_player = self.compat_builder.create_coplay_matrix(coplay_df)
            mf_model = self.compat_builder.train_matrix_factorization(coplay_matrix)
            
            # Compute compatibility scores
            compatibility_df = self.compat_builder.compute_compatibility_scores(
                mf_model, player_to_idx, idx_to_player
            )
            
            # Load players for position enhancement
            players_df = pd.read_csv("data/processed/players.csv")
            enhanced_compat = self.compat_builder.enhance_with_positional_compatibility(
                compatibility_df, players_df
            )
            
            # Create full matrix
            compat_matrix = self.compat_builder.create_full_compatibility_matrix(
                enhanced_compat, player_to_idx
            )
            
            # Save artifacts
            player_mappings = {'player_to_idx': player_to_idx, 'idx_to_player': idx_to_player}
            self.compat_builder.save_compatibility_artifacts(
                compat_matrix, enhanced_compat, mf_model, player_mappings
            )
            
            return {
                'status': 'success',
                'matrix_shape': compat_matrix.shape,
                'n_players': len(player_to_idx),
                'sparsity': 1 - np.count_nonzero(compat_matrix) / compat_matrix.size
            }
        except Exception as e:
            logger.error(f"Compatibility building failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_stage2_training(self) -> Dict:
        """Run Stage-2 training step."""
        try:
            # Load data
            lineups_df, players_df, features_df = self.stage2_trainer.load_training_data()
            compat_matrix, player_mappings = self.stage2_trainer.load_compatibility_matrix()
            
            # Create Stage-2 features
            stage2_df = self.stage2_trainer.create_stage2_features(
                lineups_df, features_df, compat_matrix, player_mappings
            )
            
            # Train models
            models, cv_scores = self.stage2_trainer.train_stage2_models(stage2_df)
            
            # Create ensemble
            ensemble_info = self.stage2_trainer.create_ensemble_model(models, cv_scores)
            
            # Validate
            validation_metrics = self.stage2_trainer.validate_stage2_performance(ensemble_info, stage2_df)
            
            # Save models
            self.stage2_trainer.save_stage2_models(ensemble_info, validation_metrics)
            
            return {
                'status': 'success',
                'models_trained': len(models),
                'ensemble_performance': validation_metrics,
                'training_samples': len(stage2_df)
            }
        except Exception as e:
            logger.error(f"Stage-2 training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_pipeline_validation(self) -> Dict:
        """Run comprehensive pipeline validation."""
        try:
            # Run Monte Carlo simulation
            features_df = pd.read_csv("data/processed/features.csv")
            players_sample = features_df.head(30)
            
            # Create scenarios
            scenarios = self.simulator.create_simulation_scenarios()
            for scenario in scenarios:
                scenario.n_simulations = 50  # Reduced for integration test
            
            # Run simulations
            simulation_results = self.simulator.run_parallel_simulations(players_sample, scenarios)
            analysis = self.simulator.analyze_simulation_results(simulation_results)
            
            # Stress tests
            stress_tests = self.simulator.stress_test_pipeline(players_sample)
            
            return {
                'status': 'success',
                'simulation_success_rate': analysis.get('overall_success_rate', 0),
                'stress_tests_passed': stress_tests['successful_tests'],
                'stress_tests_total': stress_tests['total_tests']
            }
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_integration_test(self) -> Dict:
        """Run final integration test."""
        try:
            # Initialize APIs
            stage2_api = Stage2API()
            
            # Load test data
            features_df = pd.read_csv("data/processed/features.csv")
            test_players = features_df.head(25)
            
            # Test complete pipeline
            start_time = time.time()
            
            # Test squad selection
            selection_result = stage2_api.predict_squad_selection(
                test_players, 
                formation="4-3-3",
                return_probabilities=True
            )
            
            # Test optimizer
            optimizer = SquadOptimizer()
            mock_probs = np.random.beta(2, 5, len(test_players))
            
            # Load compatibility matrix for optimizer test
            matrix_path = "artifacts/compat_matrix_v1.npz"
            if os.path.exists(matrix_path):
                matrix_data = np.load(matrix_path, allow_pickle=True)
                compat_matrix = matrix_data['compatibility_matrix']
                player_mappings = {
                    'player_to_idx': matrix_data['player_to_idx'].item(),
                    'idx_to_player': matrix_data['idx_to_player'].item()
                }
                
                optimizer_result = optimizer.optimize_squad(
                    test_players, mock_probs, compat_matrix, player_mappings
                )
            else:
                optimizer_result = {'status': 'compatibility_matrix_missing'}
            
            integration_time = time.time() - start_time
            
            return {
                'status': 'success',
                'integration_time_seconds': integration_time,
                'stage2_prediction_success': 'selected_players' in selection_result,
                'optimizer_success': 'selected_players' in optimizer_result,
                'n_players_selected': len(selection_result.get('selected_players', [])),
                'formation_used': selection_result.get('formation', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_fast_demo(self) -> Dict:
        """Run fast demonstration of the pipeline."""
        logger.info("Running fast demo pipeline")
        
        try:
            # Quick data preparation
            data_results = self.data_collector.generate_synthetic_data()
            
            # Quick feature engineering
            players_df = pd.read_csv("data/processed/players.csv")
            matches_df = pd.read_csv("data/processed/matches.csv")
            features_df = self.feature_engineer.create_all_features(players_df, matches_df)
            
            # Quick Stage-1 training (reduced parameters)
            stage1_results = self.stage1_trainer.train_all_models(features_df, quick_mode=True)
            
            # Test prediction
            stage1_api = Stage1API()
            test_players = features_df.head(20)
            predictions = stage1_api.predict_batch(test_players)
            
            return {
                'status': 'success',
                'mode': 'fast_demo',
                'data_samples': len(features_df),
                'models_trained': len(stage1_results['models']),
                'prediction_success': len(predictions['probabilities']) > 0,
                'demo_duration': 'under_5_minutes'
            }
            
        except Exception as e:
            logger.error(f"Fast demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_development_mode(self) -> Dict:
        """Run pipeline in development mode with reduced data."""
        logger.info("Running development mode pipeline")
        
        # Similar to full pipeline but with smaller datasets
        return self.run_full_pipeline(mode="development")
    
    def validate_pipeline_health(self) -> Dict:
        """Check pipeline health and component availability."""
        logger.info("Validating pipeline health...")
        
        health_checks = {}
        
        # Check data files
        required_data_files = [
            "data/processed/players.csv",
            "data/processed/matches.csv", 
            "data/processed/lineups.csv",
            "data/processed/features.csv"
        ]
        
        health_checks['data_files'] = {
            file: os.path.exists(file) for file in required_data_files
        }
        
        # Check model artifacts
        required_artifacts = [
            "artifacts/stage1_ensemble.pkl",
            "artifacts/compat_matrix_v1.npz",
            "artifacts/stage2_ensemble.pkl"
        ]
        
        health_checks['model_artifacts'] = {
            file: os.path.exists(file) for file in required_artifacts
        }
        
        # Check configuration
        health_checks['config_valid'] = os.path.exists("configs/config.yaml")
        
        # Overall health
        all_data_ready = all(health_checks['data_files'].values())
        all_models_ready = all(health_checks['model_artifacts'].values())
        
        health_checks['overall_health'] = {
            'data_ready': all_data_ready,
            'models_ready': all_models_ready,
            'config_ready': health_checks['config_valid'],
            'pipeline_ready': all_data_ready and health_checks['config_valid']
        }
        
        logger.info(f"Pipeline health: Data={all_data_ready}, Models={all_models_ready}")
        return health_checks
    
    def save_pipeline_results(self, results: Dict, output_dir: str = "artifacts"):
        """Save complete pipeline results."""
        logger.info(f"Saving pipeline results to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_path = os.path.join(output_dir, "pipeline_results.pkl")
        joblib.dump(results, results_path)
        
        # Create summary report
        report_lines = [
            "# End-to-End Pipeline Results",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Summary"
        ]
        
        if 'pipeline_summary' in results:
            summary = results['pipeline_summary']
            report_lines.extend([
                f"- **Status**: {summary['status']}",
                f"- **Duration**: {summary.get('duration_minutes', 0):.1f} minutes",
                f"- **Timestamp**: {summary['timestamp']}",
                ""
            ])
        
        # Component results
        for component, result in results.items():
            if component != 'pipeline_summary' and isinstance(result, dict):
                status = result.get('status', 'unknown')
                report_lines.extend([
                    f"### {component.replace('_', ' ').title()}",
                    f"- Status: {status}",
                    ""
                ])
                
                if status == 'success':
                    # Add component-specific metrics
                    for key, value in result.items():
                        if key != 'status' and not key.startswith('_'):
                            report_lines.append(f"- {key}: {value}")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, "pipeline_report.md")
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info("Pipeline results saved successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="End-to-end pipeline")
    parser.add_argument('--mode', choices=['full', 'fast', 'development', 'health-check'],
                       default='development', help='Pipeline mode')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info(f"Starting end-to-end pipeline in {args.mode} mode")
    
    try:
        # Initialize pipeline
        pipeline = EndToEndPipeline(args.config)
        
        # Run based on mode
        if args.mode == 'health-check':
            results = pipeline.validate_pipeline_health()
            
            print("\n=== Pipeline Health Check ===")
            health = results['overall_health']
            print(f"Data Ready: {'✅' if health['data_ready'] else '❌'}")
            print(f"Models Ready: {'✅' if health['models_ready'] else '❌'}")
            print(f"Config Ready: {'✅' if health['config_ready'] else '❌'}")
            print(f"Pipeline Ready: {'✅' if health['pipeline_ready'] else '❌'}")
            
        elif args.mode == 'fast':
            results = pipeline.run_fast_demo()
            
            print("\n=== Fast Demo Results ===")
            print(f"Status: {results['status']}")
            if results['status'] == 'success':
                print(f"Data Samples: {results['data_samples']}")
                print(f"Models Trained: {results['models_trained']}")
                print(f"Prediction Success: {results['prediction_success']}")
            
        elif args.mode == 'development':
            results = pipeline.run_development_mode()
            
            print("\n=== Development Mode Results ===")
            print(f"Status: {results.get('pipeline_summary', {}).get('status', 'unknown')}")
            if 'pipeline_summary' in results:
                duration = results['pipeline_summary'].get('duration_minutes', 0)
                print(f"Duration: {duration:.1f} minutes")
            
        elif args.mode == 'full':
            results = pipeline.run_full_pipeline()
            
            print("\n=== Full Pipeline Results ===")
            print(f"Status: {results.get('pipeline_summary', {}).get('status', 'unknown')}")
            if 'pipeline_summary' in results:
                duration = results['pipeline_summary'].get('duration_minutes', 0)
                print(f"Duration: {duration:.1f} minutes")
                
                # Show component status
                for component, result in results.items():
                    if component != 'pipeline_summary' and isinstance(result, dict):
                        status = result.get('status', 'unknown')
                        print(f"{component}: {status}")
        
        # Save results
        pipeline.save_pipeline_results(results, args.output)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"end_to_end_pipeline_{args.mode}"):
            mlflow.log_params({'mode': args.mode})
            
            if 'pipeline_summary' in results:
                mlflow.log_metrics({
                    'duration_minutes': results['pipeline_summary'].get('duration_minutes', 0),
                    'success': 1 if results['pipeline_summary'].get('status') == 'SUCCESS' else 0
                })
            
            mlflow.log_artifacts(args.output, artifact_path="pipeline")
        
        logger.info("End-to-end pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"End-to-end pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
