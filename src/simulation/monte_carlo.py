#!/usr/bin/env python3
"""
Monte Carlo Simulation Framework for Squad Selection Validation.
Simulates multiple scenarios to validate model robustness and performance.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging
from models.stage2.api import Stage2API

logger = logging.getLogger(__name__)


@dataclass
class SimulationScenario:
    """Simulation scenario configuration."""
    scenario_id: str
    formation: str
    injury_rate: float
    fitness_variance: float
    opponent_strength: float
    tactical_style: str
    n_simulations: int = 1000


class MonteCarloSimulator:
    """Monte Carlo simulation framework for squad selection."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sim_config = self.config['simulation']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
        # Initialize Stage-2 API
        self.stage2_api = Stage2API()
        
    def create_simulation_scenarios(self) -> List[SimulationScenario]:
        """Create diverse simulation scenarios."""
        logger.info("Creating simulation scenarios...")
        
        scenarios = []
        
        # Base scenarios for different formations
        formations = ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1"]
        
        for formation in formations:
            # Normal conditions
            scenarios.append(SimulationScenario(
                scenario_id=f"{formation}_normal",
                formation=formation,
                injury_rate=0.1,
                fitness_variance=0.1,
                opponent_strength=0.5,
                tactical_style="balanced"
            ))
            
            # High injury scenario
            scenarios.append(SimulationScenario(
                scenario_id=f"{formation}_high_injury",
                formation=formation,
                injury_rate=0.3,
                fitness_variance=0.2,
                opponent_strength=0.5,
                tactical_style="defensive"
            ))
            
            # Strong opponent scenario
            scenarios.append(SimulationScenario(
                scenario_id=f"{formation}_strong_opponent",
                formation=formation,
                injury_rate=0.1,
                fitness_variance=0.1,
                opponent_strength=0.8,
                tactical_style="attacking"
            ))
        
        logger.info(f"Created {len(scenarios)} simulation scenarios")
        return scenarios
    
    def simulate_match_conditions(self, base_players: pd.DataFrame,
                                scenario: SimulationScenario) -> pd.DataFrame:
        """Simulate match conditions for a scenario."""
        players = base_players.copy()
        
        # Simulate injuries
        n_players = len(players)
        n_injured = int(n_players * scenario.injury_rate)
        
        if n_injured > 0:
            injured_indices = np.random.choice(n_players, n_injured, replace=False)
            players.loc[injured_indices, 'fitness_score'] = np.random.uniform(0.1, 0.4, n_injured)
        
        # Add fitness variance
        fitness_noise = np.random.normal(0, scenario.fitness_variance, n_players)
        players['fitness_score'] = np.clip(
            players['fitness_score'] + fitness_noise, 0.0, 1.0
        )
        
        # Adjust performance based on opponent strength
        performance_modifier = 1.0 - (scenario.opponent_strength - 0.5) * 0.3
        if 'performance_score' in players.columns:
            players['performance_score'] *= performance_modifier
        
        # Tactical style adjustments
        if scenario.tactical_style == "attacking":
            # Boost attacking players
            attacking_positions = ['LW', 'RW', 'CF', 'AM']
            mask = players['primary_position'].isin(attacking_positions)
            players.loc[mask, 'performance_score'] *= 1.1
            
        elif scenario.tactical_style == "defensive":
            # Boost defensive players
            defensive_positions = ['GK', 'CB', 'LB', 'RB', 'DM']
            mask = players['primary_position'].isin(defensive_positions)
            players.loc[mask, 'performance_score'] *= 1.1
        
        return players
    
    def run_single_simulation(self, players: pd.DataFrame, 
                            scenario: SimulationScenario) -> Dict:
        """Run a single Monte Carlo simulation."""
        try:
            # Simulate match conditions
            simulated_players = self.simulate_match_conditions(players, scenario)
            
            # Get squad selection
            selection_result = self.stage2_api.predict_squad_selection(
                simulated_players,
                formation=scenario.formation,
                return_probabilities=True
            )
            
            # Calculate simulation metrics
            simulation_result = {
                'scenario_id': scenario.scenario_id,
                'formation': scenario.formation,
                'selected_players': selection_result['selected_players'],
                'total_score': selection_result['total_score'],
                'n_selected': len(selection_result['selected_players']),
                'avg_fitness': simulated_players.loc[
                    simulated_players['player_id'].isin(selection_result['selected_players']),
                    'fitness_score'
                ].mean(),
                'formation_complete': len(selection_result['selected_players']) >= 11,
                'position_assignments': selection_result['position_assignments']
            }
            
            return simulation_result
            
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return {
                'scenario_id': scenario.scenario_id,
                'formation': scenario.formation,
                'error': str(e),
                'simulation_failed': True
            }
    
    def run_scenario_simulations(self, players: pd.DataFrame,
                               scenario: SimulationScenario) -> Dict:
        """Run multiple simulations for a scenario."""
        logger.info(f"Running {scenario.n_simulations} simulations for {scenario.scenario_id}")
        
        simulation_results = []
        
        # Run simulations
        for sim_id in range(scenario.n_simulations):
            result = self.run_single_simulation(players, scenario)
            result['simulation_id'] = sim_id
            simulation_results.append(result)
            
            if (sim_id + 1) % 100 == 0:
                logger.info(f"Completed {sim_id + 1}/{scenario.n_simulations} simulations")
        
        # Aggregate results
        successful_sims = [r for r in simulation_results if not r.get('simulation_failed', False)]
        
        if successful_sims:
            scenario_summary = {
                'scenario_id': scenario.scenario_id,
                'n_simulations': len(successful_sims),
                'success_rate': len(successful_sims) / scenario.n_simulations,
                'avg_total_score': np.mean([r['total_score'] for r in successful_sims]),
                'std_total_score': np.std([r['total_score'] for r in successful_sims]),
                'avg_fitness': np.mean([r['avg_fitness'] for r in successful_sims if 'avg_fitness' in r]),
                'formation_completion_rate': np.mean([r['formation_complete'] for r in successful_sims]),
                'squad_size_consistency': np.std([r['n_selected'] for r in successful_sims])
            }
        else:
            scenario_summary = {
                'scenario_id': scenario.scenario_id,
                'n_simulations': 0,
                'success_rate': 0.0,
                'error': 'All simulations failed'
            }
        
        return {
            'scenario_summary': scenario_summary,
            'individual_results': simulation_results
        }
    
    def run_parallel_simulations(self, players: pd.DataFrame,
                               scenarios: List[SimulationScenario],
                               n_workers: int = None) -> Dict:
        """Run simulations in parallel across scenarios."""
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(scenarios))
        
        logger.info(f"Running parallel simulations with {n_workers} workers")
        
        all_results = {}
        
        # For now, run sequentially due to complexity of parallel Stage2API
        for scenario in scenarios:
            scenario_results = self.run_scenario_simulations(players, scenario)
            all_results[scenario.scenario_id] = scenario_results
        
        return all_results
    
    def analyze_simulation_results(self, simulation_results: Dict) -> Dict:
        """Analyze Monte Carlo simulation results."""
        logger.info("Analyzing simulation results...")
        
        # Aggregate across all scenarios
        all_summaries = [results['scenario_summary'] for results in simulation_results.values()
                        if 'scenario_summary' in results]
        
        if not all_summaries:
            return {'error': 'No successful simulations to analyze'}
        
        analysis = {
            'overall_success_rate': np.mean([s['success_rate'] for s in all_summaries]),
            'avg_score_across_scenarios': np.mean([s.get('avg_total_score', 0) for s in all_summaries]),
            'score_stability': np.std([s.get('avg_total_score', 0) for s in all_summaries]),
            'formation_robustness': np.mean([s.get('formation_completion_rate', 0) for s in all_summaries]),
            'n_scenarios_tested': len(all_summaries)
        }
        
        # Formation-specific analysis
        formation_analysis = {}
        for summary in all_summaries:
            formation = summary['scenario_id'].split('_')[0]
            if formation not in formation_analysis:
                formation_analysis[formation] = []
            formation_analysis[formation].append(summary)
        
        formation_stats = {}
        for formation, summaries in formation_analysis.items():
            formation_stats[formation] = {
                'avg_score': np.mean([s.get('avg_total_score', 0) for s in summaries]),
                'score_stability': np.std([s.get('avg_total_score', 0) for s in summaries]),
                'success_rate': np.mean([s['success_rate'] for s in summaries]),
                'n_scenarios': len(summaries)
            }
        
        analysis['formation_analysis'] = formation_stats
        
        # Scenario difficulty analysis
        scenario_difficulty = {}
        for summary in all_summaries:
            scenario_type = '_'.join(summary['scenario_id'].split('_')[1:])
            if scenario_type not in scenario_difficulty:
                scenario_difficulty[scenario_type] = []
            scenario_difficulty[scenario_type].append(summary)
        
        difficulty_stats = {}
        for scenario_type, summaries in scenario_difficulty.items():
            difficulty_stats[scenario_type] = {
                'avg_score': np.mean([s.get('avg_total_score', 0) for s in summaries]),
                'success_rate': np.mean([s['success_rate'] for s in summaries]),
                'n_formations': len(summaries)
            }
        
        analysis['scenario_difficulty'] = difficulty_stats
        
        logger.info(f"Analysis completed: {analysis['overall_success_rate']:.3f} success rate")
        return analysis
    
    def create_simulation_plots(self, simulation_results: Dict, analysis: Dict,
                              output_dir: str = "artifacts") -> List[str]:
        """Create visualization plots for simulation results."""
        logger.info("Creating simulation plots...")
        
        plot_paths = []
        
        # Plot 1: Score distribution across scenarios
        plt.figure(figsize=(15, 10))
        
        # Collect scores by scenario type
        scenario_scores = {}
        for scenario_id, results in simulation_results.items():
            if 'individual_results' in results:
                scores = [r['total_score'] for r in results['individual_results'] 
                         if not r.get('simulation_failed', False)]
                scenario_type = '_'.join(scenario_id.split('_')[1:])
                if scenario_type not in scenario_scores:
                    scenario_scores[scenario_type] = []
                scenario_scores[scenario_type].extend(scores)
        
        # Box plot of scores by scenario
        plt.subplot(2, 2, 1)
        if scenario_scores:
            data_for_plot = []
            labels_for_plot = []
            for scenario_type, scores in scenario_scores.items():
                data_for_plot.append(scores)
                labels_for_plot.append(scenario_type)
            
            plt.boxplot(data_for_plot, labels=labels_for_plot)
            plt.title('Score Distribution by Scenario Type')
            plt.ylabel('Total Score')
            plt.xticks(rotation=45)
        
        # Plot 2: Formation performance comparison
        plt.subplot(2, 2, 2)
        if 'formation_analysis' in analysis:
            formations = list(analysis['formation_analysis'].keys())
            avg_scores = [analysis['formation_analysis'][f]['avg_score'] for f in formations]
            
            plt.bar(formations, avg_scores)
            plt.title('Average Score by Formation')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
        
        # Plot 3: Success rate by scenario difficulty
        plt.subplot(2, 2, 3)
        if 'scenario_difficulty' in analysis:
            scenarios = list(analysis['scenario_difficulty'].keys())
            success_rates = [analysis['scenario_difficulty'][s]['success_rate'] for s in scenarios]
            
            plt.bar(scenarios, success_rates)
            plt.title('Success Rate by Scenario Difficulty')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        # Plot 4: Score stability analysis
        plt.subplot(2, 2, 4)
        if 'formation_analysis' in analysis:
            formations = list(analysis['formation_analysis'].keys())
            stabilities = [analysis['formation_analysis'][f]['score_stability'] for f in formations]
            
            plt.bar(formations, stabilities)
            plt.title('Score Stability by Formation')
            plt.ylabel('Score Standard Deviation')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "monte_carlo_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(plot_path)
        
        # Plot 5: Detailed score distributions
        plt.figure(figsize=(12, 8))
        
        if scenario_scores:
            n_scenarios = len(scenario_scores)
            cols = min(3, n_scenarios)
            rows = (n_scenarios + cols - 1) // cols
            
            for i, (scenario_type, scores) in enumerate(scenario_scores.items()):
                plt.subplot(rows, cols, i + 1)
                plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{scenario_type}\n(μ={np.mean(scores):.3f}, σ={np.std(scores):.3f})')
                plt.xlabel('Total Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "score_distributions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(plot_path)
        
        logger.info(f"Created {len(plot_paths)} simulation plots")
        return plot_paths
    
    def stress_test_pipeline(self, players: pd.DataFrame) -> Dict:
        """Stress test the pipeline with extreme scenarios."""
        logger.info("Running pipeline stress tests...")
        
        stress_tests = []
        
        # Test 1: Minimal squad (exactly 11 players)
        minimal_squad = players.head(11).copy()
        try:
            result = self.stage2_api.predict_squad_selection(minimal_squad, formation="4-3-3")
            stress_tests.append({
                'test': 'minimal_squad',
                'success': True,
                'n_players': len(minimal_squad),
                'selected': len(result['selected_players'])
            })
        except Exception as e:
            stress_tests.append({
                'test': 'minimal_squad',
                'success': False,
                'error': str(e)
            })
        
        # Test 2: Large squad (50+ players)
        if len(players) >= 50:
            large_squad = players.head(50).copy()
            try:
                result = self.stage2_api.predict_squad_selection(large_squad, formation="4-3-3")
                stress_tests.append({
                    'test': 'large_squad',
                    'success': True,
                    'n_players': len(large_squad),
                    'selected': len(result['selected_players'])
                })
            except Exception as e:
                stress_tests.append({
                    'test': 'large_squad',
                    'success': False,
                    'error': str(e)
                })
        
        # Test 3: All injured players
        injured_squad = players.head(25).copy()
        injured_squad['fitness_score'] = 0.2  # Very low fitness
        try:
            result = self.stage2_api.predict_squad_selection(injured_squad, formation="4-3-3")
            stress_tests.append({
                'test': 'all_injured',
                'success': True,
                'avg_fitness': injured_squad['fitness_score'].mean(),
                'selected': len(result['selected_players'])
            })
        except Exception as e:
            stress_tests.append({
                'test': 'all_injured',
                'success': False,
                'error': str(e)
            })
        
        # Test 4: Missing features
        incomplete_squad = players.head(20).copy()
        # Remove some features
        features_to_remove = ['performance_score', 'recent_form_3_matches']
        for feature in features_to_remove:
            if feature in incomplete_squad.columns:
                incomplete_squad = incomplete_squad.drop(columns=[feature])
        
        try:
            result = self.stage2_api.predict_squad_selection(incomplete_squad, formation="4-3-3")
            stress_tests.append({
                'test': 'missing_features',
                'success': True,
                'n_features': len(incomplete_squad.columns),
                'selected': len(result['selected_players'])
            })
        except Exception as e:
            stress_tests.append({
                'test': 'missing_features',
                'success': False,
                'error': str(e)
            })
        
        stress_summary = {
            'total_tests': len(stress_tests),
            'successful_tests': sum(1 for t in stress_tests if t['success']),
            'stress_test_results': stress_tests
        }
        
        logger.info(f"Stress tests: {stress_summary['successful_tests']}/{stress_summary['total_tests']} passed")
        return stress_summary
    
    def save_simulation_results(self, simulation_results: Dict, analysis: Dict,
                              stress_tests: Dict, output_dir: str = "artifacts"):
        """Save all simulation results and analysis."""
        logger.info(f"Saving simulation results to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_path = os.path.join(output_dir, "monte_carlo_results.pkl")
        joblib.dump(simulation_results, results_path)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, "simulation_analysis.pkl")
        joblib.dump(analysis, analysis_path)
        
        # Save stress tests
        stress_path = os.path.join(output_dir, "stress_test_results.pkl")
        joblib.dump(stress_tests, stress_path)
        
        # Create summary report
        report_lines = [
            "# Monte Carlo Simulation Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- **Overall Success Rate**: {analysis.get('overall_success_rate', 0):.3f}",
            f"- **Average Score**: {analysis.get('avg_score_across_scenarios', 0):.3f}",
            f"- **Score Stability**: {analysis.get('score_stability', 0):.3f}",
            f"- **Formation Robustness**: {analysis.get('formation_robustness', 0):.3f}",
            f"- **Scenarios Tested**: {analysis.get('n_scenarios_tested', 0)}",
            "",
            "## Formation Performance",
            ""
        ]
        
        if 'formation_analysis' in analysis:
            for formation, stats in analysis['formation_analysis'].items():
                report_lines.extend([
                    f"### {formation}",
                    f"- Average Score: {stats['avg_score']:.3f}",
                    f"- Score Stability: {stats['score_stability']:.3f}",
                    f"- Success Rate: {stats['success_rate']:.3f}",
                    ""
                ])
        
        report_lines.extend([
            "## Stress Test Results",
            f"- **Tests Passed**: {stress_tests['successful_tests']}/{stress_tests['total_tests']}",
            ""
        ])
        
        for test in stress_tests['stress_test_results']:
            status = "✅ PASSED" if test['success'] else "❌ FAILED"
            report_lines.append(f"- **{test['test']}**: {status}")
            if not test['success']:
                report_lines.append(f"  - Error: {test.get('error', 'Unknown')}")
        
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, "monte_carlo_report.md")
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info("Simulation results saved successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Monte Carlo simulation")
    parser.add_argument('--data', default='data/processed/features.csv',
                       help='Feature data file')
    parser.add_argument('--n-simulations', type=int, default=100,
                       help='Number of simulations per scenario')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory')
    parser.add_argument('--stress-test', action='store_true',
                       help='Run stress tests')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting Monte Carlo simulation")
    
    try:
        # Initialize simulator
        simulator = MonteCarloSimulator()
        
        # Load data
        features_df = pd.read_csv(args.data)
        
        # Use recent data for simulation
        players_sample = features_df.head(30).copy()  # Sample for demo
        
        logger.info(f"Simulation data: {len(players_sample)} players")
        
        # Create scenarios
        scenarios = simulator.create_simulation_scenarios()
        
        # Update simulation count
        for scenario in scenarios:
            scenario.n_simulations = args.n_simulations
        
        # Run simulations
        simulation_results = simulator.run_parallel_simulations(
            players_sample, scenarios, args.n_workers
        )
        
        # Analyze results
        analysis = simulator.analyze_simulation_results(simulation_results)
        
        # Run stress tests if requested
        stress_tests = {}
        if args.stress_test:
            stress_tests = simulator.stress_test_pipeline(players_sample)
        
        # Create plots
        plot_paths = simulator.create_simulation_plots(
            simulation_results, analysis, args.output
        )
        
        # Save results
        simulator.save_simulation_results(
            simulation_results, analysis, stress_tests, args.output
        )
        
        print(f"\n=== Monte Carlo Simulation Results ===")
        print(f"Scenarios Tested: {len(scenarios)}")
        print(f"Overall Success Rate: {analysis.get('overall_success_rate', 0):.3f}")
        print(f"Average Score: {analysis.get('avg_score_across_scenarios', 0):.3f}")
        print(f"Score Stability: {analysis.get('score_stability', 0):.3f}")
        
        if 'formation_analysis' in analysis:
            print(f"\nBest Formation: {max(analysis['formation_analysis'].keys(), key=lambda f: analysis['formation_analysis'][f]['avg_score'])}")
        
        if stress_tests:
            print(f"\nStress Tests: {stress_tests['successful_tests']}/{stress_tests['total_tests']} passed")
        
        print(f"\nPlots created: {len(plot_paths)}")
        print(f"Results saved to: {args.output}")
        
        logger.info("Monte Carlo simulation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
