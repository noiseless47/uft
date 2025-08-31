#!/usr/bin/env python3
"""
XI + Bench Optimizer using Mixed Integer Programming.
Optimizes squad selection with formation constraints and compatibility.
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
from pulp import *
import itertools

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class SquadOptimizer:
    """Mixed Integer Programming optimizer for squad selection."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.optimizer_config = self.config['stage2']['optimizer']
        
    def optimize_squad(self, players_df: pd.DataFrame, 
                      selection_probs: np.ndarray,
                      compat_matrix: np.ndarray,
                      player_mappings: Dict,
                      formation: str = "4-3-3",
                      squad_size: int = 18) -> Dict:
        """
        Optimize squad selection using MIP.
        
        Args:
            players_df: Player data with positions and attributes
            selection_probs: Final selection probabilities from Stage-2
            compat_matrix: Player compatibility matrix
            player_mappings: Player ID to matrix index mappings
            formation: Target formation
            squad_size: Total squad size (XI + bench)
            
        Returns:
            Optimized squad selection with positions
        """
        logger.info(f"Optimizing squad selection for formation: {formation}")
        
        # Parse formation requirements
        formation_reqs = self._parse_formation_requirements(formation)
        
        # Create optimization problem
        prob = LpProblem("Squad_Selection", LpMaximize)
        
        # Decision variables
        n_players = len(players_df)
        x = {}  # x[i] = 1 if player i is selected
        y = {}  # y[i,p] = 1 if player i is assigned to position p
        
        for i in range(n_players):
            x[i] = LpVariable(f"select_{i}", cat='Binary')
            
            for pos in formation_reqs.keys():
                y[(i, pos)] = LpVariable(f"assign_{i}_{pos}", cat='Binary')
        
        # Objective function: maximize selection probability + compatibility bonus
        objective_terms = []
        
        # Individual selection probabilities
        for i in range(n_players):
            objective_terms.append(selection_probs[i] * x[i])
        
        # Pairwise compatibility bonus
        compatibility_weight = self.optimizer_config.get('compatibility_weight', 0.1)
        
        for i in range(n_players):
            for j in range(i + 1, n_players):
                player_i_id = players_df.iloc[i]['player_id']
                player_j_id = players_df.iloc[j]['player_id']
                
                if (player_i_id in player_mappings['player_to_idx'] and 
                    player_j_id in player_mappings['player_to_idx']):
                    
                    idx_i = player_mappings['player_to_idx'][player_i_id]
                    idx_j = player_mappings['player_to_idx'][player_j_id]
                    compat_score = compat_matrix[idx_i, idx_j]
                    
                    # Add compatibility bonus when both players selected
                    objective_terms.append(
                        compatibility_weight * compat_score * x[i] * x[j]
                    )
        
        prob += lpSum(objective_terms)
        
        # Constraints
        
        # 1. Squad size constraint
        prob += lpSum([x[i] for i in range(n_players)]) == squad_size
        
        # 2. Formation constraints (starting XI)
        for pos, required_count in formation_reqs.items():
            prob += lpSum([y[(i, pos)] for i in range(n_players)]) == required_count
        
        # 3. Position assignment constraints
        for i in range(n_players):
            # Each player can be assigned to at most one position
            prob += lpSum([y[(i, pos)] for pos in formation_reqs.keys()]) <= x[i]
            
            # Player can only be assigned to compatible positions
            player_row = players_df.iloc[i]
            primary_pos = player_row.get('primary_position', 'CM')
            compatible_positions = self._get_compatible_positions(primary_pos)
            
            for pos in formation_reqs.keys():
                if pos not in compatible_positions:
                    prob += y[(i, pos)] == 0
        
        # 4. Minimum fitness constraint
        min_fitness = self.optimizer_config.get('min_fitness_score', 0.3)
        for i in range(n_players):
            fitness_score = players_df.iloc[i].get('fitness_score', 1.0)
            if fitness_score < min_fitness:
                prob += x[i] == 0
        
        # 5. Position diversity on bench
        bench_players = squad_size - sum(formation_reqs.values())
        if bench_players > 0:
            # Ensure bench has at least one player from each position group
            position_groups = {
                'defense': ['CB', 'LB', 'RB'],
                'midfield': ['DM', 'CM', 'AM'],
                'attack': ['LW', 'RW', 'CF']
            }
            
            for group_name, positions in position_groups.items():
                bench_in_group = []
                for i in range(n_players):
                    player_pos = players_df.iloc[i].get('primary_position', 'CM')
                    if player_pos in positions:
                        # Bench player = selected but not in starting XI
                        bench_in_group.append(
                            x[i] - lpSum([y[(i, pos)] for pos in formation_reqs.keys()])
                        )
                
                if bench_in_group:
                    prob += lpSum(bench_in_group) >= 1
        
        # Solve optimization
        logger.info("Solving MIP optimization...")
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status != LpStatusOptimal:
            logger.error(f"Optimization failed with status: {LpStatus[prob.status]}")
            return self._fallback_selection(players_df, selection_probs, formation, squad_size)
        
        # Extract solution
        selected_indices = [i for i in range(n_players) if x[i].varValue == 1]
        selected_players = players_df.iloc[selected_indices]['player_id'].tolist()
        
        # Extract position assignments
        position_assignments = {}
        starting_xi = []
        
        for i in range(n_players):
            if x[i].varValue == 1:
                player_id = players_df.iloc[i]['player_id']
                assigned_pos = None
                
                for pos in formation_reqs.keys():
                    if y[(i, pos)].varValue == 1:
                        assigned_pos = pos
                        starting_xi.append(player_id)
                        break
                
                position_assignments[player_id] = assigned_pos or 'BENCH'
        
        bench_players = [pid for pid in selected_players if pid not in starting_xi]
        
        result = {
            'selected_players': selected_players,
            'starting_xi': starting_xi,
            'bench_players': bench_players,
            'position_assignments': position_assignments,
            'formation': formation,
            'total_score': value(prob.objective),
            'optimization_status': LpStatus[prob.status]
        }
        
        logger.info(f"Optimization completed: {len(starting_xi)} starters, {len(bench_players)} bench")
        return result
    
    def _parse_formation_requirements(self, formation: str) -> Dict[str, int]:
        """Parse formation into position requirements."""
        formations = {
            "4-3-3": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'CM': 3, 'LW': 1, 'RW': 1, 'CF': 1},
            "4-4-2": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'CM': 2, 'LW': 1, 'RW': 1, 'CF': 2},
            "3-5-2": {'GK': 1, 'CB': 3, 'CM': 3, 'LW': 1, 'RW': 1, 'CF': 2},
            "4-2-3-1": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'DM': 2, 'AM': 3, 'CF': 1},
            "5-3-2": {'GK': 1, 'CB': 3, 'LB': 1, 'RB': 1, 'CM': 3, 'CF': 2}
        }
        
        return formations.get(formation, formations["4-3-3"])
    
    def _get_compatible_positions(self, primary_position: str) -> List[str]:
        """Get positions a player can play."""
        position_compatibility = {
            'GK': ['GK'],
            'CB': ['CB', 'DM'],
            'LB': ['LB', 'LW', 'CM'],
            'RB': ['RB', 'RW', 'CM'],
            'DM': ['DM', 'CB', 'CM'],
            'CM': ['CM', 'DM', 'AM', 'LW', 'RW'],
            'AM': ['AM', 'CM', 'LW', 'RW', 'CF'],
            'LW': ['LW', 'LB', 'AM', 'CF'],
            'RW': ['RW', 'RB', 'AM', 'CF'],
            'CF': ['CF', 'AM', 'LW', 'RW']
        }
        
        return position_compatibility.get(primary_position, [primary_position])
    
    def _fallback_selection(self, players_df: pd.DataFrame, 
                          selection_probs: np.ndarray,
                          formation: str, squad_size: int) -> Dict:
        """Fallback greedy selection if MIP fails."""
        logger.warning("Using fallback greedy selection")
        
        formation_reqs = self._parse_formation_requirements(formation)
        
        # Sort by probability
        sorted_indices = np.argsort(selection_probs)[::-1]
        
        selected_players = []
        position_assignments = {}
        position_counts = {pos: 0 for pos in formation_reqs.keys()}
        
        # Select starting XI first
        for idx in sorted_indices:
            if len([p for p in position_assignments.values() if p != 'BENCH']) >= sum(formation_reqs.values()):
                break
                
            player_row = players_df.iloc[idx]
            player_id = player_row['player_id']
            primary_pos = player_row.get('primary_position', 'CM')
            
            # Try to assign to formation position
            compatible_positions = self._get_compatible_positions(primary_pos)
            assigned = False
            
            for pos in compatible_positions:
                if pos in position_counts and position_counts[pos] < formation_reqs[pos]:
                    selected_players.append(player_id)
                    position_assignments[player_id] = pos
                    position_counts[pos] += 1
                    assigned = True
                    break
            
            if not assigned and len(selected_players) < squad_size:
                selected_players.append(player_id)
                position_assignments[player_id] = 'BENCH'
        
        # Fill remaining squad with bench players
        for idx in sorted_indices:
            if len(selected_players) >= squad_size:
                break
                
            player_id = players_df.iloc[idx]['player_id']
            if player_id not in selected_players:
                selected_players.append(player_id)
                position_assignments[player_id] = 'BENCH'
        
        starting_xi = [pid for pid, pos in position_assignments.items() if pos != 'BENCH']
        bench_players = [pid for pid, pos in position_assignments.items() if pos == 'BENCH']
        
        return {
            'selected_players': selected_players,
            'starting_xi': starting_xi,
            'bench_players': bench_players,
            'position_assignments': position_assignments,
            'formation': formation,
            'total_score': sum(selection_probs[players_df['player_id'] == pid].iloc[0] 
                             for pid in selected_players if len(players_df[players_df['player_id'] == pid]) > 0),
            'optimization_status': 'FALLBACK_GREEDY'
        }
    
    def validate_squad_selection(self, result: Dict, formation: str) -> Dict:
        """Validate that squad selection meets all constraints."""
        logger.info("Validating squad selection...")
        
        formation_reqs = self._parse_formation_requirements(formation)
        
        validation = {
            'formation_complete': True,
            'position_violations': [],
            'squad_size_correct': len(result['selected_players']) <= 25,  # FIFA limit
            'starting_xi_size': len(result['starting_xi']) == sum(formation_reqs.values())
        }
        
        # Check formation requirements
        position_counts = {}
        for player_id, position in result['position_assignments'].items():
            if position != 'BENCH':
                position_counts[position] = position_counts.get(position, 0) + 1
        
        for pos, required in formation_reqs.items():
            actual = position_counts.get(pos, 0)
            if actual != required:
                validation['formation_complete'] = False
                validation['position_violations'].append(f"{pos}: need {required}, got {actual}")
        
        # Log validation results
        if validation['formation_complete']:
            logger.info("Squad selection validation: PASSED")
        else:
            logger.warning("Squad selection validation: FAILED")
            for violation in validation['position_violations']:
                logger.warning(f"  Violation: {violation}")
        
        return validation
    
    def optimize_with_constraints(self, players_df: pd.DataFrame,
                                selection_probs: np.ndarray,
                                compat_matrix: np.ndarray,
                                player_mappings: Dict,
                                constraints: Dict) -> Dict:
        """
        Advanced optimization with custom constraints.
        
        Args:
            constraints: Dictionary with constraint specifications
                - formation: Required formation
                - squad_size: Total squad size
                - min_fitness: Minimum fitness threshold
                - max_age: Maximum average age
                - min_experience: Minimum caps/experience
                - budget_limit: Salary budget limit (if available)
        """
        logger.info("Running advanced optimization with custom constraints")
        
        formation = constraints.get('formation', '4-3-3')
        squad_size = constraints.get('squad_size', 18)
        
        # Create MIP problem
        prob = LpProblem("Advanced_Squad_Selection", LpMaximize)
        
        n_players = len(players_df)
        
        # Decision variables
        x = [LpVariable(f"select_{i}", cat='Binary') for i in range(n_players)]
        
        # Position assignment variables
        formation_reqs = self._parse_formation_requirements(formation)
        y = {}
        for i in range(n_players):
            for pos in formation_reqs.keys():
                y[(i, pos)] = LpVariable(f"assign_{i}_{pos}", cat='Binary')
        
        # Objective: maximize weighted sum of probabilities and compatibility
        prob_weight = 0.7
        compat_weight = 0.3
        
        objective_terms = []
        
        # Individual probabilities
        for i in range(n_players):
            objective_terms.append(prob_weight * selection_probs[i] * x[i])
        
        # Compatibility terms
        for i in range(n_players):
            for j in range(i + 1, n_players):
                player_i_id = players_df.iloc[i]['player_id']
                player_j_id = players_df.iloc[j]['player_id']
                
                if (player_i_id in player_mappings['player_to_idx'] and 
                    player_j_id in player_mappings['player_to_idx']):
                    
                    idx_i = player_mappings['player_to_idx'][player_i_id]
                    idx_j = player_mappings['player_to_idx'][player_j_id]
                    compat_score = compat_matrix[idx_i, idx_j]
                    
                    objective_terms.append(compat_weight * compat_score * x[i] * x[j])
        
        prob += lpSum(objective_terms)
        
        # Basic constraints
        prob += lpSum(x) == squad_size  # Squad size
        
        # Formation constraints
        for pos, count in formation_reqs.items():
            prob += lpSum([y[(i, pos)] for i in range(n_players)]) == count
        
        # Position assignment logic
        for i in range(n_players):
            prob += lpSum([y[(i, pos)] for pos in formation_reqs.keys()]) <= x[i]
            
            # Position compatibility
            primary_pos = players_df.iloc[i].get('primary_position', 'CM')
            compatible_positions = self._get_compatible_positions(primary_pos)
            
            for pos in formation_reqs.keys():
                if pos not in compatible_positions:
                    prob += y[(i, pos)] == 0
        
        # Additional constraints
        
        # Fitness constraint
        if 'min_fitness' in constraints:
            min_fitness = constraints['min_fitness']
            for i in range(n_players):
                fitness = players_df.iloc[i].get('fitness_score', 1.0)
                if fitness < min_fitness:
                    prob += x[i] == 0
        
        # Age constraint
        if 'max_avg_age' in constraints:
            max_avg_age = constraints['max_avg_age']
            ages = [players_df.iloc[i].get('age', 25) for i in range(n_players)]
            prob += lpSum([ages[i] * x[i] for i in range(n_players)]) <= max_avg_age * squad_size
        
        # Experience constraint
        if 'min_experience' in constraints:
            min_exp = constraints['min_experience']
            for i in range(n_players):
                experience = players_df.iloc[i].get('caps', 0)
                if experience < min_exp:
                    prob += x[i] == 0
        
        # Solve
        logger.info("Solving advanced MIP...")
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status != LpStatusOptimal:
            logger.warning(f"Advanced optimization failed: {LpStatus[prob.status]}")
            return self.optimize_squad(players_df, selection_probs, compat_matrix, 
                                     player_mappings, formation, squad_size)
        
        # Extract results
        selected_indices = [i for i in range(n_players) if x[i].varValue == 1]
        selected_players = players_df.iloc[selected_indices]['player_id'].tolist()
        
        position_assignments = {}
        starting_xi = []
        
        for i in range(n_players):
            if x[i].varValue == 1:
                player_id = players_df.iloc[i]['player_id']
                assigned_pos = None
                
                for pos in formation_reqs.keys():
                    if y[(i, pos)].varValue == 1:
                        assigned_pos = pos
                        starting_xi.append(player_id)
                        break
                
                position_assignments[player_id] = assigned_pos or 'BENCH'
        
        bench_players = [pid for pid in selected_players if pid not in starting_xi]
        
        result = {
            'selected_players': selected_players,
            'starting_xi': starting_xi,
            'bench_players': bench_players,
            'position_assignments': position_assignments,
            'formation': formation,
            'total_score': value(prob.objective),
            'optimization_status': 'OPTIMAL',
            'constraints_applied': constraints
        }
        
        logger.info(f"Advanced optimization completed: score = {result['total_score']:.3f}")
        return result
    
    def compare_formations(self, players_df: pd.DataFrame,
                         selection_probs: np.ndarray,
                         compat_matrix: np.ndarray,
                         player_mappings: Dict,
                         formations: List[str] = None) -> Dict:
        """Compare multiple formations and recommend best."""
        if formations is None:
            formations = ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1"]
        
        logger.info(f"Comparing {len(formations)} formations")
        
        results = {}
        
        for formation in formations:
            try:
                result = self.optimize_squad(
                    players_df, selection_probs, compat_matrix, 
                    player_mappings, formation
                )
                results[formation] = result
                logger.info(f"{formation}: score = {result['total_score']:.3f}")
                
            except Exception as e:
                logger.warning(f"Formation {formation} optimization failed: {e}")
                continue
        
        # Find best formation
        if results:
            best_formation = max(results.keys(), key=lambda f: results[f]['total_score'])
            logger.info(f"Best formation: {best_formation}")
            
            return {
                'best_formation': best_formation,
                'best_result': results[best_formation],
                'all_results': results,
                'formation_scores': {f: r['total_score'] for f, r in results.items()}
            }
        
        return {'error': 'No formations could be optimized'}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Optimize squad selection")
    parser.add_argument('--data', default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--artifacts', default='artifacts',
                       help='Artifacts directory')
    parser.add_argument('--formation', default='4-3-3',
                       help='Target formation')
    parser.add_argument('--squad-size', type=int, default=18,
                       help='Squad size')
    parser.add_argument('--compare-formations', action='store_true',
                       help='Compare multiple formations')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting squad optimization")
    
    try:
        # Initialize optimizer
        optimizer = SquadOptimizer()
        
        # Load data (mock for demo)
        players_df = pd.read_csv(os.path.join(args.data, "players.csv"))
        
        # Mock selection probabilities (would come from Stage-2 API)
        n_players = len(players_df)
        selection_probs = np.random.beta(2, 5, n_players)  # Realistic distribution
        
        # Load compatibility matrix
        matrix_path = os.path.join(args.artifacts, "compat_matrix_v1.npz")
        matrix_data = np.load(matrix_path, allow_pickle=True)
        compat_matrix = matrix_data['compatibility_matrix']
        player_mappings = {
            'player_to_idx': matrix_data['player_to_idx'].item(),
            'idx_to_player': matrix_data['idx_to_player'].item()
        }
        
        if args.compare_formations:
            # Compare formations
            comparison = optimizer.compare_formations(
                players_df, selection_probs, compat_matrix, player_mappings
            )
            
            print("\n=== Formation Comparison ===")
            for formation, score in comparison['formation_scores'].items():
                print(f"{formation}: {score:.3f}")
            print(f"Best: {comparison['best_formation']}")
            
        else:
            # Single formation optimization
            result = optimizer.optimize_squad(
                players_df, selection_probs, compat_matrix, player_mappings,
                args.formation, args.squad_size
            )
            
            # Validate
            validation = optimizer.validate_squad_selection(result, args.formation)
            
            print(f"\n=== Squad Optimization Results ===")
            print(f"Formation: {result['formation']}")
            print(f"Total Score: {result['total_score']:.3f}")
            print(f"Status: {result['optimization_status']}")
            print(f"Starting XI: {len(result['starting_xi'])}")
            print(f"Bench: {len(result['bench_players'])}")
            print(f"Validation: {'PASSED' if validation['formation_complete'] else 'FAILED'}")
        
        logger.info("Squad optimization completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Squad optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
