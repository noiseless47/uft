#!/usr/bin/env python3
"""
Multi-Armed Bandit Feature Selection.
Dynamically selects optimal feature subsets using contextual bandits.
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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import itertools

sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ContextualBandit:
    """Contextual Multi-Armed Bandit for feature selection."""
    
    def __init__(self, n_features: int, context_dim: int = 5, 
                 alpha: float = 1.0, lambda_reg: float = 1.0):
        """
        Initialize contextual bandit.
        
        Args:
            n_features: Total number of features available
            context_dim: Dimension of context vector
            alpha: Exploration parameter
            lambda_reg: Regularization parameter
        """
        self.n_features = n_features
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # LinUCB parameters
        self.A = {}  # Covariance matrices for each arm
        self.b = {}  # Reward vectors for each arm
        
        # Track arm statistics
        self.arm_counts = {}
        self.arm_rewards = {}
        
    def get_context_vector(self, data_stats: Dict) -> np.ndarray:
        """Create context vector from data characteristics."""
        context = np.array([
            data_stats.get('n_samples', 1000) / 10000,  # Normalized sample size
            data_stats.get('positive_rate', 0.1),       # Class balance
            data_stats.get('feature_correlation', 0.5), # Feature correlation
            data_stats.get('missing_rate', 0.0),        # Missing data rate
            data_stats.get('temporal_drift', 0.0)       # Temporal drift measure
        ])
        
        return context[:self.context_dim]
    
    def select_feature_subset(self, context: np.ndarray, 
                            available_features: List[str],
                            subset_size: int = 20) -> List[str]:
        """Select feature subset using LinUCB algorithm."""
        
        # Generate candidate feature subsets (arms)
        if len(available_features) <= subset_size:
            return available_features
        
        # Use predefined feature groups for efficient exploration
        feature_groups = self._create_feature_groups(available_features)
        candidate_subsets = self._generate_candidate_subsets(feature_groups, subset_size)
        
        best_subset = None
        best_ucb = -np.inf
        
        for subset in candidate_subsets:
            arm_id = self._subset_to_arm_id(subset)
            ucb_value = self._calculate_ucb(arm_id, context)
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_subset = subset
        
        logger.info(f"Selected feature subset with UCB: {best_ucb:.3f}")
        return best_subset
    
    def update_reward(self, selected_subset: List[str], reward: float, context: np.ndarray):
        """Update bandit with observed reward."""
        arm_id = self._subset_to_arm_id(selected_subset)
        
        # Initialize arm if new
        if arm_id not in self.A:
            self.A[arm_id] = self.lambda_reg * np.eye(self.context_dim)
            self.b[arm_id] = np.zeros(self.context_dim)
            self.arm_counts[arm_id] = 0
            self.arm_rewards[arm_id] = []
        
        # Update LinUCB parameters
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context
        
        # Update statistics
        self.arm_counts[arm_id] += 1
        self.arm_rewards[arm_id].append(reward)
        
        logger.debug(f"Updated arm {arm_id} with reward {reward:.3f}")
    
    def _calculate_ucb(self, arm_id: str, context: np.ndarray) -> float:
        """Calculate Upper Confidence Bound for arm."""
        if arm_id not in self.A:
            return np.inf  # Explore new arms
        
        A_inv = np.linalg.inv(self.A[arm_id])
        theta = A_inv @ self.b[arm_id]
        
        # UCB calculation
        confidence_width = self.alpha * np.sqrt(context.T @ A_inv @ context)
        ucb = context.T @ theta + confidence_width
        
        return ucb
    
    def _create_feature_groups(self, features: List[str]) -> Dict[str, List[str]]:
        """Group features by type for structured exploration."""
        groups = {
            'performance': [],
            'fitness': [],
            'position': [],
            'opponent': [],
            'temporal': [],
            'compatibility': [],
            'other': []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['performance', 'score', 'rating']):
                groups['performance'].append(feature)
            elif any(keyword in feature_lower for keyword in ['fitness', 'injury', 'health']):
                groups['fitness'].append(feature)
            elif any(keyword in feature_lower for keyword in ['position', 'versatility']):
                groups['position'].append(feature)
            elif any(keyword in feature_lower for keyword in ['opponent', 'vs', 'against']):
                groups['opponent'].append(feature)
            elif any(keyword in feature_lower for keyword in ['recent', 'form', 'window', 'rolling']):
                groups['temporal'].append(feature)
            elif any(keyword in feature_lower for keyword in ['compatibility', 'synergy', 'teammate']):
                groups['compatibility'].append(feature)
            else:
                groups['other'].append(feature)
        
        return groups
    
    def _generate_candidate_subsets(self, feature_groups: Dict[str, List[str]], 
                                  subset_size: int) -> List[List[str]]:
        """Generate candidate feature subsets for exploration."""
        candidates = []
        
        # Strategy 1: Balanced selection from each group
        group_sizes = {name: max(1, len(features) // 3) for name, features in feature_groups.items() if features}
        
        # Adjust group sizes to meet subset_size
        total_selected = sum(group_sizes.values())
        if total_selected > subset_size:
            # Scale down proportionally
            scale_factor = subset_size / total_selected
            group_sizes = {name: max(1, int(size * scale_factor)) 
                          for name, size in group_sizes.items()}
        
        balanced_subset = []
        for group_name, size in group_sizes.items():
            if feature_groups[group_name]:
                selected = np.random.choice(feature_groups[group_name], 
                                          min(size, len(feature_groups[group_name])), 
                                          replace=False).tolist()
                balanced_subset.extend(selected)
        
        candidates.append(balanced_subset[:subset_size])
        
        # Strategy 2: Performance-heavy subset
        performance_heavy = []
        for group_name in ['performance', 'temporal', 'fitness']:
            if feature_groups[group_name]:
                n_select = min(subset_size // 3, len(feature_groups[group_name]))
                selected = np.random.choice(feature_groups[group_name], n_select, replace=False).tolist()
                performance_heavy.extend(selected)
        
        # Fill remaining with other features
        remaining_features = [f for group in feature_groups.values() for f in group 
                            if f not in performance_heavy]
        if len(performance_heavy) < subset_size and remaining_features:
            n_remaining = subset_size - len(performance_heavy)
            additional = np.random.choice(remaining_features, 
                                        min(n_remaining, len(remaining_features)), 
                                        replace=False).tolist()
            performance_heavy.extend(additional)
        
        candidates.append(performance_heavy[:subset_size])
        
        # Strategy 3: Compatibility-focused subset
        compatibility_focused = []
        for group_name in ['compatibility', 'position', 'temporal']:
            if feature_groups[group_name]:
                n_select = min(subset_size // 3, len(feature_groups[group_name]))
                selected = np.random.choice(feature_groups[group_name], n_select, replace=False).tolist()
                compatibility_focused.extend(selected)
        
        # Fill remaining
        remaining_features = [f for group in feature_groups.values() for f in group 
                            if f not in compatibility_focused]
        if len(compatibility_focused) < subset_size and remaining_features:
            n_remaining = subset_size - len(compatibility_focused)
            additional = np.random.choice(remaining_features, 
                                        min(n_remaining, len(remaining_features)), 
                                        replace=False).tolist()
            compatibility_focused.extend(additional)
        
        candidates.append(compatibility_focused[:subset_size])
        
        return candidates
    
    def _subset_to_arm_id(self, subset: List[str]) -> str:
        """Convert feature subset to arm identifier."""
        return "_".join(sorted(subset)[:5])  # Use first 5 features for ID
    
    def get_arm_statistics(self) -> Dict:
        """Get statistics for all arms."""
        stats = {}
        
        for arm_id in self.arm_counts:
            rewards = self.arm_rewards[arm_id]
            stats[arm_id] = {
                'count': self.arm_counts[arm_id],
                'mean_reward': np.mean(rewards) if rewards else 0,
                'std_reward': np.std(rewards) if len(rewards) > 1 else 0,
                'total_reward': sum(rewards)
            }
        
        return stats


class MABFeatureSelector:
    """Multi-Armed Bandit Feature Selection system."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mab_config = self.config['mab_feature_selection']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
        self.bandit = None
        
    def initialize_bandit(self, feature_names: List[str], context_dim: int = 5):
        """Initialize the contextual bandit."""
        self.bandit = ContextualBandit(
            n_features=len(feature_names),
            context_dim=context_dim,
            alpha=self.mab_config['alpha'],
            lambda_reg=self.mab_config['lambda_reg']
        )
        
        self.feature_names = feature_names
        logger.info(f"Initialized MAB with {len(feature_names)} features")
    
    def evaluate_feature_subset(self, X: pd.DataFrame, y: pd.Series, 
                              feature_subset: List[str]) -> float:
        """Evaluate feature subset using cross-validation."""
        if not feature_subset or len(feature_subset) == 0:
            return 0.0
        
        # Filter features
        available_features = [f for f in feature_subset if f in X.columns]
        if len(available_features) == 0:
            return 0.0
        
        X_subset = X[available_features]
        
        # Quick evaluation with Random Forest
        rf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        try:
            # Use 3-fold CV for speed
            scores = cross_val_score(rf, X_subset, y, cv=3, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        except Exception as e:
            logger.warning(f"Feature evaluation failed: {e}")
            return 0.0
    
    def adaptive_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                 n_iterations: int = 20,
                                 subset_size: int = 20) -> Dict:
        """Run adaptive feature selection using MAB."""
        logger.info(f"Starting MAB feature selection for {n_iterations} iterations")
        
        if self.bandit is None:
            self.initialize_bandit(X.columns.tolist())
        
        # Track selection history
        selection_history = []
        
        for iteration in range(n_iterations):
            logger.info(f"MAB Iteration {iteration + 1}/{n_iterations}")
            
            # Create context from current data characteristics
            data_stats = {
                'n_samples': len(X),
                'positive_rate': y.mean(),
                'feature_correlation': X.corr().abs().mean().mean(),
                'missing_rate': X.isnull().mean().mean(),
                'temporal_drift': 0.1  # Mock temporal drift
            }
            
            context = self.bandit.get_context_vector(data_stats)
            
            # Select feature subset
            selected_features = self.bandit.select_feature_subset(
                context, self.feature_names, subset_size
            )
            
            # Evaluate subset
            reward = self.evaluate_feature_subset(X, y, selected_features)
            
            # Update bandit
            self.bandit.update_reward(selected_features, reward, context)
            
            # Record history
            selection_history.append({
                'iteration': iteration + 1,
                'selected_features': selected_features.copy(),
                'reward': reward,
                'context': context.copy(),
                'n_features_selected': len(selected_features)
            })
            
            logger.info(f"Iteration {iteration + 1}: Reward = {reward:.3f}, "
                       f"Features = {len(selected_features)}")
        
        # Find best feature subset
        best_iteration = max(selection_history, key=lambda x: x['reward'])
        best_features = best_iteration['selected_features']
        best_reward = best_iteration['reward']
        
        logger.info(f"Best feature subset found: {len(best_features)} features, "
                   f"reward = {best_reward:.3f}")
        
        return {
            'best_features': best_features,
            'best_reward': best_reward,
            'selection_history': selection_history,
            'bandit_stats': self.bandit.get_arm_statistics()
        }
    
    def feature_importance_mab(self, X: pd.DataFrame, y: pd.Series,
                             importance_threshold: float = 0.01) -> Dict:
        """Use MAB to identify most important features iteratively."""
        logger.info("Running importance-based MAB feature selection")
        
        # Start with all features
        current_features = X.columns.tolist()
        removed_features = []
        importance_history = []
        
        while len(current_features) > 10:  # Minimum feature set
            # Evaluate current feature set
            baseline_score = self.evaluate_feature_subset(X, y, current_features)
            
            # Test removing each feature
            feature_impacts = []
            
            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                test_score = self.evaluate_feature_subset(X, y, test_features)
                
                impact = baseline_score - test_score  # Positive = important feature
                feature_impacts.append((feature, impact))
            
            # Sort by impact (ascending = least important first)
            feature_impacts.sort(key=lambda x: x[1])
            
            # Remove least important feature if impact is small
            least_important_feature, impact = feature_impacts[0]
            
            if abs(impact) < importance_threshold:
                current_features.remove(least_important_feature)
                removed_features.append((least_important_feature, impact))
                
                importance_history.append({
                    'removed_feature': least_important_feature,
                    'impact': impact,
                    'remaining_features': len(current_features),
                    'baseline_score': baseline_score
                })
                
                logger.info(f"Removed {least_important_feature} (impact: {impact:.4f}), "
                           f"{len(current_features)} features remaining")
            else:
                break  # Stop if all remaining features are important
        
        return {
            'final_features': current_features,
            'removed_features': removed_features,
            'importance_history': importance_history,
            'final_score': self.evaluate_feature_subset(X, y, current_features)
        }
    
    def thompson_sampling_selection(self, X: pd.DataFrame, y: pd.Series,
                                  n_iterations: int = 15) -> Dict:
        """Feature selection using Thompson Sampling."""
        logger.info("Running Thompson Sampling feature selection")
        
        # Initialize Beta distributions for each feature
        feature_alphas = {feature: 1.0 for feature in X.columns}
        feature_betas = {feature: 1.0 for feature in X.columns}
        
        selection_history = []
        
        for iteration in range(n_iterations):
            # Sample from Beta distributions
            feature_samples = {}
            for feature in X.columns:
                sample = np.random.beta(feature_alphas[feature], feature_betas[feature])
                feature_samples[feature] = sample
            
            # Select top features based on samples
            subset_size = min(20, len(X.columns))
            selected_features = sorted(feature_samples.keys(), 
                                     key=lambda f: feature_samples[f], 
                                     reverse=True)[:subset_size]
            
            # Evaluate subset
            reward = self.evaluate_feature_subset(X, y, selected_features)
            
            # Update Beta distributions
            for feature in X.columns:
                if feature in selected_features:
                    # Feature was selected
                    feature_alphas[feature] += reward
                    feature_betas[feature] += (1 - reward)
                else:
                    # Feature was not selected (neutral update)
                    feature_betas[feature] += 0.1
            
            selection_history.append({
                'iteration': iteration + 1,
                'selected_features': selected_features.copy(),
                'reward': reward,
                'feature_samples': feature_samples.copy()
            })
            
            logger.info(f"TS Iteration {iteration + 1}: Reward = {reward:.3f}")
        
        # Find best iteration
        best_iteration = max(selection_history, key=lambda x: x['reward'])
        
        return {
            'best_features': best_iteration['selected_features'],
            'best_reward': best_iteration['reward'],
            'selection_history': selection_history,
            'final_feature_probs': {f: feature_alphas[f] / (feature_alphas[f] + feature_betas[f]) 
                                   for f in X.columns}
        }
    
    def save_mab_results(self, mab_results: Dict, method: str = "contextual",
                        output_dir: str = "artifacts"):
        """Save MAB feature selection results."""
        logger.info(f"Saving MAB results to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_path = os.path.join(output_dir, f"mab_feature_selection_{method}.pkl")
        joblib.dump(mab_results, results_path)
        
        # Save best features as text
        best_features_path = os.path.join(output_dir, f"best_features_{method}.txt")
        with open(best_features_path, 'w') as f:
            f.write(f"# Best Features Selected by {method.upper()} MAB\n")
            f.write(f"# Reward: {mab_results['best_reward']:.3f}\n")
            f.write(f"# Number of features: {len(mab_results['best_features'])}\n\n")
            
            for i, feature in enumerate(mab_results['best_features'], 1):
                f.write(f"{i:2d}. {feature}\n")
        
        # Save selection history
        if 'selection_history' in mab_results:
            history_df = pd.DataFrame(mab_results['selection_history'])
            history_path = os.path.join(output_dir, f"mab_selection_history_{method}.csv")
            history_df.to_csv(history_path, index=False)
        
        logger.info("MAB results saved successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MAB Feature Selection")
    parser.add_argument('--data', required=True,
                       help='Feature data file')
    parser.add_argument('--method', choices=['contextual', 'importance', 'thompson'],
                       default='contextual', help='MAB method to use')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of MAB iterations')
    parser.add_argument('--subset-size', type=int, default=20,
                       help='Feature subset size')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info(f"Starting MAB feature selection with {args.method} method")
    
    try:
        # Initialize selector
        selector = MABFeatureSelector()
        
        # Load data
        data = pd.read_csv(args.data)
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col not in 
                       ['player_id', 'match_date', 'team_id', 'selected']]
        
        X = data[feature_cols]
        y = data['selected'] if 'selected' in data.columns else pd.Series(np.random.binomial(1, 0.1, len(data)))
        
        logger.info(f"Input data: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Positive rate: {y.mean():.3f}")
        
        # Run MAB feature selection
        if args.method == 'contextual':
            results = selector.adaptive_feature_selection(
                X, y, args.iterations, args.subset_size
            )
        elif args.method == 'importance':
            results = selector.feature_importance_mab(X, y)
        elif args.method == 'thompson':
            results = selector.thompson_sampling_selection(X, y, args.iterations)
        
        # Save results
        selector.save_mab_results(results, args.method, args.output)
        
        print(f"\n=== MAB Feature Selection Results ({args.method.upper()}) ===")
        print(f"Best Reward: {results['best_reward']:.3f}")
        print(f"Selected Features: {len(results['best_features'])}")
        print(f"Top 10 Features:")
        
        for i, feature in enumerate(results['best_features'][:10], 1):
            print(f"  {i:2d}. {feature}")
        
        if args.method == 'contextual' and 'bandit_stats' in results:
            print(f"\nBandit Statistics:")
            stats = results['bandit_stats']
            print(f"  Arms explored: {len(stats)}")
            if stats:
                best_arm = max(stats.keys(), key=lambda k: stats[k]['mean_reward'])
                print(f"  Best arm reward: {stats[best_arm]['mean_reward']:.3f}")
        
        logger.info("MAB feature selection completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"MAB feature selection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
