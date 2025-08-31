#!/usr/bin/env python3
"""
Transfer Learning for Cross-League Domain Adaptation.
Adapts models trained on one league to perform well on another league.
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class DomainAdversarialNetwork(nn.Module):
    """Domain Adversarial Neural Network for transfer learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_domains: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Selection predictor
        self.selection_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier (adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_domains),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x, alpha=1.0):
        """Forward pass with gradient reversal."""
        features = self.feature_extractor(x)
        
        # Selection prediction
        selection_pred = self.selection_predictor(features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        
        return selection_pred, domain_pred, features


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class TransferLearningAdapter:
    """Transfer learning adapter for cross-league model adaptation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.transfer_config = self.config['transfer_learning']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
    def prepare_domain_data(self, source_data: pd.DataFrame, 
                          target_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for domain adaptation."""
        logger.info("Preparing domain adaptation data...")
        
        # Align features between domains
        common_features = list(set(source_data.columns) & set(target_data.columns))
        common_features = [col for col in common_features if col not in 
                          ['player_id', 'match_date', 'team_id', 'league', 'selected']]
        
        logger.info(f"Using {len(common_features)} common features")
        
        # Prepare source domain data
        X_source = source_data[common_features].fillna(0).values
        y_source = source_data['selected'].values if 'selected' in source_data.columns else None
        domain_source = np.zeros(len(X_source))  # Domain 0
        
        # Prepare target domain data
        X_target = target_data[common_features].fillna(0).values
        y_target = target_data['selected'].values if 'selected' in target_data.columns else None
        domain_target = np.ones(len(X_target))  # Domain 1
        
        # Combine data
        X_combined = np.vstack([X_source, X_target])
        domain_combined = np.concatenate([domain_source, domain_target])
        
        if y_source is not None and y_target is not None:
            y_combined = np.concatenate([y_source, y_target])
        else:
            y_combined = None
        
        # Normalize features
        scaler = StandardScaler()
        X_combined = scaler.fit_transform(X_combined)
        
        logger.info(f"Prepared domain data: {X_combined.shape}")
        return X_combined, y_combined, domain_combined, scaler, common_features
    
    def train_domain_adversarial_model(self, X: np.ndarray, y: np.ndarray, 
                                     domains: np.ndarray) -> DomainAdversarialNetwork:
        """Train Domain Adversarial Network."""
        logger.info("Training Domain Adversarial Network...")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        domain_tensor = torch.LongTensor(domains)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor, domain_tensor)
        dataloader = DataLoader(dataset, batch_size=self.transfer_config['batch_size'], shuffle=True)
        
        # Initialize model
        model = DomainAdversarialNetwork(
            input_dim=X.shape[1],
            hidden_dim=self.transfer_config['hidden_dim'],
            n_domains=2,
            dropout=self.transfer_config['dropout']
        )
        
        # Optimizers
        optimizer = optim.Adam(model.parameters(), lr=self.transfer_config['learning_rate'])
        
        # Loss functions
        selection_criterion = nn.BCELoss()
        domain_criterion = nn.NLLLoss()
        
        # Training loop
        model.train()
        n_epochs = self.transfer_config['epochs']
        
        for epoch in range(n_epochs):
            total_loss = 0
            selection_loss_sum = 0
            domain_loss_sum = 0
            
            for batch_X, batch_y, batch_domains in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                alpha = 2.0 / (1.0 + np.exp(-10 * epoch / n_epochs)) - 1  # Gradual alpha increase
                selection_pred, domain_pred, features = model(batch_X, alpha)
                
                # Calculate losses
                selection_loss = selection_criterion(selection_pred, batch_y)
                domain_loss = domain_criterion(domain_pred, batch_domains)
                
                # Combined loss
                lambda_domain = self.transfer_config['lambda_domain']
                total_batch_loss = selection_loss + lambda_domain * domain_loss
                
                # Backward pass
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                selection_loss_sum += selection_loss.item()
                domain_loss_sum += domain_loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Total={total_loss:.3f}, "
                          f"Selection={selection_loss_sum:.3f}, Domain={domain_loss_sum:.3f}")
        
        logger.info("Domain Adversarial training completed")
        return model
    
    def fine_tune_target_domain(self, model: DomainAdversarialNetwork,
                              target_X: np.ndarray, target_y: np.ndarray) -> DomainAdversarialNetwork:
        """Fine-tune model on target domain data."""
        logger.info("Fine-tuning on target domain...")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(target_X)
        y_tensor = torch.FloatTensor(target_y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Fine-tuning with lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=self.transfer_config['learning_rate'] * 0.1)
        criterion = nn.BCELoss()
        
        model.train()
        
        for epoch in range(self.transfer_config['fine_tune_epochs']):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Only use selection predictor for fine-tuning
                selection_pred, _, _ = model(batch_X, alpha=0)
                loss = criterion(selection_pred, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logger.info(f"Fine-tune Epoch {epoch}: Loss={total_loss:.3f}")
        
        logger.info("Fine-tuning completed")
        return model
    
    def adapt_sklearn_model(self, source_model, source_data: pd.DataFrame,
                          target_data: pd.DataFrame) -> Dict:
        """Adapt sklearn model using feature alignment and reweighting."""
        logger.info("Adapting sklearn model for target domain...")
        
        # Prepare data
        X_combined, y_combined, domains, scaler, features = self.prepare_domain_data(
            source_data, target_data
        )
        
        # Split back into source and target
        n_source = len(source_data)
        X_source = X_combined[:n_source]
        X_target = X_combined[n_source:]
        
        if y_combined is not None:
            y_source = y_combined[:n_source]
            y_target = y_combined[n_source:]
        else:
            y_source = source_data['selected'].values
            y_target = None
        
        # Method 1: Feature alignment with PCA
        pca = PCA(n_components=min(50, X_combined.shape[1]))
        X_source_pca = pca.fit_transform(X_source)
        X_target_pca = pca.transform(X_target)
        
        # Method 2: Instance reweighting based on domain similarity
        weights = self._calculate_instance_weights(X_source, X_target)
        
        # Retrain source model with weights
        adapted_model = source_model.__class__(**source_model.get_params())
        
        # Check if model supports sample weights
        if hasattr(adapted_model, 'fit') and 'sample_weight' in adapted_model.fit.__code__.co_varnames:
            adapted_model.fit(X_source_pca, y_source, sample_weight=weights)
        else:
            adapted_model.fit(X_source_pca, y_source)
        
        # Evaluate adaptation if target labels available
        adaptation_metrics = {}
        if y_target is not None:
            y_pred_proba = adapted_model.predict_proba(X_target_pca)[:, 1]
            
            adaptation_metrics = {
                'target_roc_auc': roc_auc_score(y_target, y_pred_proba),
                'target_pr_auc': average_precision_score(y_target, y_pred_proba),
                'feature_alignment_variance': pca.explained_variance_ratio_.sum()
            }
            
            logger.info(f"Adapted model ROC-AUC on target: {adaptation_metrics['target_roc_auc']:.3f}")
        
        return {
            'adapted_model': adapted_model,
            'scaler': scaler,
            'pca': pca,
            'feature_names': features,
            'adaptation_metrics': adaptation_metrics,
            'instance_weights': weights
        }
    
    def _calculate_instance_weights(self, X_source: np.ndarray, 
                                  X_target: np.ndarray) -> np.ndarray:
        """Calculate instance weights for domain adaptation."""
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors in target domain for each source instance
        nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn_model.fit(X_target)
        
        distances, _ = nn_model.kneighbors(X_source)
        avg_distances = distances.mean(axis=1)
        
        # Convert distances to weights (closer to target = higher weight)
        max_distance = avg_distances.max()
        weights = 1 - (avg_distances / max_distance)
        weights = np.clip(weights, 0.1, 2.0)  # Prevent extreme weights
        
        return weights
    
    def create_league_embeddings(self, league_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create league-specific embeddings for transfer learning."""
        logger.info("Creating league embeddings...")
        
        league_embeddings = {}
        
        for league_name, data in league_data.items():
            logger.info(f"Processing {league_name}...")
            
            # Extract numerical features
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col not in 
                            ['player_id', 'match_date', 'team_id', 'selected']]
            
            if len(numerical_cols) == 0:
                continue
            
            X = data[numerical_cols].fillna(0).values
            
            # Create league embedding using PCA
            pca = PCA(n_components=min(20, X.shape[1]))
            league_embedding = pca.fit_transform(X)
            
            # Calculate league statistics
            league_stats = {
                'mean_embedding': league_embedding.mean(axis=0),
                'std_embedding': league_embedding.std(axis=0),
                'n_samples': len(data),
                'feature_variance_explained': pca.explained_variance_ratio_.sum(),
                'pca_model': pca
            }
            
            league_embeddings[league_name] = league_stats
            logger.info(f"{league_name}: {len(data)} samples, "
                       f"variance explained: {league_stats['feature_variance_explained']:.3f}")
        
        return league_embeddings
    
    def measure_domain_distance(self, source_embedding: Dict, 
                              target_embedding: Dict) -> Dict:
        """Measure distance between source and target domains."""
        logger.info("Measuring domain distance...")
        
        # Euclidean distance between mean embeddings
        euclidean_distance = np.linalg.norm(
            source_embedding['mean_embedding'] - target_embedding['mean_embedding']
        )
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = cosine_similarity(
            source_embedding['mean_embedding'].reshape(1, -1),
            target_embedding['mean_embedding'].reshape(1, -1)
        )[0, 0]
        
        # Wasserstein distance approximation
        wasserstein_approx = np.mean(np.abs(
            source_embedding['std_embedding'] - target_embedding['std_embedding']
        ))
        
        distance_metrics = {
            'euclidean_distance': euclidean_distance,
            'cosine_similarity': cosine_sim,
            'wasserstein_approx': wasserstein_approx,
            'domain_shift_severity': self._classify_domain_shift(euclidean_distance, cosine_sim)
        }
        
        logger.info(f"Domain distance - Euclidean: {euclidean_distance:.3f}, "
                   f"Cosine: {cosine_sim:.3f}")
        
        return distance_metrics
    
    def _classify_domain_shift(self, euclidean_dist: float, cosine_sim: float) -> str:
        """Classify severity of domain shift."""
        if cosine_sim > 0.9 and euclidean_dist < 1.0:
            return "MINIMAL"
        elif cosine_sim > 0.7 and euclidean_dist < 2.0:
            return "MODERATE"
        elif cosine_sim > 0.5:
            return "SIGNIFICANT"
        else:
            return "SEVERE"
    
    def select_adaptation_strategy(self, domain_distance: Dict, 
                                 source_data_size: int,
                                 target_data_size: int) -> str:
        """Select best adaptation strategy based on domain characteristics."""
        logger.info("Selecting adaptation strategy...")
        
        shift_severity = domain_distance['domain_shift_severity']
        cosine_sim = domain_distance['cosine_similarity']
        
        # Decision logic
        if shift_severity == "MINIMAL":
            strategy = "DIRECT_TRANSFER"
        elif shift_severity == "MODERATE" and target_data_size > 100:
            strategy = "FINE_TUNING"
        elif shift_severity in ["SIGNIFICANT", "SEVERE"] and target_data_size > 500:
            strategy = "DOMAIN_ADVERSARIAL"
        elif target_data_size < 50:
            strategy = "FEATURE_ALIGNMENT"
        else:
            strategy = "GRADUAL_ADAPTATION"
        
        logger.info(f"Selected strategy: {strategy} (shift: {shift_severity})")
        return strategy
    
    def apply_transfer_learning(self, source_model, source_data: pd.DataFrame,
                              target_data: pd.DataFrame, strategy: str = None) -> Dict:
        """Apply transfer learning with selected strategy."""
        logger.info("Applying transfer learning...")
        
        # Prepare data
        X_combined, y_combined, domains, scaler, features = self.prepare_domain_data(
            source_data, target_data
        )
        
        n_source = len(source_data)
        X_source = X_combined[:n_source]
        X_target = X_combined[n_source:]
        
        if y_combined is not None:
            y_source = y_combined[:n_source]
            y_target = y_combined[n_source:]
        else:
            y_source = source_data['selected'].values
            y_target = target_data.get('selected', pd.Series()).values
        
        # Auto-select strategy if not provided
        if strategy is None:
            # Create embeddings for strategy selection
            source_embedding = {'mean_embedding': X_source.mean(axis=0), 'std_embedding': X_source.std(axis=0)}
            target_embedding = {'mean_embedding': X_target.mean(axis=0), 'std_embedding': X_target.std(axis=0)}
            
            domain_distance = self.measure_domain_distance(source_embedding, target_embedding)
            strategy = self.select_adaptation_strategy(domain_distance, len(source_data), len(target_data))
        
        logger.info(f"Applying {strategy} strategy")
        
        if strategy == "DIRECT_TRANSFER":
            # Simply apply source model to target domain
            adapted_model = source_model
            adaptation_info = {'strategy': strategy, 'scaler': scaler}
            
        elif strategy == "FINE_TUNING":
            # Fine-tune source model on target data
            adapted_model = source_model.__class__(**source_model.get_params())
            
            if len(y_target) > 0 and not np.isnan(y_target).all():
                adapted_model.fit(X_target, y_target)
            else:
                adapted_model = source_model  # Fallback
            
            adaptation_info = {'strategy': strategy, 'scaler': scaler}
            
        elif strategy == "DOMAIN_ADVERSARIAL":
            # Train domain adversarial network
            if len(y_target) > 0 and not np.isnan(y_combined).any():
                dan_model = self.train_domain_adversarial_model(X_combined, y_combined, domains)
                adapted_model = dan_model
            else:
                adapted_model = source_model  # Fallback
            
            adaptation_info = {'strategy': strategy, 'scaler': scaler, 'model_type': 'neural_network'}
            
        elif strategy == "FEATURE_ALIGNMENT":
            # Use PCA alignment
            adaptation_result = self.adapt_sklearn_model(source_model, source_data, target_data)
            adapted_model = adaptation_result['adapted_model']
            adaptation_info = adaptation_result
            adaptation_info['strategy'] = strategy
            
        else:  # GRADUAL_ADAPTATION
            # Gradual adaptation with weighted training
            adaptation_result = self.adapt_sklearn_model(source_model, source_data, target_data)
            adapted_model = adaptation_result['adapted_model']
            adaptation_info = adaptation_result
            adaptation_info['strategy'] = strategy
        
        # Evaluate adaptation if possible
        if len(y_target) > 0 and not np.isnan(y_target).all():
            try:
                if hasattr(adapted_model, 'predict_proba'):
                    y_pred_proba = adapted_model.predict_proba(X_target)[:, 1]
                elif hasattr(adapted_model, 'forward'):  # Neural network
                    adapted_model.eval()
                    with torch.no_grad():
                        X_target_tensor = torch.FloatTensor(X_target)
                        y_pred_proba, _, _ = adapted_model(X_target_tensor)
                        y_pred_proba = y_pred_proba.numpy().flatten()
                else:
                    y_pred_proba = adapted_model.predict(X_target)
                
                adaptation_metrics = {
                    'target_roc_auc': roc_auc_score(y_target, y_pred_proba),
                    'target_pr_auc': average_precision_score(y_target, y_pred_proba)
                }
                
                adaptation_info['adaptation_metrics'] = adaptation_metrics
                logger.info(f"Adaptation ROC-AUC: {adaptation_metrics['target_roc_auc']:.3f}")
                
            except Exception as e:
                logger.warning(f"Adaptation evaluation failed: {e}")
        
        return {
            'adapted_model': adapted_model,
            'adaptation_info': adaptation_info,
            'feature_names': features,
            'scaler': scaler
        }
    
    def save_transfer_learning_artifacts(self, adaptation_result: Dict,
                                       source_league: str, target_league: str,
                                       output_dir: str = "artifacts"):
        """Save transfer learning artifacts."""
        logger.info(f"Saving transfer learning artifacts to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save adapted model
        model_path = os.path.join(output_dir, f"adapted_model_{source_league}_to_{target_league}.pkl")
        joblib.dump(adaptation_result, model_path)
        
        # Save metadata
        metadata = {
            'source_league': source_league,
            'target_league': target_league,
            'adaptation_strategy': adaptation_result['adaptation_info']['strategy'],
            'feature_names': adaptation_result['feature_names'],
            'adaptation_date': pd.Timestamp.now().isoformat(),
            'config': self.transfer_config
        }
        
        if 'adaptation_metrics' in adaptation_result['adaptation_info']:
            metadata['performance'] = adaptation_result['adaptation_info']['adaptation_metrics']
        
        metadata_path = os.path.join(output_dir, f"transfer_metadata_{source_league}_to_{target_league}.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info("Transfer learning artifacts saved successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Transfer learning domain adaptation")
    parser.add_argument('--source-data', required=True,
                       help='Source domain data file')
    parser.add_argument('--target-data', required=True,
                       help='Target domain data file')
    parser.add_argument('--source-model', required=True,
                       help='Source domain model file')
    parser.add_argument('--source-league', default='Premier_League',
                       help='Source league name')
    parser.add_argument('--target-league', default='La_Liga',
                       help='Target league name')
    parser.add_argument('--strategy', choices=['DIRECT_TRANSFER', 'FINE_TUNING', 
                                             'DOMAIN_ADVERSARIAL', 'FEATURE_ALIGNMENT'],
                       help='Transfer learning strategy')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info(f"Starting transfer learning: {args.source_league} → {args.target_league}")
    
    try:
        # Initialize adapter
        adapter = TransferLearningAdapter()
        
        # Load data
        source_data = pd.read_csv(args.source_data)
        target_data = pd.read_csv(args.target_data)
        source_model = joblib.load(args.source_model)
        
        logger.info(f"Source data: {len(source_data)} samples")
        logger.info(f"Target data: {len(target_data)} samples")
        
        # Apply transfer learning
        adaptation_result = adapter.apply_transfer_learning(
            source_model, source_data, target_data, args.strategy
        )
        
        # Save artifacts
        adapter.save_transfer_learning_artifacts(
            adaptation_result, args.source_league, args.target_league, args.output
        )
        
        print(f"\n=== Transfer Learning Results ===")
        print(f"Strategy: {adaptation_result['adaptation_info']['strategy']}")
        
        if 'adaptation_metrics' in adaptation_result['adaptation_info']:
            metrics = adaptation_result['adaptation_info']['adaptation_metrics']
            print(f"Target ROC-AUC: {metrics.get('target_roc_auc', 0):.3f}")
            print(f"Target PR-AUC: {metrics.get('target_pr_auc', 0):.3f}")
        
        print(f"Artifacts saved to: {args.output}")
        
        logger.info("Transfer learning completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Transfer learning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
