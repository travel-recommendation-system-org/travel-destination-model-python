# evaluator.py
# Model Evaluation Metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import pandas as pd

class Evaluator:
    """
    Evaluation class for recommendation systems
    """

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """
        Calculate Root Mean Square Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE score
        """
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """
        Calculate Mean Absolute Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAE score
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_precision_at_k(recommended, relevant, k):
        """
        Calculate Precision@K

        Args:
            recommended: List of recommended items
            relevant: Set of relevant items
            k: Cut-off position

        Returns:
            Precision@K score
        """
        if k > len(recommended):
            k = len(recommended)

        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])

        return relevant_count / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(recommended, relevant, k):
        """
        Calculate Recall@K

        Args:
            recommended: List of recommended items
            relevant: Set of relevant items
            k: Cut-off position

        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0

        if k > len(recommended):
            k = len(recommended)

        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])

        return relevant_count / len(relevant) if relevant else 0.0

    @staticmethod
    def calculate_ndcg_at_k(recommended, relevant, k):
        """
        Calculate Normalized Discounted Cumulative Gain@K

        Args:
            recommended: List of recommended items
            relevant: Set of relevant items
            k: Cut-off position

        Returns:
            NDCG@K score
        """
        if k > len(recommended):
            k = len(recommended)

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k], 1):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_map_at_k(recommended_lists, relevant_lists, k):
        """
        Calculate Mean Average Precision@K

        Args:
            recommended_lists: List of recommendation lists per user
            relevant_lists: List of relevant items per user
            k: Cut-off position

        Returns:
            MAP@K score
        """
        ap_scores = []

        for recommended, relevant in zip(recommended_lists, relevant_lists):
            if not relevant:
                continue

            precision_sum = 0.0
            relevant_count = 0

            for i, item in enumerate(recommended[:k], 1):
                if item in relevant:
                    relevant_count += 1
                    precision_sum += relevant_count / i

            if relevant_count > 0:
                ap = precision_sum / min(len(relevant), k)
                ap_scores.append(ap)

        return np.mean(ap_scores) if ap_scores else 0.0

    def evaluate_ncf_model(self, model, X_test, y_test):
        """
        Evaluate NCF model on test data

        Args:
            model: Trained NCF model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        user_test = X_test[:, 0]
        attraction_test = X_test[:, 1]

        y_pred = model.model.predict([user_test, attraction_test],
                                    batch_size=1024, verbose=0).flatten()

        # Calculate metrics
        metrics = {
            'rmse': self.calculate_rmse(y_test, y_pred),
            'mae': self.calculate_mae(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred)
        }

        # Convert to rating scale (1-5)
        y_test_rating = y_test * 4 + 1
        y_pred_rating = y_pred * 4 + 1

        metrics['rmse_rating'] = self.calculate_rmse(y_test_rating, y_pred_rating)
        metrics['mae_rating'] = self.calculate_mae(y_test_rating, y_pred_rating)

        return metrics

    def improved_evaluation(self, hybrid_recommender, data_loader, n_users=30, k=10):
        """
        Improved evaluation with hold-out items

        Args:
            hybrid_recommender: Hybrid recommender instance
            data_loader: DataLoader instance
            n_users: Number of users to evaluate
            k: Number of recommendations

        Returns:
            DataFrame with evaluation results
        """
        results = []

        # Sélectionner des utilisateurs avec suffisamment d'historique
        user_interaction_counts = data_loader.interactions['user_id'].value_counts()
        active_users = user_interaction_counts[user_interaction_counts >= 3].index.tolist()

        if len(active_users) > n_users:
            test_users = np.random.choice(active_users, n_users, replace=False)
        else:
            test_users = active_users

        print(f"Evaluating on {len(test_users)} users...")

        for user_id in test_users:
            try:
                # Obtenir l'historique complet
                user_history = data_loader.get_user_history(user_id)

                if len(user_history) < 2:
                    continue  # Besoin d'au moins 2 items pour train/test

                # Séparer en train/test (hold-out)
                train_items = user_history.sample(frac=0.7, random_state=42)
                test_items = user_history.drop(train_items.index)

                if len(test_items) == 0:
                    continue

                # Obtenir des recommandations (elles excluent automatiquement les items notés)
                recommendations = hybrid_recommender.recommend_for_user(
                    user_id=user_id,
                    k=k,
                    use_cnn=True,
                    exclude_rated=True,  # Exclure les items déjà notés
                    force_recompute=True
                )

                if not recommendations:
                    continue

                recommended_ids = [rec[0] for rec in recommendations]
                test_ids = set(test_items['attraction_id'].values)

                # Calculer les métriques
                hits = len([rid for rid in recommended_ids if rid in test_ids])
                precision = hits / k if len(recommended_ids) > 0 else 0
                recall = hits / len(test_ids) if len(test_ids) > 0 else 0

                # Calculer NDCG
                dcg = 0.0
                for i, rid in enumerate(recommended_ids[:k], 1):
                    if rid in test_ids:
                        dcg += 1.0 / np.log2(i + 1)

                idcg = 0.0
                for i in range(1, min(len(test_ids), k) + 1):
                    idcg += 1.0 / np.log2(i + 1)

                ndcg = dcg / idcg if idcg > 0 else 0

                results.append({
                    'user_id': user_id,
                    'hits': hits,
                    'precision': precision,
                    'recall': recall,
                    'ndcg': ndcg,
                    'test_items': len(test_ids)
                })

            except Exception as e:
                print(f"Error for user {user_id}: {str(e)[:100]}...")
                continue

        if results:
            results_df = pd.DataFrame(results)
            print("\nEvaluation Results:")
            print(f"Number of users evaluated: {len(results_df)}")
            print(f"Average Precision@{k}: {results_df['precision'].mean():.4f}")
            print(f"Average Recall@{k}: {results_df['recall'].mean():.4f}")
            print(f"Average NDCG@{k}: {results_df['ndcg'].mean():.4f}")
            print(f"Total Hits: {results_df['hits'].sum()}")
            print(f"Average test items per user: {results_df['test_items'].mean():.1f}")

            # Distribution des hits
            hits_distribution = results_df['hits'].value_counts().sort_index()
            print("\nHits Distribution:")
            for hits, count in hits_distribution.items():
                print(f"  {hits} hits: {count} users ({count/len(results_df)*100:.1f}%)")

            return results_df
        else:
            print("No valid evaluation results.")
            return pd.DataFrame()

    def compare_alpha_values(self, hybrid_recommender, data_loader, alphas=[0.0, 0.3, 0.5, 0.7, 1.0], n_users=20, k=10):
        """
        Compare different alpha values

        Args:
            hybrid_recommender: Hybrid recommender instance
            data_loader: DataLoader instance
            alphas: List of alpha values to test
            n_users: Number of users for testing
            k: Number of recommendations

        Returns:
            DataFrame with comparison results
        """
        print("Comparing different alpha values...")

        comparison_results = []

        # Select test users
        user_interaction_counts = data_loader.interactions['user_id'].value_counts()
        active_users = user_interaction_counts[user_interaction_counts >= 3].index.tolist()

        if len(active_users) > n_users:
            test_users = np.random.choice(active_users, n_users, replace=False)
        else:
            test_users = active_users

        for alpha in alphas:
            hybrid_recommender.set_alpha(alpha)

            scores = []
            for user_id in test_users:
                try:
                    recommendations = hybrid_recommender.recommend_for_user(user_id, k=5)
                    if recommendations:
                        # Score moyen des recommandations
                        avg_score = np.mean([score for _, score, _ in recommendations])
                        scores.append(avg_score)
                except:
                    continue

            if scores:
                comparison_results.append({
                    'alpha': alpha,
                    'avg_recommendation_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'n_users': len(scores)
                })

        if comparison_results:
            results_df = pd.DataFrame(comparison_results)
            print("\nAlpha Comparison Results:")
            print(results_df.to_string(index=False))

            return results_df
        else:
            print("No comparison results available.")
            return pd.DataFrame()

    def generate_evaluation_report(self, ncf_model, hybrid_recommender, data_loader, X_test, y_test):
        """
        Generate comprehensive evaluation report

        Args:
            ncf_model: NCF model instance
            hybrid_recommender: Hybrid recommender instance
            data_loader: DataLoader instance
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation report
        """
        report = {}

        # NCF Model Evaluation
        print("\n" + "=" * 60)
        print("NCF MODEL EVALUATION")
        print("=" * 60)

        ncf_metrics = self.evaluate_ncf_model(ncf_model, X_test, y_test)
        report['ncf_metrics'] = ncf_metrics

        print("NCF Model Metrics:")
        for metric, value in ncf_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Recommendation Evaluation
        print("\n" + "=" * 60)
        print("RECOMMENDATION EVALUATION")
        print("=" * 60)

        rec_results = self.improved_evaluation(hybrid_recommender, data_loader, n_users=30, k=10)
        report['recommendation_evaluation'] = rec_results.to_dict('records') if not rec_results.empty else None

        # Alpha Comparison
        print("\n" + "=" * 60)
        print("ALPHA PARAMETER ANALYSIS")
        print("=" * 60)

        alpha_results = self.compare_alpha_values(hybrid_recommender, data_loader,
                                                  alphas=[0.0, 0.3, 0.5, 0.7, 1.0],
                                                  n_users=20, k=10)
        report['alpha_comparison'] = alpha_results.to_dict('records') if not alpha_results.empty else None

        # System Analysis
        print("\n" + "=" * 60)
        print("SYSTEM ANALYSIS")
        print("=" * 60)

        analysis = hybrid_recommender.analyze_recommendations(n_users=20, k=10)
        report['system_analysis'] = analysis

        return report