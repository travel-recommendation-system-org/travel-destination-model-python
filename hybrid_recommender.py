# hybrid_recommender.py
# Hybrid Recommender combining NCF and CNN
import numpy as np
from collections import defaultdict
import pandas as pd

class HybridRecommender:
    """
    Hybrid Recommender combining NCF and CNN-based recommendations
    """

    def __init__(self, ncf_model, cnn_extractor, data_loader,
                 attractions_features=None, alpha=0.7):
        """
        Initialize the hybrid recommender

        Args:
            ncf_model: Trained NCF model
            cnn_extractor: CNN feature extractor
            data_loader: DataLoader instance
            attractions_features: Dictionary of attraction features
            alpha: Weight for NCF score (1-alpha for CNN)
        """
        self.ncf_model = ncf_model
        self.cnn_extractor = cnn_extractor
        self.data_loader = data_loader
        self.attractions_features = attractions_features
        self.alpha = alpha

        # Cache for user recommendations
        self.user_recommendation_cache = {}

        print(f"Hybrid Recommender initialized with alpha={alpha}")

    def set_alpha(self, alpha):
        """
        Set the weight parameter for NCF vs CNN

        Args:
            alpha: New alpha value (0-1)
        """
        self.alpha = alpha
        print(f"Alpha updated to {alpha}")

    def combine_scores(self, ncf_scores, cnn_scores, normalization=True):
        """
        Combine NCF and CNN scores

        Args:
            ncf_scores: Dictionary {attraction_id: ncf_score}
            cnn_scores: Dictionary {attraction_id: cnn_score}
            normalization: Whether to normalize scores

        Returns:
            Dictionary {attraction_id: combined_score}
        """
        all_attractions = set(ncf_scores.keys()) | set(cnn_scores.keys())
        combined_scores = {}

        # Normalize scores if requested
        if normalization and ncf_scores:
            ncf_values = np.array(list(ncf_scores.values()))
            if len(ncf_values) > 0 and ncf_values.max() > ncf_values.min():
                ncf_values = (ncf_values - ncf_values.min()) / (ncf_values.max() - ncf_values.min())
                ncf_scores = dict(zip(ncf_scores.keys(), ncf_values))

        if normalization and cnn_scores:
            cnn_values = np.array(list(cnn_scores.values()))
            if len(cnn_values) > 0 and cnn_values.max() > cnn_values.min():
                cnn_values = (cnn_values - cnn_values.min()) / (cnn_values.max() - cnn_values.min())
                cnn_scores = dict(zip(cnn_scores.keys(), cnn_values))

        # Combine scores
        for attraction_id in all_attractions:
            ncf_score = ncf_scores.get(attraction_id, 0)
            cnn_score = cnn_scores.get(attraction_id, 0)

            # Weighted combination
            combined_score = self.alpha * ncf_score + (1 - self.alpha) * cnn_score
            combined_scores[attraction_id] = combined_score

        return combined_scores

    def recommend_for_user(self, user_id, k=10, use_cnn=True,
                          exclude_rated=True, force_recompute=False):
        """
        Generate hybrid recommendations for a user

        Args:
            user_id: User ID
            k: Number of recommendations
            use_cnn: Whether to use CNN features
            exclude_rated: Whether to exclude already rated attractions
            force_recompute: Force recomputation even if cached

        Returns:
            List of (attraction_id, score, type) tuples
        """
        # Check cache
        cache_key = (user_id, k, use_cnn, exclude_rated, self.alpha)
        if not force_recompute and cache_key in self.user_recommendation_cache:
            return self.user_recommendation_cache[cache_key]

        # Get NCF recommendations
        ncf_recommendations = self.ncf_model.recommend_top_k(
            user_id, k=k*2,  # Get more for combination
            exclude_rated=exclude_rated,
            data_loader=self.data_loader
        )

        ncf_scores = {att_id: score for att_id, score in ncf_recommendations}

        if not use_cnn or self.attractions_features is None:
            # Return only NCF recommendations
            recommendations = [(att_id, score, 'ncf')
                              for att_id, score in ncf_recommendations[:k]]
            self.user_recommendation_cache[cache_key] = recommendations
            return recommendations

        # Get user's rated attractions for CNN similarity
        user_history = self.data_loader.get_user_history(user_id)

        if len(user_history) > 0:
            # Get user's top-rated attractions for visual similarity
            top_rated = user_history.nlargest(3, 'rating')['attraction_id'].values

            cnn_scores = defaultdict(float)

            for rated_attraction in top_rated:
                if rated_attraction in self.attractions_features:
                    target_features = self.attractions_features[rated_attraction]

                    # Find visually similar attractions
                    similar = self.cnn_extractor.find_similar_attractions(
                        target_features,
                        self.attractions_features,
                        k=k*3
                    )

                    # Aggregate scores
                    for att_id, similarity in similar:
                        if exclude_rated:
                            if att_id not in user_history['attraction_id'].values:
                                cnn_scores[att_id] = max(cnn_scores[att_id], similarity)
                        else:
                            cnn_scores[att_id] = max(cnn_scores[att_id], similarity)
        else:
            # If no history, use all attractions with random scores
            all_attractions = list(self.attractions_features.keys())
            cnn_scores = {att_id: np.random.rand() * 0.5 for att_id in all_attractions}

        # Combine NCF and CNN scores
        combined_scores = self.combine_scores(ncf_scores, cnn_scores)

        # Sort and get top-k
        if combined_scores:
            sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Fallback to NCF only
            sorted_combined = sorted(ncf_scores.items(), key=lambda x: x[1], reverse=True)

        # Prepare final recommendations with type info
        recommendations = []
        for att_id, score in sorted_combined[:k]:
            # Determine recommendation type
            ncf_score = ncf_scores.get(att_id, 0)
            cnn_score = cnn_scores.get(att_id, 0)

            if ncf_score > 0 and cnn_score > 0:
                # Both contribute
                ncf_ratio = ncf_score / (ncf_score + cnn_score) if (ncf_score + cnn_score) > 0 else 0

                if ncf_ratio > 0.7:
                    rec_type = 'ncf'
                elif ncf_ratio < 0.3:
                    rec_type = 'cnn'
                else:
                    rec_type = 'hybrid'
            elif ncf_score > 0:
                rec_type = 'ncf'
            elif cnn_score > 0:
                rec_type = 'cnn'
            else:
                rec_type = 'baseline'

            recommendations.append((att_id, score, rec_type))

        # Cache results
        self.user_recommendation_cache[cache_key] = recommendations

        return recommendations

    def recommend_by_image(self, image_path, user_id=None, k=10,
                          use_ncf=True, attractions_features=None):
        """
        Recommend based on an input image

        Args:
            image_path: Path to input image
            user_id: Optional user ID for personalization
            k: Number of recommendations
            use_ncf: Whether to use NCF for personalization
            attractions_features: Attraction features dictionary

        Returns:
            List of (attraction_id, score, type) tuples
        """
        if attractions_features is None:
            attractions_features = self.attractions_features

        if attractions_features is None:
            raise ValueError("Attraction features not provided")

        # Get CNN-based recommendations
        cnn_recommendations = self.cnn_extractor.find_similar_by_image(
            image_path, attractions_features, k=k*2
        )

        cnn_scores = {att_id: score for att_id, score in cnn_recommendations}

        if not use_ncf or user_id is None:
            # Return only CNN recommendations
            recommendations = [(att_id, score, 'cnn')
                              for att_id, score in cnn_recommendations[:k]]
            return recommendations

        # Get NCF recommendations for user
        ncf_recommendations = self.ncf_model.recommend_top_k(
            user_id, k=k*2, exclude_rated=True,
            data_loader=self.data_loader
        )

        ncf_scores = {att_id: score for att_id, score in ncf_recommendations}

        # Combine scores
        combined_scores = self.combine_scores(ncf_scores, cnn_scores)

        # Sort and get top-k
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Prepare final recommendations with type info
        recommendations = []
        for att_id, score in sorted_combined[:k]:
            # Determine recommendation type
            ncf_score = ncf_scores.get(att_id, 0)
            cnn_score = cnn_scores.get(att_id, 0)

            if ncf_score > 0 and cnn_score > 0:
                rec_type = 'hybrid'
            elif cnn_score > 0:
                rec_type = 'cnn'
            else:
                rec_type = 'ncf'

            recommendations.append((att_id, score, rec_type))

        return recommendations

    def get_recommendation_details(self, recommendations):
        """
        Get detailed information for recommended attractions

        Args:
            recommendations: List of (attraction_id, score, type) tuples

        Returns:
            DataFrame with attraction details
        """
        details = []

        for att_id, score, rec_type in recommendations:
            try:
                attraction_info = self.data_loader.get_attraction_info(att_id)

                # Clean attraction name
                name = attraction_info.get('attraction_url', '')
                if isinstance(name, str):
                    # Remove common prefixes and clean the name
                    if 'Reviews-' in name:
                        name = name.split('Reviews-')[1]
                    if '.html' in name:
                        name = name.replace('.html', '')
                    name = name.replace('_', ' ').replace('-', ' ').title()
                else:
                    name = f'Attraction {att_id}'

                category = attraction_info.get('category', 'Unknown')
                if pd.isna(category):
                    category = 'Not specified'

                rating = attraction_info.get('rating', 0)
                if pd.isna(rating):
                    rating = 0

                details.append({
                    'attraction_id': att_id,
                    'name': name[:80],  # Limit name length
                    'category': category,
                    'rating': rating,
                    'score': round(score, 4),
                    'type': rec_type,
                    'has_images': len(self.data_loader.get_attraction_image_paths(att_id)) > 0
                })
            except Exception as e:
                # Skip attractions with errors
                continue

        return pd.DataFrame(details)

    def analyze_recommendations(self, n_users=20, k=10):
        """
        Analyze recommendation patterns

        Args:
            n_users: Number of users to analyze
            k: Number of recommendations per user

        Returns:
            Analysis results
        """
        print("Analyzing recommendation patterns...")

        # Select random users
        user_ids = np.random.choice(range(self.data_loader.num_users),
                                    min(n_users, self.data_loader.num_users),
                                    replace=False)

        type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        score_stats = []

        for user_id in user_ids:
            try:
                recommendations = self.recommend_for_user(user_id, k=k, use_cnn=True)

                for att_id, score, rec_type in recommendations:
                    type_counts[rec_type] += 1
                    score_stats.append(score)

                    # Get attraction category
                    try:
                        info = self.data_loader.get_attraction_info(att_id)
                        category = info.get('category', 'Unknown')
                        category_counts[category] += 1
                    except:
                        category_counts['Unknown'] += 1

            except Exception as e:
                continue

        print(f"\nAnalysis Results (based on {len(user_ids)} users):")
        print("-" * 40)

        print("\nRecommendation Types:")
        total_recs = sum(type_counts.values())
        for rec_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_recs * 100) if total_recs > 0 else 0
            print(f"  {rec_type}: {count} ({percentage:.1f}%)")

        print("\nTop Categories Recommended:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_recs * 100) if total_recs > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")

        if score_stats:
            print(f"\nScore Statistics:")
            print(f"  Average: {np.mean(score_stats):.4f}")
            print(f"  Std Dev: {np.std(score_stats):.4f}")
            print(f"  Min: {np.min(score_stats):.4f}")
            print(f"  Max: {np.max(score_stats):.4f}")

        return {
            'type_counts': dict(type_counts),
            'category_counts': dict(category_counts),
            'score_stats': score_stats
        }