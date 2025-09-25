import numpy as np
from sklearn.cluster import KMeans


def random_selection(cheap_ratings, n_expensive, seed):
    """Random selection strategy."""
    np.random.seed(seed+)
    return np.random.choice(len(cheap_ratings), n_expensive, replace=False)

def stratified_selection(cheap_ratings, n_expensive, seed):
    """Stratified selection by cheap rating quantiles."""
    np.random.seed(seed)
    n_items = len(cheap_ratings)
    
    # Create quantile-based strata
    quantiles = np.linspace(0, 1, n_expensive + 1)
    thresholds = np.quantile(cheap_ratings, quantiles)
    
    selected = []
    for i in range(n_expensive):
        # Find items in this stratum
        if i == 0:
            mask = cheap_ratings <= thresholds[i + 1]
        elif i == n_expensive - 1:
            mask = cheap_ratings >= thresholds[i]
        else:
            mask = (cheap_ratings >= thresholds[i]) & (cheap_ratings <= thresholds[i + 1])
        
        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            selected.append(np.random.choice(candidates))
        else:
            # Fallback to random selection if stratum is empty
            remaining = [j for j in range(n_items) if j not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
    
    return np.array(selected[:n_expensive])

def QBC_selection(cheap_ratings, cheap_ratings_2, n_expensive, seed):
    """Select items where cheap raters disagree most."""
    np.random.seed(seed)
    disagreement = np.abs(cheap_ratings - cheap_ratings_2)
    return np.argsort(disagreement)[-n_expensive:]

def hybrid_selection(cheap_ratings, cheap_ratings_2, n_expensive, seed):
    """Hybrid: stratified + disagreement."""
    # Select half using stratification
    np.random.seed(seed)
    n_strat = n_expensive // 2
    strat_indices = stratified_selection(cheap_ratings, n_strat, seed)
    
    # Select remaining using disagreement (excluding already selected)
    disagreement = np.abs(cheap_ratings - cheap_ratings_2)
    disagreement = disagreement.copy()  # Make a copy to avoid modifying original
    disagreement[strat_indices] = -1  # Use -1 instead of -inf to avoid overflow
    disagree_indices = np.argsort(disagreement)[-(n_expensive - n_strat):]
    return np.concatenate([strat_indices, disagree_indices])

# def active_selection(cheap_ratings, n_expensive):
#     """
#     Active selection: iteratively select items to minimize ICC uncertainty.
#     Uses a greedy approximation.
#     """
#     n_items = len(cheap_ratings)
#     selected = []
    
#     for step in range(n_expensive):
#         best_item = None
#         best_score = -np.inf
        
#         candidates = [i for i in range(n_items) if i not in selected]
        
#         for candidate in candidates:
#             # Create temporary selection including this candidate
#             temp_selected = selected + [candidate]
            
#             # Calculate score based on range coverage and representativeness
#             temp_cheap = cheap_ratings[temp_selected]
#             temp_range = np.max(temp_cheap) - np.min(temp_cheap)
#             full_range = np.max(cheap_ratings) - np.min(cheap_ratings)
#             range_score = temp_range / full_range if full_range > 0 else 0
            
#             # Add disagreement bonus if available
#             if len(temp_selected) > 1:
#                 disagreement_score = np.std(temp_cheap)
#             else:
#                 disagreement_score = 0
            
#             total_score = range_score + 0.3 * disagreement_score
            
#             if total_score > best_score:
#                 best_score = total_score
#                 best_item = candidate
        
#         if best_item is not None:
#             selected.append(best_item)
    
#     return np.array(selected)

def cluster_selection(cheap_ratings, n_expensive, seed, random_state=None):
    """
    Cluster-based selection: Use K-means clustering on cheap ratings and 
    select representative items from each cluster to ensure diverse coverage.
    """
    np.random.seed(seed)
    n_items = len(cheap_ratings)
    
    # Reshape for sklearn (needs 2D array)
    ratings_2d = cheap_ratings.reshape(-1, 1)
    
    # Use K-means with k=n_expensive clusters
    kmeans = KMeans(n_clusters=n_expensive, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(ratings_2d)
    
    selected = []
    for cluster_id in range(n_expensive):
        # Find items in this cluster that haven't been selected yet
        cluster_items = np.array([i for i in np.where(cluster_labels == cluster_id)[0] 
                                if i not in selected])
        
        if len(cluster_items) > 0:
            # Select item closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id][0]
            distances = np.abs(cheap_ratings[cluster_items] - cluster_center)
            closest_idx = cluster_items[np.argmin(distances)]
            selected.append(closest_idx)
        else:
            # Fallback if cluster is empty or all items used
            remaining = [j for j in range(n_items) if j not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
    
    return np.array(selected[:n_expensive])

def maximum_variation_selection(cheap_ratings, n_expensive, seed):
    """
    Variance-weighted selection: Select items that maximize the variance
    of the selected subset while maintaining representativeness.
    This helps preserve the between-item variance component crucial for ICC.
    """
    np.random.seed(seed)
    selected = []
    
    # Start with the item having median rating to anchor the selection
    median_idx = np.argmin(np.abs(cheap_ratings - np.median(cheap_ratings)))
    selected.append(median_idx)
    
    # Iteratively add items that maximize subset variance
    for step in range(1, n_expensive):
        best_item = None
        best_variance = -1
        
        candidates = [i for i in range(len(cheap_ratings)) if i not in selected]
        
        for candidate in candidates:
            temp_selected = selected + [candidate]
            temp_ratings = cheap_ratings[temp_selected]
            temp_variance = np.var(temp_ratings)
            
            # Bonus for maintaining good distribution coverage
            temp_range = np.max(temp_ratings) - np.min(temp_ratings)
            full_range = np.max(cheap_ratings) - np.min(cheap_ratings)
            coverage_bonus = (temp_range / full_range) * 0.1 if full_range > 0 else 0
            
            score = temp_variance # + coverage_bonus
            
            if score > best_variance:
                best_variance = score
                best_item = candidate
        
        if best_item is not None:
            selected.append(best_item)
    
    return np.array(selected)

def density_based_selection(cheap_ratings, n_expensive, seed):
    """
    Density-based selection: Select items from both high-density regions
    (where many items cluster) and low-density regions (outliers/extremes).
    This balances representation of typical cases with edge cases.
    """
    from scipy.stats import gaussian_kde
    np.random.seed(seed)
    
    # Create KDE of cheap ratings
    kde = gaussian_kde(cheap_ratings)
    densities = kde(cheap_ratings)
    
    # Normalize densities
    densities = (densities - densities.min()) / (densities.max() - densities.min())
    
    # Select mix of high-density and low-density items
    n_high_density = n_expensive // 2
    n_low_density = n_expensive - n_high_density
    
    # Select high-density items (representative of typical cases)
    high_density_indices = np.argsort(densities)[-n_high_density*2:]  # Get top candidates
    high_density_selected = np.random.choice(high_density_indices, n_high_density, replace=False)
    
    # Select low-density items (outliers/edge cases), excluding already selected
    remaining_indices = [i for i in range(len(cheap_ratings)) if i not in high_density_selected]
    remaining_densities = densities[remaining_indices]
    
    if len(remaining_indices) >= n_low_density:
        low_density_candidates = np.argsort(remaining_densities)[:n_low_density*2]  # Bottom candidates
        if len(low_density_candidates) >= n_low_density:
            low_density_selected_rel = np.random.choice(low_density_candidates, n_low_density, replace=False)
            low_density_selected = [remaining_indices[i] for i in low_density_selected_rel]
        else:
            low_density_selected = remaining_indices[:n_low_density]
    else:
        low_density_selected = remaining_indices
    
    # Combine selections
    selected = np.concatenate([high_density_selected, low_density_selected])
    return selected[:n_expensive]
