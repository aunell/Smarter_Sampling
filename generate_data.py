import numpy as np
import scipy.stats as stats
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score


def calculate_icc_krip(rater1, rater2, metric='interval'):
    """
    Compute Krippendorff's alpha reliability coefficient for two raters.

    Parameters:
    rater1, rater2 : array-like
        Ratings from two raters. Must be same length.
        Use np.nan for missing values.
    metric : str
        'nominal', 'ordinal', or 'interval'.

    Returns:
    float: Krippendorff's alpha
    """
    # Stack raters into matrix (n_raters x n_items)
    data = np.vstack([rater1, rater2]).astype(float)
    n_raters, n_items = data.shape

    # Handle missing data
    mask = ~np.isnan(data)
    valid_counts = mask.sum(axis=0)
    valid_items = valid_counts > 1
    data = data[:, valid_items]
    mask = mask[:, valid_items]

    # Define distance metric
    def delta(x, y):
        if metric == 'nominal':
            return 0 if x == y else 1
        elif metric == 'ordinal':
            return ((x - y) / (np.nanmax(data) - np.nanmin(data))) ** 2
        elif metric == 'interval':
            return (x - y) ** 2
        else:
            raise ValueError("metric must be 'nominal', 'ordinal', or 'interval'")

    # Observed disagreement
    Do = 0.0
    total_pairs = 0
    for j in range(data.shape[1]):
        ratings = data[:, j][mask[:, j]]
        for i in range(len(ratings)):
            for k in range(i + 1, len(ratings)):
                Do += delta(ratings[i], ratings[k])
                total_pairs += 1
    Do /= total_pairs

    # Expected disagreement
    all_ratings = data[mask]
    De = 0.0
    total_pairs = 0
    for i in range(len(all_ratings)):
        for k in range(i + 1, len(all_ratings)):
            De += delta(all_ratings[i], all_ratings[k])
            total_pairs += 1
    De /= total_pairs

    return 1 - Do / De if De != 0 else 1.0

def calculate_icc_cohen(rater1, rater2, weights='linear'):
    """
    Calculate weighted Cohen's kappa for ordinal data.
    
    Parameters:
    rater1, rater2: array-like, ratings from two raters
    weights: str, 'linear' or 'quadratic'
    
    Returns:
    float: Weighted Cohen's kappa coefficient
    """
    return cohen_kappa_score(rater1, rater2, weights='linear')

def calculate_icc_icc(ratings1, ratings2):
    """
    Main ICC calculation method - backwards compatible with original code.
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating arrays must have same length")
    
    n = len(ratings1)
    if n < 2:
        return np.nan
    
    # Create dataframe in long format for pingouin
    data = []
    for i in range(n):
        data.append({'Subject': i, 'Rater': 'Rater1', 'Rating': ratings1[i]})
        data.append({'Subject': i, 'Rater': 'Rater2', 'Rating': ratings2[i]})
    
    df = pd.DataFrame(data)
    
    # Calculate ICC(3,k) - this is the most appropriate for two-rater consistency
    icc_result = pg.intraclass_corr(data=df, targets='Subject', raters='Rater', ratings='Rating')
    
    # Get ICC(3,k) which is "ICC3k" in pingouin
    # breakpoint()
    icc3_row = icc_result[icc_result['Type'] == 'ICC3']
    
    if len(icc3_row) > 0:
        return max(0,float(icc3_row['ICC'].iloc[0]))
    else:
        raise ValueError("ICC(3,k) calculation failed")

def generate_realistic_ratings(n_items, true_item_effects, between_var, within_var):
    """
    Generate ratings from same distribution with dependent noise
    """
    n_raters = 3
    all_ratings = np.zeros((n_items, n_raters))
    
    # Generate base item effects that all raters will share
    base_effects = true_item_effects + np.random.normal(0, np.sqrt(within_var/2), n_items)
    
    # Generate correlated noise for cheap raters
    cheap_rater_bias = np.random.normal(0, np.sqrt(within_var*2), n_items)  # Stronger variation
    cheap_rater_2_bias = 0.8 * cheap_rater_bias + np.random.normal(0, np.sqrt(within_var/2), n_items)  # Correlated with first cheap rater
    
    # Generate less variable noise for expensive raters
    expensive_bias = np.random.normal(0, np.sqrt(within_var/4), n_items)  # Much less variation
    
    # Combine effects and biases
    cheap_ratings = 5.0 + base_effects + cheap_rater_bias
    expensive_ratings = 5.0 + base_effects + expensive_bias  # Reference rating with less noise
    cheap_ratings_2 = 5.0 + base_effects + cheap_rater_2_bias
    return cheap_ratings, cheap_ratings_2, expensive_ratings

def generate_data(n_items=100, n_expensive=10, true_icc=0.8, random_state=42, distribution_type='gaussian', distribution_params=None, optimize_icc=True, verbose=False, categorical=True):
    """
    Generate synthetic rating data with controlled ICC using various distributions.
    
    Parameters:
    -----------
    n_items : int
        Number of items to generate ratings for
    n_expensive : int 
        Number of expensive ratings to generate
    true_icc : float
        Target ICC value to achieve
    random_state : int
        Random seed for reproducibility
    distribution_type : str
        Type of distribution to use: 'gaussian', 'skewed_normal', 'gamma', 'nonparametric'
    distribution_params : dict
        Parameters for the chosen distribution
    optimize_icc : bool
        Whether to automatically optimize parameters to match target ICC
    categorical : bool
        Whether to discretize ratings into categories (1-5) or keep continuous
    """
    # Calculate variance components based on desired ICC
    total_var = 1.0
    between_var = true_icc * total_var  # Item variance
    within_var = (1 - true_icc) * total_var  # Error variance
    
    # Set default parameters if none provided
    if distribution_params is None:
        distribution_params = _get_default_params(distribution_type)
    
    # Optimize parameters if requested
    if optimize_icc:
        distribution_params = _optimize_distribution_params(
            distribution_type, distribution_params, between_var, within_var, true_icc, n_items
        )
    
    # Generate true item effects based on distribution type
    true_item_effects = _generate_item_effects(
        distribution_type, distribution_params, between_var, n_items
    )
    
    
    #Generate ratings from same distribution with dependent noise
    cheap_ratings, cheap_ratings_2, expensive_ratings = generate_realistic_ratings(n_items, true_item_effects, between_var, within_var)

    true_scores = expensive_ratings
    if categorical:
        cheap_ratings = np.clip(np.round(cheap_ratings), 1, 5)
        expensive_ratings = np.clip(np.round(expensive_ratings), 1, 5)
        cheap_ratings_2 = np.clip(np.round(cheap_ratings_2), 1, 5)
    
    # Calculate oracle ICC using the proper method
    oracle_icc = calculate_icc_icc(cheap_ratings, expensive_ratings)
    # oracle_icc = calculate_cohens_kappa(cheap_ratings, expensive_ratings)
    
    # Store distribution info for reference
    distribution_info = {
        'type': distribution_type,
        'params': distribution_params,
        'between_var': between_var,
        'within_var': within_var,
        'categorical': categorical
    }
    if verbose:    
        print(f"  Target ICC: {true_icc:.3f}")
        print(f"  Oracle ICC: {oracle_icc:.3f}")
        print(f"  Difference: {abs(oracle_icc - true_icc):.4f}")
        print(f"  Distribution: {distribution_info['type']}")
        print(f"  Categorical: {distribution_info['categorical']}")
        
        if distribution_info['type'] != 'gaussian':
            print(f"  Distribution params: {distribution_info['params']}")
        
        print(f"  Between-item variance: {distribution_info['between_var']:.4f}")
        print(f"  Within-item variance: {distribution_info['within_var']:.4f}")
        print(f"  Cheap ratings range: [{cheap_ratings.min():.2f}, {cheap_ratings.max():.2f}]")
        print(f"  Expensive ratings range: [{expensive_ratings.min():.2f}, {expensive_ratings.max():.2f}]")
        
        # Print distribution statistics
        print(f"  Item effects - Mean: {np.mean(true_item_effects):.3f}, "
            f"Std: {np.std(true_item_effects):.3f}, "
            f"Skew: {stats.skew(true_item_effects):.3f}")
    return cheap_ratings, cheap_ratings_2, expensive_ratings, true_item_effects, true_scores, oracle_icc, distribution_info

def _get_default_params(distribution_type):
    """Get default parameters for each distribution type"""
    defaults = {
        'gaussian': {},
        'skewed_normal': {'skewness': 3.0},
        'gamma': {'shape': 2.0},
        'nonparametric': {'base_distribution': 'mixed', 'mixture_weights': [0.3, 0.3, 0.3]}
    }
    return defaults.get(distribution_type, {})

def _generate_item_effects(distribution_type, params, between_var, n_items):
    """Generate item effects using specified distribution"""
    if distribution_type == 'gaussian':
        return np.random.normal(0, np.sqrt(between_var), n_items)
    
    elif distribution_type == 'skewed_normal':
        skewness = params.get('skewness', 3.0)
        effects = stats.skewnorm.rvs(
            a=skewness, 
            loc=0, 
            scale=np.sqrt(between_var), 
            size=n_items
        )
        return effects
    
    elif distribution_type == 'gamma':
        shape = params.get('shape', 2.0)
        # For gamma: var = shape * scale^2, so scale = sqrt(var/shape)
        scale = np.sqrt(between_var / shape)
        effects = np.random.gamma(shape, scale, n_items)
        # Center around 0
        return effects - np.mean(effects)
    
    elif distribution_type == 'nonparametric':
        return _generate_nonparametric_effects(params, between_var, n_items)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def _generate_error_term(distribution_type, params, within_var):
    """Generate error term using specified distribution"""
    if distribution_type == 'gaussian':
        return np.random.normal(0, np.sqrt(within_var))
    
    elif distribution_type == 'skewed_normal':
        skewness = params.get('skewness', 3.0)
        return stats.skewnorm.rvs(a=skewness/2, loc=0, scale=np.sqrt(within_var))
    
    elif distribution_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = np.sqrt(within_var / shape)
        return np.random.gamma(shape, scale) - shape * scale
    
    elif distribution_type == 'nonparametric':
        return _generate_nonparametric_error(params, within_var)
    
    else:
        return np.random.normal(0, np.sqrt(within_var))

def _generate_nonparametric_effects(params, between_var, n_items):
    """Generate non-parametric item effects"""
    base_dist = params.get('base_distribution', 'mixed')
    
    if base_dist == 'mixed':
        weights = params.get('mixture_weights', [0.3, 0.3, 0.3])
        n_exp = int(weights[0] * n_items)
        n_unif = int(weights[1] * n_items)
        n_norm = n_items - n_exp - n_unif
        
        # Generate different components
        exp_effects = np.random.exponential(scale=np.sqrt(between_var), size=n_exp)
        unif_effects = np.random.uniform(-np.sqrt(3*between_var), np.sqrt(3*between_var), size=n_unif)
        norm_effects = np.random.normal(0, np.sqrt(between_var), size=n_norm)
        
        # Combine and shuffle
        all_effects = np.concatenate([exp_effects, unif_effects, norm_effects])
        np.random.shuffle(all_effects)
        
        # Center the distribution
        return all_effects - np.mean(all_effects)
    
    elif base_dist == 'empirical_bootstrap':
        # Create an empirical distribution by bootstrapping from a skewed seed
        seed_data = params.get('seed_data', [0.1, 0.2, 0.8, 1.2, 2.1, 0.3, 0.7, 1.8, 0.4, 1.5])
        
        # Bootstrap to create larger sample
        bootstrapped = np.random.choice(seed_data, size=n_items, replace=True)
        # Add some noise and scale to match desired variance
        noise = np.random.normal(0, 0.1, n_items)
        scaled_effects = (bootstrapped + noise) * np.sqrt(between_var) / np.std(bootstrapped + noise)
        return scaled_effects - np.mean(scaled_effects)
    
    else:
        # Fallback to mixed
        return _generate_nonparametric_effects({'base_distribution': 'mixed'}, between_var, n_items)

def _generate_nonparametric_error(params, within_var):
    """Generate non-parametric error term"""
    # Use different error distributions randomly
    rand_choice = np.random.random()
    if rand_choice < 0.5:
        return np.random.exponential(scale=np.sqrt(within_var)) - np.sqrt(within_var)
    elif rand_choice < 0.8:
        return np.random.uniform(-np.sqrt(3*within_var), np.sqrt(3*within_var))
    else:
        return np.random.normal(0, np.sqrt(within_var))

def _optimize_distribution_params(distribution_type, initial_params, between_var, within_var, true_icc, n_items):
    """Optimize distribution parameters to achieve target ICC"""
    best_params = initial_params.copy()
    best_error = float('inf')
    target_tolerance = 0.01
    max_attempts = 20
    
    if distribution_type == 'skewed_normal':
        # Try different skewness values
        for skewness in np.linspace(0.5, 5.0, 15):
            for attempt in range(3):  # Multiple attempts due to randomness
                test_params = initial_params.copy()
                test_params['skewness'] = skewness
                
                empirical_icc = _test_icc_with_params(
                    distribution_type, test_params, between_var, within_var, n_items
                )
                error = abs(empirical_icc - true_icc)
                
                if error < best_error:
                    best_error = error
                    best_params = test_params.copy()
                
                if error < target_tolerance:
                    return best_params
    
    elif distribution_type == 'gamma':
        # Try different shape parameters
        for shape in np.linspace(0.5, 5.0, 15):
            for attempt in range(3):
                test_params = initial_params.copy()
                test_params['shape'] = shape
                
                empirical_icc = _test_icc_with_params(
                    distribution_type, test_params, between_var, within_var, n_items
                )
                error = abs(empirical_icc - true_icc)
                
                if error < best_error:
                    best_error = error
                    best_params = test_params.copy()
                
                if error < target_tolerance:
                    return best_params
    
    elif distribution_type == 'nonparametric':
        # Try different base distributions and mixture weights
        base_distributions = ['empirical_bootstrap']
        mixture_options = [[0.3, 0.3, 0.3], [0.4, 0.2, 0.4], [0.5, 0.2, 0.3], [.3,.2,.5]]
        
        for base_dist in base_distributions:
            for weights in mixture_options:
                for attempt in range(5):  # More attempts for non-parametric
                    test_params = initial_params.copy()
                    test_params['base_distribution'] = base_dist
                    test_params['mixture_weights'] = weights
                    
                    empirical_icc = _test_icc_with_params(
                        distribution_type, test_params, between_var, within_var, n_items
                    )
                    error = abs(empirical_icc - true_icc)
                    
                    if error < best_error:
                        best_error = error
                        best_params = test_params.copy()
                    
                    if error < target_tolerance:
                        return best_params
    
    return best_params

def _test_icc_with_params(distribution_type, params, between_var, within_var, n_items):
    """Test ICC calculation with given parameters (for optimization)"""
    # Generate small test dataset
    test_n_items = min(50, n_items)
    
    # Generate test item effects
    if distribution_type == 'gaussian':
        test_effects = np.random.normal(0, np.sqrt(between_var), test_n_items)
    elif distribution_type == 'skewed_normal':
        skewness = params.get('skewness', 3.0)
        test_effects = stats.skewnorm.rvs(
            a=skewness, loc=0, scale=np.sqrt(between_var), size=test_n_items
        )
    elif distribution_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = np.sqrt(between_var / shape)
        effects = np.random.gamma(shape, scale, test_n_items)
        test_effects = effects - np.mean(effects)
    elif distribution_type == 'nonparametric':
        # Simplified test for nonparametric
        test_effects = np.random.exponential(scale=np.sqrt(between_var), size=test_n_items)
        test_effects = test_effects - np.mean(test_effects)
    
    # Generate test ratings
    test_ratings_1 = []
    test_ratings_2 = []
    
    for i in range(test_n_items):
        error1 = _generate_error_term(distribution_type, params, within_var)
        error2 = _generate_error_term(distribution_type, params, within_var)
        
        test_ratings_1.append(5.0 + test_effects[i] + error1)
        test_ratings_2.append(5.0 + test_effects[i] + error2)
    
    # Calculate empirical ICC
    return calculate_icc_icc(np.array(test_ratings_1), np.array(test_ratings_2))
