import pandas as pd
import numpy as np

# Columns to drop during preprocessing (same as the notebook)
DROP_COLUMNS = [
    # ID columns
    'QueryID', 'ResponseID',
    # Text / URL / irrelevant columns
    'QueryName', 'ResponseName', 'AboutText', 'Background',
    'ShortDescrip', 'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice',
    'HeaderImage', 'LegalNotice', 'Reviews', 'SupportedLanguages',
    'SupportEmail', 'SupportURL', 'Website', 'PCMinReqsText',
    'PCRecReqsText', 'LinuxMinReqsText', 'LinuxRecReqsText',
    'MacMinReqsText', 'MacRecReqsText',
    # Additional columns dropped in the notebook
    'ReleaseDate', 'RequiredAge', 'PackageCount', 'DLCCount',
    'SteamSpyOwnersVariance', 'SteamSpyPlayersVariance',
    'ScreenshotCount', 'PriceCurrency',
]

REGRESSION_TARGET = 'RecommendationCount'

# Columns that get log1p transform (skewness reduction, from notebook)
LOG_TRANSFORM_COLUMNS = [
    'SteamSpyOwners',
    'SteamSpyPlayersEstimate',
    'PriceInitial',
    'PriceFinal',
]

# Numeric columns used for outlier clipping (from notebook)
CLIP_COLUMNS = [
    'DemoCount', 'DeveloperCount', 'Metacritic', 'MovieCount',
    'RecommendationCount', 'PublisherCount', 'SteamSpyOwners',
    'SteamSpyPlayersEstimate', 'AchievementCount',
    'AchievementHighlightedCount', 'PriceInitial', 'PriceFinal',
]

# Classification target from M2 — drop it if present during regression
CLASSIFICATION_TARGET = 'GamePopularity'


def regression_preprocess(df, is_training=True):
    """Shared preprocessing for regression (mirrors notebook logic).

    Args:
        df: pandas DataFrame (modified in place).
        is_training: if True, applies clipping & row filtering.

    Returns:
        The modified DataFrame.
    """
    # 1. Replace whitespace-only strings with NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # 2. Drop irrelevant / text columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # 3. Drop classification target if present
    if CLASSIFICATION_TARGET in df.columns:
        df.drop(columns=[CLASSIFICATION_TARGET], inplace=True)

    # 4. Convert boolean features to integer
    bool_cols = df.select_dtypes(include=['bool']).columns
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # 2. Outlier Clipping
    # -------------------------------------
    import os
    import json
    
    quantiles_file = 'saved_models/regression_quantiles.json'

    if is_training:
        os.makedirs('saved_models', exist_ok=True)
        quantiles = {}
        for col in CLIP_COLUMNS:
            if col in df.columns:
                lower = float(df[col].quantile(0.01))
                upper = float(df[col].quantile(0.99))
                quantiles[col] = {'lower': lower, 'upper': upper}
                df[col] = df[col].clip(lower, upper)
        
        with open(quantiles_file, 'w') as f:
            json.dump(quantiles, f)
    else:
        if os.path.exists(quantiles_file):
            with open(quantiles_file, 'r') as f:
                quantiles = json.load(f)
            for col in CLIP_COLUMNS:
                if col in df.columns and col in quantiles:
                    df[col] = df[col].clip(quantiles[col]['lower'], quantiles[col]['upper'])
        else:
            print("WARNING: regression_quantiles.json not found, skipping outlier clipping.")

    # 6. Remove rows with negative numeric values (training only)
    if is_training:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df.drop(df[(df[num_cols] < 0).any(axis=1)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 7. Log1p transform on skewed columns
    for c in LOG_TRANSFORM_COLUMNS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    return df
