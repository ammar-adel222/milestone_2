import pandas as pd

# Columns to drop during preprocessing
DROP_COLUMNS = [
    # ID columns - no predictive value (data leakage via identifiers)
    'QueryID', 'ResponseID',
    # Text / URL columns - irrelevant for tabular classification
    'QueryName', 'ResponseName', 'AboutText', 'Background',
    'ShortDescrip', 'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice',
    'HeaderImage', 'LegalNotice', 'Reviews', 'SupportedLanguages',
    'SupportEmail', 'SupportURL', 'Website', 'PCMinReqsText',
    'PCRecReqsText', 'LinuxMinReqsText', 'LinuxRecReqsText',
    'MacMinReqsText', 'MacRecReqsText',
]

TARGET_COLUMN = 'GamePopularity'


def preprocess(df):
    """Apply shared preprocessing steps to the dataframe.

    Steps:
        1. Parse ReleaseDate and extract ReleaseYear
        2. Drop irrelevant, text, and potential-leakage columns
        3. Convert boolean features to integer
        4. Drop PriceCurrency

    Args:
        df: pandas DataFrame to preprocess (modified in place).

    Returns:
        The modified DataFrame.
    """
    # Handle ReleaseDate
    if 'ReleaseDate' in df.columns:
        df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
        df['ReleaseYear'] = df['ReleaseDate'].dt.year
        df.drop(columns=['ReleaseDate'], inplace=True)

    # Drop irrelevant / leakage columns
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Convert boolean features to integer
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)

    # Remove PriceCurrency (single constant value, not useful)
    if 'PriceCurrency' in df.columns:
        df.drop(columns=['PriceCurrency'], inplace=True)

    return df
