

def med_filter(df, pmed, cmed):
    """
    Takes in a df and 2 lists of med time points for
    Parkinson's and Controls.

    Returns a new df where only those med
    time points are included (filter all others).
    """
    df_group1 = df[((df['diagnosis'] == 1) & (df['medtimepoint'].isin(pmed))) |
             ((df['diagnosis'] == 0) & (df['medtimepoint'].isin(cmed)))]
    return df_group1


def sample_filter(df, lower = None, upper = None):
    """
    Takes in a df and a lower and upper thresholds
    to filter sample lengths.

    Returns a new df that only contains
    sample lengths within those thresholds.

    if lower and upper are none, the df
    filters for the sample with highest count.
    If the highest count is 400 then the upper
    and lower equivalently
    are set to 400.
    """
    if lower == None or upper == None:
        shc = df['sample_len'].value_counts().head(1).index[0]
        return df[df['sample_len'] == shc]
    else:
        return df[(df['sample_len'] >= lower) & (df['sample_len'] <= upper)]
