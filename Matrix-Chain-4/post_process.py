import pandas as pd

def calculate_durations(df):
    dfs= df.drop_duplicates('case:concept:name', keep='first')[['case:concept:name','time:start']]
    dfe = df.drop_duplicates('case:concept:name', keep='last')[['case:concept:name','time:end']]
    dfm = dfs.merge(dfe, on='case:concept:name')
    dfm['duration'] = dfm.apply(lambda row: row['time:end']-row['time:start'], axis=1)
    return dfm


def get_reference_time(variants, df):
    ref = {}
    for variant in variants:
        # ref[variant] = df[df['case:concept:name'].str.contains(variant)].iloc[[0]]['duration'].values[0]
        ref[variant] = df[df['case:concept:name'].str.contains(variant)]['duration'].min()

    return ref

def add_deviation(row,refs):
    name = row['case:concept:name'].split('R')[0]
    return (row['duration'] - refs[name])/float(refs[name])

def add_deviations_to_df(df,refs):
    df['deviation'] = df.apply(lambda row: add_deviation(row,refs), axis=1)

def mask_deviations(df, c):
    df['deviation'].where(df['deviation'] < c, 1, inplace=True)
    df['deviation'].where(df['deviation'] >= c, 0, inplace=True)



def add_duration_deviations(csv_in_path, csv_out_path, variants, cut_off):
    df = pd.read_csv(csv_in_path)
    dfm = calculate_durations(df)
    ref = get_reference_time(variants, dfm)
    add_deviations_to_df(dfm, ref)
    mask_deviations(dfm, cut_off)

    dfm.to_csv(csv_out_path, encoding='utf-8')