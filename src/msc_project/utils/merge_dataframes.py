import pandas as pd
import argparse

def merge_dataframes(df1, df2, match_columns = None):
    if match_columns is None:
        match_columns = df1.columns.intersection(df2.columns).tolist()

    merged_df = pd.merge(df1, df2, on=match_columns)
    return merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two dataframes')
    parser.add_argument('df1', help='Path to first csv')
    parser.add_argument('df2', help='Path to second csv')
    parser.add_argument('output', help='Path to output csv')
    parser.add_argument('--match_columns', nargs='+', help='Columns to match on')

    args = parser.parse_args()

    df1 = pd.read_csv(args.df1)
    df2 = pd.read_csv(args.df2)

    merged_df = merge_dataframes(df1, df2, args.match_columns)
    merged_df.to_csv(args.output, index=False)