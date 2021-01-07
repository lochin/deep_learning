from sklearn.datasets import load_files
import pandas as pd

def uza_to_csv():
    uza = load_files('.\\uza', encoding='utf-8')

    df = pd.DataFrame([uza.data, uza.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame(uza.target_names)
    targets.columns = ['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv('20_newsgroup.csv')

uza_to_csv()