import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from dbfread import DBF

if not os.path.exists('PAimg'):
    os.makedirs('PAimg')
if not os.path.exists('logPAdata'):
    os.makedirs('logPAdata')

result_data = []

dbf_files_path = 'PA'
dbf_files = [f for f in os.listdir(dbf_files_path) if f.endswith('.dbf')]
for dbf_file in dbf_files:
    dbf_file_path = os.path.join(dbf_files_path, dbf_file)

    dbf_data = DBF(dbf_file_path)
    data = []
    for record in dbf_data:
        data.append(record)
    df = pd.DataFrame(data)

    df['logP'] = np.log(df['P']).replace([np.inf, -np.inf], np.nan)
    df['logA'] = np.log(df['A']).replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['logP', 'logA'])

    plt.scatter(df['logA'], df['logP'], label=dbf_file[:-4])

    model = LinearRegression()
    model.fit(df[['logA']], df['logP'])
    predicted_logP = model.predict(df[['logA']])

    plt.plot(df['logA'], predicted_logP, color='red')

    # Add fitting equation and R-squared value on the plot
    coeff = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(df[['logA']], df['logP'])
    equation = f'y = {coeff:.8f}x + {intercept:.8f}\n$R^2$ = {r_squared:.8f}'

    plt.text(0.1, 0.8, equation, fontsize=12, transform=plt.gca().transAxes)

    plt.title(f"Scatter plot and linear fit: {dbf_file[:-4]}")
    plt.xlabel("logA")
    plt.ylabel("logP")
    plt.savefig(f"PAimg/{dbf_file[:-4]}.png")
    log_pa_data = pd.DataFrame({'logA': df['logA'], 'logP': df['logP']})
    log_pa_data.to_csv(f"logPAdata/{dbf_file[:-4]}.csv", index=False)
    plt.clf()

    result_data.append({
        'File': dbf_file[:-4],
        'Slope': coeff,
        'Intercept': intercept,
        'R-squared': r_squared
    })

result_df = pd.DataFrame(result_data)
result_df.to_csv('results.csv', index=False)

print("save")
