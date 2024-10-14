import pandas as pd

df = pd.read_csv('BANCO DE DADOS OD 2018 Março_2020.csv', sep=';')

result_rows = {}

for row in df.to_dict(orient="records"):
    if row['Zona Educacao'] in result_rows:
        result_rows[row['Zona Educacao']] = int(row['FREQUENCIA AULA']) + int(result_rows[row['Zona Educacao']])
    else:
        result_rows[row['Zona Educacao']] = int(row['FREQUENCIA AULA'])

    if row['Zona Trabalho'] in result_rows:
        result_rows[row['Zona Trabalho']] = int(row['FREQUENCIA TRABALHO']) + int(result_rows[row['Zona Trabalho']])
    else:
        result_rows[row['Zona Trabalho']] = int(row['FREQUENCIA TRABALHO'])
    if row['ORIGEM TRABALHO'] == 'RESIDENCIA':
        if row['Zona Residencia'] in result_rows:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA TRABALHO']) + int(result_rows[row['Zona Residencia']])
        else:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA TRABALHO'])

    if row['ORIGEM AULA'] == 'RESIDENCIA':
        if row['Zona Residencia'] in result_rows:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA AULA']) + int(result_rows[row['Zona Residencia']])
        else:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA AULA'])

data_list = [{'ZONA': key, 'FREQUENCIA': value} for key, value in result_rows.items()]


result_df = pd.DataFrame(data_list)

result_df.to_csv('resultado.csv', index=False)
