import pandas as pd

# Leitura da planilha CSV
df = pd.read_csv('BANCO DE DADOS OD 2018 Março_2020.csv', sep=';')

# Lista para armazenar os resultados
result_rows = {}

# Iterando sobre as linhas da tabela original
for row in df.to_dict(orient="records"):
    # Para Zona Educacao
    if row['Zona Educacao'] in result_rows:
        result_rows[row['Zona Educacao']] = int(row['FREQUENCIA AULA']) + int(result_rows[row['Zona Educacao']])
    else:
        result_rows[row['Zona Educacao']] = int(row['FREQUENCIA AULA'])

    if row['Zona Trabalho'] in result_rows:
        result_rows[row['Zona Trabalho']] = int(row['FREQUENCIA TRABALHO']) + int(result_rows[row['Zona Trabalho']])
    else:
        result_rows[row['Zona Trabalho']] = int(row['FREQUENCIA TRABALHO'])
    # Para Zona Residencia com ORIGEM TRABALHO igual a RESIDENCIA
    if row['ORIGEM TRABALHO'] == 'RESIDENCIA':
        if row['Zona Residencia'] in result_rows:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA TRABALHO']) + int(result_rows[row['Zona Residencia']])
        else:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA TRABALHO'])

    # Para Zona Residencia com ORIGEM AULA igual a RESIDENCIA
    if row['ORIGEM AULA'] == 'RESIDENCIA':
        if row['Zona Residencia'] in result_rows:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA AULA']) + int(result_rows[row['Zona Residencia']])
        else:
            result_rows[row['Zona Residencia']] = int(row['FREQUENCIA AULA'])

data_list = [{'ZONA': key, 'FREQUENCIA': value} for key, value in result_rows.items()]


# Criar DataFrame a partir dos resultados
result_df = pd.DataFrame(data_list)

# Salvar o resultado em um novo CSV
result_df.to_csv('resultado.csv', index=False)