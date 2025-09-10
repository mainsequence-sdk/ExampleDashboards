import pandas as pd

df=pd.read_excel("/home/jose/Downloads/OneDrive_1_9-10-2025/VectorAnalitico24h_2024-09-03.xls")

fixed_income=df[df["CUPON ACTUAL"]!=0.0]
floating_tiie=df[df["SUBYACENTE"]=="TIIE28"]