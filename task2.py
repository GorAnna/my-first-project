import pandas as pd
df = pd.read_csv('100 Sales Records.csv', sep = ',')

#данные о таблице
df.info() 

#первые 5 записей
df.head()

#вывод осн. статистич. характеристик
print(df.describe())

# Проверка наличия пропущенных значений по каждому столбцу
print(df.isnull().sum())

#удаление строк с пропущенными значениями 
print(df.dropna(subset=['Sales Channel'])) 

#удаление ненужных колонок
cont_drop=['Region', 'Country', 'Item Type', 'Sales Channel','Order Priority','Order Date', 'Ship Date']
df.drop(cont_drop, axis=1, inplace=True)

# Преобразование категориальных признаков в числовые с использованием one-hot encoding
df_encoded = pd.get_dummies(df)

# Сохранение предобработанных данных в новый CSV-файл
df_encoded.to_csv('100_Sales_Rec_preprocessing.csv', index=False)