import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


df = pd.read_csv('https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv', delimiter = ',')
#df.head()

def preprocessing_data_frame(frame):
    df = frame.copy()
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
    
    question_dist = df[(df.Year <2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    # Анализ и очистка данных
    # анализ гистограмм
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    
    # анализ гистограмм
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    
    #анализ гистограмм
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)
    
    df = df.reset_index(drop=True)  # обновим индексы в датафрейме DF. если бы мы прописали drop = False, то была бы еще одна колонка - старые индексы
    # Разделение данных на признаки и целевую переменную
    
    
    # Предварительная обработка категориальных данных
    # Порядковое кодирование. Обучение, трансформация и упаковка в df
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns]);
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    return df


output_path = '/var/lib/jenkins/workspace/download/cars.csv'

preprocessing_data_frame(df).to_csv(output_path, index=False)
