import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## АНАЛИТИКА ДАННЫХ

pd.pandas.set_option('display.max_columns',None)

df_train=pd.read_csv("...\\train.csv") ## считывание данных из файла

print(df_train.shape)

df_train.head()
##ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ
## Проверяем значения NaN, получаем список объектов с NaN
missing_values = [values
                  for values in df_train.columns
                  if df_train[values].isnull().sum() > 1]

df_train_missing = pd.DataFrame(np.round(df_train[missing_values].isnull().mean(), 4), columns=['% missing values'])

df_train_missing
## Конвертируем значения NaN в 1, остальные в 0, для упрощения построение зависимости от цены продажи
for value in missing_values:
    df_train_temp = df_train.copy()

    df_train_temp[value] = np.where(df_train_temp[value].isnull(), 1, 0)

##Средняя цена продажи, при которой информация отсутствует или присутствует в наличии
    df_train_temp.groupby(value)['SalePrice'].median().plot.bar()
    plt.title(value)
    plt.show()

##Настройка таблицы
df_train_html = df_train.head().copy()

def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])

styles = [
    hover(),
    dict(selector="th", props=[("font-size", "150%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]
html = (df_train_html.style.set_table_styles(styles)
          .set_caption("Hover to highlight."))
df_train_html = df_train.head().copy()

def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])

styles = [
    hover(),
    dict(selector="th", props=[("font-size", "150%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]
html = (df_train_html.style.set_table_styles(styles)
          .set_caption("Hover to highlight."))

##КОЛИЧЕСТВЕННЫЕ ПЕРЕМЕННЫЕ
##Получаем список количественных переменных
df_train.dtypes.unique()

number_value = [value
                for value in df_train.columns
                if df_train[value].dtypes != 'O'] ##тип данных "O" представляет объект, который включает категориальные переменные

print('Количество числовых значений: ', len(number_value))

df_train[number_value].head()
##Настройка таблицы
def color_negative_red(col):
    col1 = col
    color = 'green' if 'Yr' in col1 or 'Year'in col1 else 'black'
    return 'color: %s' % color

pd.DataFrame(df_train.columns).style.map(color_negative_red)
##Из набора данных у нас есть переменные на 4 года. Нам нужно извлечь информацию из переменных даты и времени, например, количество лет или дней. Одним из примеров в этом конкретном сценарии может быть разница в годах между годом постройки дома и годом его продажи. Мы проведем этот анализ в рамках разработки генерации признаков.
for_year_value = [value for value in number_value if 'Yr' in value or 'Year' in value]

for value in for_year_value:
    print(value, df_train[value].unique(), end ='\n\n', sep='\n')

df_train[for_year_value].describe()


##Мы проверим взаимосвязь между временные переменные даты и времени за год и ценой продажи и посмотрим, как цена меняется с течением времени
##Анализировать будем по медиане для надежности
for yr_value in for_year_value:
    df_train.groupby(yr_value)['SalePrice'].median().plot()
    plt.xlabel(yr_value)
    plt.ylabel('Медиана стоимости дома')
    plt.title("Стоимость дома против {}".format(yr_value))
    plt.show()
##Cравним разницу между значениями "За все года" и годом продажи по цене продажи
data=df_train.copy()
for value in for_year_value:
    if value!= 'YrSold':
        data[value]= data['YrSold'] - data[value]

        plt.scatter(data[value], data['SalePrice'])
        plt.xlabel(value)
        plt.ylabel('Цена продажи')
        plt.show()
##Вычленяем дискретные значения
discret_values=[value for value in number_value if len(data[value].unique()) < 25 and value not in for_year_value + ['Id']]
print("Общее количество дискретных значений: {}".format(len(discret_values)))

data[discret_values].head()

data[discret_values].describe()
## Найдем взаимосвязь между дискретными значениями и ценой продажи
for value in discret_values:
    data.groupby(value)['SalePrice'].median().plot.bar()
    plt.xlabel(value)
    plt.ylabel('Цена продажи')
    plt.title(value)
    plt.show()
##Вычленяем непрерывные значения
continue_value=[value for value in number_value if value not in discret_values + for_year_value + ['Id']]
print("Количество непрерывных значений {}".format(len(continue_value)))
##Для анализа построим гистограммы
for value in continue_value:
    data[value].hist(bins=25)
    plt.xlabel(value)
    plt.ylabel("Количество")
    plt.title(value)
    plt.show()

df_train[continue_value].head()
##Для лучшего анализа скошенные данные сделаем более симметричными с помощью логарифмического преобразования
data = df_train.copy()
for value in continue_value:
    if 0 in data[value].unique() or value in ['SalePrice']:
        pass
    else:
        data[value]=np.log(data[value])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[value], data['SalePrice'])
        plt.xlabel(value)
        plt.ylabel('Цена продажи')
        plt.title(value)
        plt.show()

data=df_train.copy()
for value in continue_value:
    if 0 in data[value].unique(): # пропускаем значения, которые выбиваются из общей массы
        pass
    else:
        data[value]=np.log(data[value])
        data.boxplot(column=value)
        plt.ylabel(value)
        plt.title(value)
        plt.show()
##Вычленяем категориальные значения
categorical_values = [value for value in df_train.columns if data[value].dtypes == 'O']

df_train[categorical_values].head()

cat_count = []
for value in categorical_values:
    cat_count.append(len(df_train[value].unique()))

data_cat = {'Значения':categorical_values, 'Без категории':cat_count}

data_cat = pd.DataFrame(data_cat)

data_cat
##Взаимосвязь между категориальной переменной и ценой продажи зависимого объекта
data=df_train.copy()
for value in categorical_values:
    data.groupby(value)['SalePrice'].median().plot.bar()
    plt.xlabel(value)
    plt.ylabel('Цена продажи')
    plt.title(value)
    plt.show()