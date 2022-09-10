import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('titanic.csv')
print(df.head())
print(df.info())
print(df.describe())
#Gender Penumpang
sns.catplot(x='Sex', kind='count', data=df, orient='h')
plt.tight_layout()
plt.show()
#Umur Penumpang
df['Age'].hist(bins=70)
plt.title('Age Distribution:', size=14)
plt.xlabel('Age')
plt.ylabel('Passenger Count')
plt.tight_layout()
plt.show()
#Membuat new collumn
def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex
df['person'] = df[['Age','Sex']].apply(male_female_child,axis=1)
print(df)
#Umur dan kelas tiket
fig = sns.FacetGrid(df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
plt.title('Distribusi umur penumpang berdasarkan tiket', size=14)
plt.tight_layout()
plt.show()
#Gender dan kelas tiket
sns.catplot(x='Pclass',kind='count', hue='person', data=df)
plt.title('Distribusi tiket berdasarkan gender', size=14)
plt.tight_layout()
plt.show()
#Di kabin mana sajakah penumpang berada?
deck = df['Cabin'].dropna()
levels = []
for level in deck:
    levels.append(level[0])
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot(x='Cabin',kind='count',data=cabin_df,palette='summer',
            order=['A','B','C','D','E','F'])
plt.title('Distribusi penumpang setiap cabin', size=14)
plt.tight_layout()
plt.show()
#Asal Penumpang
sns.catplot(x='Embarked',kind='count',data=df,hue='Pclass', order=['C','Q','S'])
plt.title('Asal penumpang', size=14)
plt.tight_layout()
plt.show()
#Proporsi Penumpang yang sendiri dan bersama keluarga
df['Alone'] = df.SibSp + df.Parch
print(df['Alone'])
df['Alone'].loc[df['Alone'] > 0] = 'With Family'
df['Alone'].loc[df['Alone'] == 0] = 'Alone'
print(df.head())
sns.catplot(x='Alone',kind='count',data=df,palette='Blues')
df['Alone'].value_counts()
plt.title('Jumlah penumpang sendiri dan bersama keluarga', size=14)
plt.tight_layout()
plt.show()
#Aspek-apsek memengaruhi survived
df['Survivor'] = df.Survived.map({0:'no',1:'yes'})
sns.catplot(x='Survivor',kind='count',data=df,palette='Set1')
df['Survivor'].value_counts()
plt.title('Jumlah penumpang selamat', size=14)
plt.tight_layout()
plt.show()
generations = [10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Sex',data=df,palette='winter',x_bins=generations)
plt.title('Jumlah penumpang selamat berdasarkan umur', size=14)
plt.tight_layout()
plt.show()
sns.catplot(x='Pclass',y='Survived',kind='bar',data=df)
plt.title('Jumlah penumpang selamat berdasarkan kelas', size=14)
plt.tight_layout()
plt.show()
sns.catplot(x='Pclass',y='Survived',hue='person',kind='point',data=df)
plt.title('Gabungan antara kelas dan sex', size=14)
plt.tight_layout()
plt.show()
generations = [10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=df,palette='winter',x_bins=generations)
plt.title('Gabungan antara kelas dan umur', size=14)
plt.tight_layout()
plt.show()
g=sns.catplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',kind='point',data=df)
g.set(yscale="log")
plt.title('Pengaruh Embarked, Pclass dan Sex terhadap Survived', size=14)
plt.tight_layout()
plt.show()
df['Relatives'] = df.SibSp + df.Parch
df.head()
sns.catplot(x='Relatives',y='Survived',kind='point',data=df)
plt.title('Pengaruh SibSp dan Parch terhadap Survived', size=14)
plt.tight_layout()
plt.show()