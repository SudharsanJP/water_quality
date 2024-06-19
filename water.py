import streamlit as st
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

st.title(':orange[💮water quality - ML project]🌞')

#) reading the dataset
df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\water quality\water_quality.csv")
#) checkbox with df
st.subheader("\n:green[1. dataset🌝]\n")
if (st.checkbox("original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")
    data = df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))

if (st.checkbox("null values")):
    #) to check the null values
    st.markdown("\n#### :violet[1.2 to check the null values:]")
    knowing_null = df.isna().sum()
    st.code(knowing_null)

#) forward fill and backward fill
#print(simple_colors.yellow("\nforward fill and backward fill:\n"))
df1 = df.ffill()
df1 = df.bfill()
#df1

#) to check the null values after ffill, bfill
#print(simple_colors.magenta("\nto check again for the null values after ffill, bfill:\n"))
#print(df1.isna().sum())

#) to replace the remaining null values
#print(simple_colors.yellow("\nto check again for the null values after ffill, bfill:\n"))
replace= {'Date':df['Date'].mode()[0],
          'DissolvedOxygen (mg/L)':df['DissolvedOxygen (mg/L)'].mean(),
         'pH':df['pH'].mean(),
         'SecchiDepth (m)':df['SecchiDepth (m)'].mean(),
         'WaterDepth (m)':df['WaterDepth (m)'].mean(),
          'WaterTemp (C)':df['WaterTemp (C)'].mean(),
          'AirTemp (C)':df['WaterTemp (C)'].mean()
         }
df2 = df1.fillna(value=replace)

#) to check the null values
if (st.checkbox("null values after fill")):
    st.markdown("\n#### :blue[1.3 null values after fill:]")
    knowing_null = df2.isna().sum()
    st.code(knowing_null)

#) converting date into year:
list_Date = df2['Date'].tolist() #) converting item_date column into list
Date_string = map(str, list_Date)
year_list = []
for date in list_Date:
    split_date = re.split("-",date)
    year = split_date[0]
    year = int(year)
    year_list.append(year)
#year_list

#) converting list into column
df2['year'] = year_list
#print(df2.head(5))

#) to check the type of data in each column
datatypes = df2.dtypes
#print(datatypes)

#)scatterplot
if (st.checkbox("scatterplot")):
    st.markdown("#### :green[1.4 scatterplot]")
    #)scatter plot
    fig = px.scatter(
    df2,
    x='pH',
    y='Salinity (ppt)',
    color="year",
    log_x=True,
    size_max=60,
)
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

#plt.show()

#) ML regression models
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df2.drop(['Date'],axis=1,inplace=True)

X = df2.drop(['Salinity (ppt)'],axis=1)
y = df2['Salinity (ppt)']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Ridge()
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

#) evaluation metrics
st.subheader("\n:green[2. linear regression🌹]")
selectBox=st.selectbox("ML models:", ['metrics',
                                      'test',
                                      'prediction'
                                       ])
if selectBox == 'metrics':
    st.markdown("#### :red[2.1 Evaluation Metrics]")
    st.text("**Train data**")
    st.write(mean_squared_error(y_train,train_pred))
    st.text("**Test data**")
    st.write(mean_squared_error(y_test,test_pred))

elif selectBox == 'test':
    st.markdown("#### :violet[2.2 actual testing vs testing prediction]")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    data =  test_df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:green'))
    

#) taking new inputs from user for the prediction
if selectBox == 'prediction':
     st.markdown("#### :blue[2.3 taking new inputs from user for the prediction]")
     dissolved_oxygen= st.number_input(":red[**Enter dissolved oxygen:**]")
     if (dissolved_oxygen):
        pH  = st.number_input(":red[**pH:**]")
        if (pH):
            sacchi_depth = st.number_input(":red[**sacchi depth:**]")
            if (sacchi_depth):
                water_depth = st.number_input(":red[**water depth:**]")
                if (water_depth): 
                    water_temp= st.number_input(":red[**water temp:**]")
                    if (water_temp):
                        air_temp= st.number_input(":red[**air temp:**]")
                        if (air_temp):
                            year= st.number_input(":red[**year:**]")
                            if (year):
                                new_input = [[ dissolved_oxygen,pH,sacchi_depth,water_depth,water_temp,air_temp,year]]
                                # get prediction for new input
                                new_output = model.predict(new_input)
                                st.write(":voilet[prediction:]")
                                st.success(new_output)
                            else:
                                st.error('you have not entered year')
                        else:
                            st.error('you have not entered air temp')
                    else:
                        st.error('you have not entered water temp')
                else:
                        st.error('you have not entered water depth')
            else:
                st.error('you have not entered sacchi depth')
        else:
            st.error('you have not entered pH')
     else:
         st.error('you have not entered dissolved oxygen')
    


                            
                    



