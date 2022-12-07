import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.header("Mini project")

uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xlsx'])

df = pd.read_csv(uploaded_file)

st.subheader('Student Performance Dataset')
st.dataframe(df)

gender = {
    'male': 1,
    'female': 0
}

level = {
    "bachelor's degree": 0,
    'some college': 1,
    "master's degree": 2,
    "associate's degree": 3,
    "high school": 4,
    "some high school": 5
}

race = {
    'group A': 0,
    'group B': 1,
    'group C': 2,
    'group D': 3,
    'group E': 4
}

df['gender'] = df['gender'].map(gender)
df['race/ethnicity'] = df['race/ethnicity'].map(race)
df['parental level of education'] = df['parental level of education'].map(
    level)

st.subheader("Updated Dataset")
st.dataframe(df)


def students_taking_classes():
    st.subheader("How many student taking test preperation course : ")
    count_test = df['test preparation course'].value_counts()
    labels = df['test preparation course'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)


students_taking_classes()

df['average_score'] = (
    df['math score']+df['reading score']+df['writing score'])/3

st.dataframe(df)


def student_performance_in_subject_gander_based():
    st.subheader("Student performance in subject based on gender :")
    st.write("1 : Male")
    st.write("0 : Female")
    st.write("Average score Vs Math score")
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['average_score'],
                    y=df['math score'], hue=df['gender'])
    st.pyplot(fig)

    st.write("Average score Vs Reading score")
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['average_score'],
                    y=df['reading score'], hue=df['gender'])
    st.pyplot(fig)

    st.write("Average score Vs Writeing score")
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['average_score'],
                    y=df['writing score'], hue=df['gender'])
    st.pyplot(fig)


student_performance_in_subject_gander_based()
df = pd.get_dummies(df, drop_first=True)
x = df.drop(columns='average_score').values
y = df['average_score'].values
# print(x)
# print(x[0])
# print(y)

st.subheader("Building Regression model to predict average score of Student")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

model = RandomForestRegressor()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(predictions)

st.write("Training Dataset :")
st.write(x_train)


st.write("Testing Dataset :")
st.write(x_test)

st.write("Prediction of Average score of test dataset :")

# x_test['Predicted Average score'] = predictions
st.write(predictions)
# Calculating accuracy of model
st.write("---------------------------------------------------------------------")
st.subheader("Acurracy of model : " + str(r2_score(predictions, y_test)))
# print(r2_score(predictions, y_test))


# st.write(x_test)
# t = [0, 1, 0, 72, 72, 74, 0, 1]
# tmp = model.predict(t)
# st.write(tmp)

# female, group B , bachelor, standard, none, 72, 72, 74,

def predict(Gender, Race, Parental_Level_of_Education, Math_score, Reading_score, Writing_score, Lunch, test_preparation_course):

    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    if(Race == 'group A'):
        Race = 0
    elif(Race == "group B"):
        Race = 1
    elif(Race == "group C"):
        Race = 2
    elif(Race == "group D"):
        Race = 3
    else:
        Race = 4

    if Parental_Level_of_Education == "Some High school":
        Parental_Level_of_Education = 5
    elif Parental_Level_of_Education == "High school":
        Parental_Level_of_Education = 4
    elif Parental_Level_of_Education == "Associate's Degree":
        Parental_Level_of_Education = 3
    elif Parental_Level_of_Education == "Master's Degree":
        Parental_Level_of_Education = 2
    elif Parental_Level_of_Education == "Some college":
        Parental_Level_of_Education = 1
    else:
        Parental_Level_of_Education = 0

    if(Lunch == "Standard"):
        Lunch = 1
    else:
        Lunch = 0

    if test_preparation_course == "None":
        test_preparation_course = 1
    else:
        test_preparation_course = 0

    math_score = int(Math_score)
    reading_score = int(Reading_score)
    writing_score = int(Writing_score)
    # lst = [Gender, Race, Parental_Level_of_Education,
    #        math_score, reading_score, writing_score, Lunch, test_preparation_course]
    # st.write(lst)

    prediction = model.predict(([[Gender, Race, Parental_Level_of_Education,
                               math_score, reading_score, writing_score, Lunch, test_preparation_course]]))
    print(prediction)
    # prediction = np.round(prediction[0], 2)
    return prediction
    #     return render_template('index.html', Average=prediction)
    # else:
    #     return render_template('index.html')


st.write("---------------------------------------------------------------------")
st.subheader("Select attributed to predict average score of student")

gender = ""
race = ""
reading_score = 0
writing_score = 0
math_score = 0
lunch = ""
preperation_course = ""
parental_education = ""

col1, col2 = st.columns(2)

with col1:
    gender = st.radio(
        "Select Gender",
        key="visibility",
        options=["Male", "Female"],
    )


with col2:
    race = st.radio(
        "Select RACE/ETHNICITY",
        key="visibility1",
        options=["group A", "group B", "group C", "group D", "group E"],
    )


col1, col2, col3 = st.columns(3)

with col1:
    math_score = st.text_input(
        "Enter Math Score",
        key="placeholder1",
    )
with col2:
    reading_score = st.text_input(
        "Enter Reading Score",
        key="placeholder2",
    )
with col3:
    writing_score = st.text_input(
        "Enter Writing Score",
        key="placeholder3",
    )

col1, col2, col3 = st.columns(3)

with col1:
    parental_education = st.radio(
        "Parental Level of Education",
        key="v1",
        options=["Some High school", "High school", "Associate's Degree",
                 "Master's Degree", "Some college", "Bachelor's Degree"],
    )

with col2:
    lunch = st.radio(
        "Lunch",
        key="v2",
        options=["Standard", "Free/Reduced"],
    )

with col3:
    preperation_course = st.radio(
        "TEST PREPARATION COURSE",
        key="v3",
        options=["None", "Completed"],
    )


st.subheader("predicted Average : ")
# lst = [gender, race, parental_education, math_score,
#        reading_score, writing_score, lunch, preperation_course]
# st.write(lst)
st.write(predict(gender, race, parental_education, math_score,
         reading_score, writing_score, lunch, preperation_course))
