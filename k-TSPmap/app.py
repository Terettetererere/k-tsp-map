import streamlit as st
import folium
import pandas as pd
import math

from mypulp import *
from streamlit_folium import st_folium

# 城の位置を可視化する
def visualize_locations(df,  zoom=5):
    f = folium.Figure(width=1200, height=1200)
    center_lat=34.686567
    center_lon=135.52000
    m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)
    for i in range(len(df)):
        folium.Marker(location=[df["lat"][i],df["lon"][i]]).add_to(m)
    return m




# 100地点をそのまま解くのは不可能なので、20のグループに分解してそれぞれまとめて回ることを想定する
# k-median
def kmedian(I, J, c, k):
    """kmedian -- minimize total cost of servicing
    customers from k facilities
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - c[i,j]: cost of servicing customer i from facility j
        - k: number of facilities to be used
    Returns a model, ready to be solved.
    """

    model = Model("k-median")
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in I:
        model.addConstr(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)
        for j in J:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.addConstr(quicksum(y[j] for j in J) == k, "Facilities")

    model.setObjective(quicksum(c[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

    model.update()
    model.__data = x, y
    return model


def distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def k_make_data(df):
    I = range(len(df))
    J = I
    x = [df["lat"][i] for i in I]
    y = [df["lon"][i] for i in I]
    c = {}
    for i in I:
        for j in J:
            c[i, j] = distance(x[i], y[i], x[j], y[j])

    return I, J, c, x, y


def make_data(selected_df):
    V = range(len(selected_df))
    x = [selected_df["lat"][i] for i in V]
    y = [selected_df["lon"][i] for i in V]
    c = {}
    for i in V:
        for j in V:
            c[i, j] = distance(x[i], y[i], x[j], y[j])
    return list(V), c, x, y

def tspdp(n, c, V):
    def f(j, S):
        FS = frozenset(S)
        if (j, FS) in memo:
            return memo[j, FS]
        elif FS == frozenset({j}):
            memo[j, FS] = c[V[0], j], V[0]
            return c[V[0], j], V[0]
        else:
            S0 = S.copy()
            S0.remove(j)
            min_value = 999999999999.
            prev = -1
            for i in S0:
                if f(i, S0)[0] + c[i, j] < min_value:
                    min_value = f(i, S0)[0] + c[i, j]
                    prev = i
            memo[j, FS] = min_value, prev
            return memo[j, FS]

    memo = {}
    n = len(V)
    opt_val, prev = f(V[0], set(V))

    j = V[0]
    S = set(V)
    tour = [j]
    while True:
        val, prev = memo[j, frozenset(S)]
        tour.append(prev)
        S = S - set({j})
        j = prev
        if j == V[0]:
            break
    tour.reverse()
    return opt_val, tour



############################################################

add_selectbox = st.sidebar.selectbox(
    "Select",
    ("名所巡り", "", "")
)

st.title("名所巡り")

df_title = ["ダム", "橋", "建築", "公園", "城"]
df_set = ["damu02.csv", "hashi02.csv", "kentiku02.csv", "koen02.csv", "shiro02.csv"]
df_selected = st.selectbox("データの選択", df_title)
df_i = df_title.index(df_selected)
df = pd.read_csv(df_set[df_i], encoding='shift_jis')
df.rename(columns={"北緯": "lat", "東経": "lon"}, inplace=True)
st.write(df.head())

# st.map(df[["lat", "lon"]])

option_k = st.slider("集約する地点数", 1, 20, 1)

checkbox = st.checkbox("計算")

if checkbox == True:

    m = visualize_locations(df)

    I, J, c, x_pos, y_pos = k_make_data(df)
    k = option_k
    model = kmedian(I, J, c, k)
    model.optimize()
    EPS = 1.0e-6
    x, y = model.__data
    edges = [(i, j) for (i, j) in x if x[i, j].X > EPS]
    selected = [j for j in y if y[j].X > EPS]
    # st.write("Optimal value=", model.ObjVal)
    # st.write("Selected facilities:", selected)

    sq1 = []
    for i in edges:
        sq1.append([[df["lat"][i[0]], df["lon"][i[0]]], [df["lat"][i[1]], df["lon"][i[1]]]])
    folium.PolyLine(locations=sq1).add_to(m)

    selected_df = df.loc[selected, :]
    selected_df.reset_index(inplace=True)
    # selected_df

    V, c, x, y = make_data(selected_df)
    opt_val, tour = tspdp(k, c, V)
    # print("Optimal value=", opt_val)
    # print("Optimal tour=", tour)

    sq2 = []
    for i in selected:
        sq2.append([df["lat"][i], df["lon"][i]])
    folium.PolyLine(locations=sq2).add_to(m)


    st_folium(m)