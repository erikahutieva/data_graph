import csv
import plotly.graph_objects as go

path = "CarPrice_Assignment.csv"


with open(path, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]


for row in data:
    row['horsepower'] = float(row['horsepower']) if row['horsepower'].isdigit() else 0
    row['curbweight'] = float(row['curbweight']) if row['curbweight'].isdigit() else 0
    row['citympg'] = float(row['citympg']) if row['citympg'].isdigit() else 0
    row['price'] = float(row['price']) if row['price'].isdigit() else 0 

def get_color(price):
    if price < 10000:
        return 'red', 'Цены < 10000'
    elif price < 20000:
        return 'yellow', 'Цены 10000-19999'
    elif price < 30000:
        return 'green', 'Цены 20000-29999'
    else:
        return 'blue', 'Цены > 30000'


fig = go.Figure()


for color, label in [('red', 'Цены < 10000'), ('yellow', 'Цены 10000-19999'), ('green', 'Цены 20000-29999'), ('blue', 'Цены > 30000')]:
    fig.add_trace(go.Scatter3d(
        x=[row['horsepower'] for row in data if get_color(row['price'])[0] == color],
        y=[row['curbweight'] for row in data if get_color(row['price'])[0] == color],
        z=[row['citympg'] for row in data if get_color(row['price'])[0] == color],
        mode='markers',
        marker=dict(
            size=[row['price'] / 1000 if row['price'] else 5 for row in data if get_color(row['price'])[0] == color],
            color=color,
            opacity=0.8
        ),
        name=label
    ))

fig.update_layout(
    title="График для horsepower, curbweight, citympg и price",
    scene=dict(
        xaxis_title="Лошадиные силы (horsepower)",
        yaxis_title="Вес кузова (curbweight)",
        zaxis_title="Городской расход топлива (citympg)"
    ),
    showlegend=True  
)

fig.show()
