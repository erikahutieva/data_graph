import numpy as np
import matplotlib.pyplot as plt


data = []
with open("CarPrice_Assignment.csv", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    head = lines[0].split(',')
    for line in lines[1:]:
        if line.strip():
            values = line.split(',')
            row = dict(zip(head, values))
            data.append(row)


columns = ["horsepower", "enginesize",  "curbweight","price"]
data_special = []
for row in data:
    data_special.append([float(row[col]) for col in columns])
array = np.array(data_special)


weights = np.array([0.9, 7.4, 9.0, 0.1])
standard = array * weights


x = standard[:, 0]  
y = standard[:, 1]  
z = standard[:, 2] 

sizes = standard[:, 3]/10


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=sizes, c='blue', alpha=0.5)

ax.set_xlabel("Мощность")
ax.set_ylabel("Объём двигателя")
ax.set_zlabel("Масса")
ax.set_title("график (Цена - 4-я координата)")


fig, proect = plt.subplots(1, 3)

proect[0].scatter(x, y, s=sizes, c='blue', alpha=0.5)
proect[0].set_title("Мощность и Объём двигателя")

proect[0].set_xlim([50, 150])
proect[0].set_ylim([500, 1500])

proect[1].scatter(x, z, s=sizes, c='green', alpha=0.5)
proect[1].set_title("Мощность и Масса")

proect[1].set_xlim([50, 150])
proect[1].set_ylim([15000, 30000])

proect[2].scatter(y, z, s=sizes, c='red', alpha=0.5)
proect[2].set_title("Объём двигателя и Масса")
proect[2].set_xlim([500, 1500])
proect[1].set_ylim([17000, 33000])


plt.tight_layout()
plt.show()
