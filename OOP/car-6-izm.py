import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
#from sklearn.cluster import KMeans  # не используется, мы пишем вручную

data = []
with open("CarPrice_Assignment.csv", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    head = lines[0].split(',')
    for line in lines[1:]:
        if line.strip():
            values = line.split(',')
            row = dict(zip(head, values))
            data.append(row)
'''for row in data[:5]:
    print(row)'''

columns = ["horsepower", "curbweight", "citympg", "highwaympg", "price", "enginesize"]
data_special = []
for row in data:
    data_special.append([float(row[col]) for col in columns])
array = np.array(data_special)
weights = np.array([0.05, 0.2, 0.1, 0.1, 0.5, 0.05])
data_weight = array * weights

# стандартизируем 
means = np.mean(data_weight, axis=0)
stds = np.std(data_weight, axis=0)
standart = (data_weight - means) / stds

# Реализация KMeans вручную (без использования библиотеки)
def manual_kmeans(X, k=4, max_iter=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    # Инициализируем центроиды, выбирая k случайных точек из X
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]
    for iteration in range(max_iter):
        # Вычисляем евклидовы расстояния от каждой точки до каждого центроида
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Определяем для каждой точки ближайший центроид
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            # Если кластер j не пуст, вычисляем его новый центроид как среднее точек
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        # Если центроиды практически не изменились, завершаем алгоритм
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return labels, centroids

# Используем ручную реализацию KMeans с 4 кластерами
clusters, centroids_manual = manual_kmeans(standart, k=4, max_iter=100, tol=1e-4, random_state=42)

cov_matrix = np.cov(standart, rowvar=False)

# собственные значения и собственные векторы 
values, vectors = np.linalg.eigh(cov_matrix)

#  Сортируем собственные значения (и соответствующие собственные векторы) по убыванию.
sorted_indices = np.argsort(values)[::-1]  # индексы в порядке убывания
sorted_values = values[sorted_indices]
sorted_vectors = vectors[:, sorted_indices]

# 4. Выбираем первые 4 собственных вектора как компоненты.
n_components = 4
components = sorted_vectors[:, :n_components]

# 5. Проецируем исходные стандартизированные данные на выбранные компоненты.
#    Это дает нам новые координаты в пространстве главных компонент.
pca_result_manual = np.dot(standart, components)

'''
print("Результат", pca_result_manual.shape)
print("Первые 5 строк :\n", pca_result_manual[:5])
'''

# Дополнительный анализ кластеров (вывод средних значений исходных признаков)
print("\nДополнительный анализ кластеров (средние значения исходных признаков):")
for cluster in np.unique(clusters):
    indices = np.where(clusters == cluster)[0]
    cluster_mean = np.mean(array[indices], axis=0)
    print(f"\nКластер {cluster}:")
    for col, mean_val in zip(columns, cluster_mean):
        print(f"  {col}: {mean_val:.2f}")

# Создание одного 3D-графика для визуализации 4D-пространства:
# По осям: PC1, PC2, PC3; размер маркера отражает значение PC4.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Для нормализации PC4 для размеров маркеров, берём PC4 из ручного PCA (pca_result_manual)
pc4_all = pca_result_manual[:, 3]
min_pc4, max_pc4 = pc4_all.min(), pc4_all.max()

# Выбираем цвета для кластеров: добавляем цвета для 4 кластеров
cluster_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}

# Для каждой группы (кластера) строим точки
for cl in np.unique(clusters):
    idx = np.where(clusters == cl)[0]
    # Координаты по первым трем компонентам
    x = pca_result_manual[idx, 0]
    y = pca_result_manual[idx, 1]
    z = pca_result_manual[idx, 2]
    # Четвёртая компонента будет отображаться через размер маркера.
    pc4 = pca_result_manual[idx, 3]
    sizes = np.interp(pc4, (min_pc4, max_pc4), (20, 100))  # размер от 20 до 100
    ax.scatter(x, y, z, s=sizes, c=cluster_colors[cl], label=f'Кластер {cl+1}', alpha=0.7)

ax.set_xlabel("v1 вектор")
ax.set_ylabel("v2 вектор")
ax.set_zlabel("v3 вектор")
plt.title("График")
plt.legend()
plt.show()
