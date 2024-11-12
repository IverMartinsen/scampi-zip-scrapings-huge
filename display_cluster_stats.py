import pandas as pd

stats = pd.read_csv("./clusters/cluster_stats.csv", sep=";")
stats['clean'] = (stats['black'] != True) * (stats['blur'] != True) * (stats['artifact'] != True) * (stats['multi'] != True)


# black clusters
print("========================================")
num_black_clusters = stats[stats['black'] == True]['black'].sum()
num_black_images = stats[stats['black'] == True]['size'].sum()
print(f"Number of black clusters: {num_black_clusters}")
print(f"Number of black images: {num_black_images}")

# blurry clusters
print("========================================")
num_blurry_clusters = stats[stats['blur'] == True]['blur'].sum()
num_blurry_images = stats[stats['blur'] == True]['size'].sum()
print(f"Number of blurry clusters: {num_blurry_clusters}")
print(f"Number of blurry images: {num_blurry_images}")

# artifacts clusters
print("========================================")
num_artifacts_clusters = stats[stats['artifact'] == True]['artifact'].sum()
num_artifacts_images = stats[stats['artifact'] == True]['size'].sum()
print(f"Number of artifacts clusters: {num_artifacts_clusters}")
print(f"Number of artifacts images: {num_artifacts_images}")

# multi clusters
print("========================================")
num_multi_clusters = stats[stats['multi'] == True]['multi'].sum()
num_multi_images = stats[stats['multi'] == True]['size'].sum()
print(f"Number of multi-object clusters: {num_multi_clusters}")
print(f"Number of multi-object images: {num_multi_images}")

# other clusters
print("========================================")
num_other_clusters = stats[stats['other'] == True]['other'].sum()
num_other_images = stats[stats['other'] == True]['size'].sum()
print(f"Number of other clusters: {num_other_clusters}")
print(f"Number of other images: {num_other_images}")
print(f"These clusters are: {stats[stats['other'] == True].index.values}")

# clean clusters
print("========================================")
num_clean_clusters = stats[stats['clean'] == True]['clean'].sum()
num_clean_images = stats[stats['clean'] == True]['size'].sum()
print(f"Number of clean clusters: {num_clean_clusters}")
print(f"Number of clean images: {num_clean_images}")

# prime clusters
print("========================================")
num_prime_clusters = stats[stats['prime'] == True]['prime'].sum()
num_prime_images = stats[stats['prime'] == True]['size'].sum()
print(f"Number of prime clusters: {num_prime_clusters}")
print(f"Number of prime images: {num_prime_images}")