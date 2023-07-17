import numpy as np
import os
import librosa
import scipy.stats

from pprint import pprint


def normalize_feature(feature):
    normalized_feature = np.zeros(feature.shape)
    _, n = feature.shape

    for i in range(n):
        min_values = np.min(feature[:, i])
        max_values = np.max(feature[:, i])
        range_values = max_values - min_values

        if range_values == 0:
            normalized_feature[:, i] = 0
        else:
            normalized_feature[:, i] = (feature[:, i] - min_values) / range_values

    return normalized_feature


from tqdm import tqdm  # to give an progress bar need. pip install tqdm

# 2.1. Processar as features do ficheiro top100_features.csv.
def process_top100_features():
    # 2.1.1. Ler o ficheiro e criar um array numpy com as features disponibilizadas.


    features = np.genfromtxt("./Features - Audio MER/top100_features.csv", delimiter=',', skip_header=1, usecols=range(1, 100))

    # 2.1.2. Normalizar as features no intervalo [0, 1].
    normalized_features = normalize_feature(features)


    # 2.1.3. Criar e gravar em ficheiro um array numpy com as features extraídas (linhas = músicas; colunas = valores das features).
    np.savetxt("./processed_features/normalized_top100_features.csv", normalized_features, delimiter=",", fmt="%.6f")

    return normalized_features


# 2.2. Extrair features da framework librosa.
def extractFeatures(music_dir):
    def get_stats(lista: np.ndarray):
        """média, desvio padrão, assimetria (skewness), curtose (kurtosis), mediana, máximo e mínimo. Para o efeito, utilizar a biblioteca scipy.stats (e.g., scipy.stats.skew)."""

        return np.asarray([
            np.mean(lista),
            np.std(lista),
            scipy.stats.skew(lista, nan_policy="omit"),
            scipy.stats.kurtosis(lista, nan_policy="omit"),
            np.median(lista),
            np.max(lista),
            np.min(lista)
        ], dtype=np.float32)

    def load_feature(file_path):
        y, sr = librosa.load(file_path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        temp = np.zeros((13, 7))
        for i in range(13):
            temp[i] = get_stats(mfcc[i])

        mfcc = temp.flatten()

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        contrast = librosa.feature.spectral_contrast(y=y)
        temp = np.zeros((7, 7))
        for i in range(7):
            temp[i] = get_stats(contrast[i])
        contrast = temp.flatten()

        flatness = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        f0 = librosa.yin(y=y, sr=sr, fmin=20, fmax=11025)
        f0[f0 == 11025]=0
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

        resp = []
        resp.extend(mfcc)
        resp.extend(get_stats(centroid.flatten()))
        resp.extend(get_stats(bandwidth.flatten()))
        resp.extend(contrast)
        resp.extend(get_stats(flatness.flatten()))
        resp.extend(get_stats(rolloff.flatten()))
        resp.extend(get_stats(f0.flatten()))
        resp.extend(get_stats(rms.flatten()))
        resp.extend(get_stats(zcr.flatten()))
        resp.append(tempo[0])
        resp = np.asarray(resp, dtype=np.float32)

        return resp

    feature_list = np.zeros((900, 190))
    i = 0

    if not os.path.exists("./processed_features/all_features.csv"):  # if features no yet saved
        for file in tqdm(os.listdir(music_dir)):  # to give an progress bar need
        #for file in os.listdir(music_dir):
            if not file.endswith(".mp3"):
                continue
            #print(f"{(i / 900) * 100:.2f}")
            # Carrega o arquivo de audio
            file_path = os.path.join(music_dir, file)
            a = load_feature(file_path)
            feature_list[i] = a
            i += 1

        np.savetxt("./processed_features/all_features.csv", feature_list, delimiter=',', fmt="%.6f")
    else:
        feature_list = np.genfromtxt("./processed_features/all_features.csv", delimiter=',')


    feature_list = normalize_feature(feature_list)

    np.savetxt("./processed_features/normalized_all_features.csv", feature_list, delimiter=',', fmt="%.6f")

    return feature_list


def euclidean_distance(normalized_features):
    sim_matriz = np.zeros((900,900))


    for i, f1 in enumerate(normalized_features):
        for j, f2 in enumerate(normalized_features):
            value = np.sqrt(np.sum(np.power(f1-f2, 2)))
            if value == 0.0:
                value = -1.0
            sim_matriz[i,j] = value

    return sim_matriz

def man_distance(normalized_features):
    sim_matriz = np.zeros((900,900))


    for i, f1 in enumerate(normalized_features):
        for j, f2 in enumerate(normalized_features):
            value = np.sum(np.abs(f1-f2))
            if value == 0.0:
                value = -1.0
            sim_matriz[i,j] = value

    return sim_matriz

def cos_distance(normalized_features):
    sim_matriz = np.zeros((900,900))


    for i, f1 in enumerate(normalized_features):
        for j, f2 in enumerate(normalized_features):
            value = 1 - (np.dot(f1,f2)/(np.linalg.norm(f1)*np.linalg.norm(f2)))

            if i == j:
                value = -1.0

            sim_matriz[i,j] = value

    return sim_matriz


def top_100_sim_matrix(top_100_features_normalized):
    euc_matrix_top100 = euclidean_distance(top_100_features_normalized)
    man_matrix_top100 = man_distance(top_100_features_normalized)
    cos_matrix_top100 = cos_distance(top_100_features_normalized)

    np.savetxt("./processed_features/top_100_features_euc.csv", euc_matrix_top100, delimiter=',', fmt="%.6f")
    np.savetxt("./processed_features/top_100_features_man.csv", man_matrix_top100, delimiter=',', fmt="%.6f")
    np.savetxt("./processed_features/top_100_features_cos.csv", cos_matrix_top100, delimiter=',', fmt="%.6f")

def all_sim_matrix(features_normalized):
    euc_matrix = euclidean_distance(features_normalized)
    man_matrix = man_distance(features_normalized)
    cos_matrix = cos_distance(features_normalized)

    np.savetxt("./processed_features/features_euc.csv", euc_matrix, delimiter=',', fmt="%.6f")
    np.savetxt("./processed_features/features_man.csv", man_matrix, delimiter=',', fmt="%.6f")
    np.savetxt("./processed_features/features_cos.csv", cos_matrix, delimiter=',', fmt="%.6f")


def get_query_indices(metadata_csv):
    # Encontre os índices das músicas de consulta no arquivo de metadados
    query_song_names = ['"MT0000202045"', '"MT0000379144"', '"MT0000414517"', '"MT0000956340"']
    metadata = np.genfromtxt(metadata_csv, delimiter=',', dtype='str', skip_header=1, usecols=0)
    query_indices = []
    for song_name in query_song_names:
        index_tuple = np.where(metadata == song_name)
        index = index_tuple[0][0]  # Access the first element of the tuple, then the first element of the numpy array
        query_indices.append(index)

    return query_indices

def name_from_index(index):
    metadata = np.genfromtxt('./MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv', delimiter=',', dtype='str', skip_header=1, usecols=0)

    return metadata[index]



def get_top_20_recommendations_ascending(query_index, distance_matrix):
    # Find the 20 recommended songs based on the distance matrix
    distances = distance_matrix[query_index]

    # Find the indices of the smallest 21 distances (including the query itself)
    top_21_indices = np.argsort(distances)[:21]

    # Exclude the query(not excluded for debugging purposes)
    #top_20_indices = top_21_indices[top_21_indices != query_index]

    # Create a list of index-value pairs
    index_value_pairs = np.zeros((21,2)).astype(object)
    for i, index in enumerate(top_21_indices):
        index_value_pairs[i][0] = name_from_index(index)
        index_value_pairs[i][1] = distances[index]

    return index_value_pairs

def get_top_20_recommendations_descending(query_index, distance_matrix):
    # Find the 20 recommended songs based on the distance matrix
    distances = distance_matrix[query_index]

    # Find the indices of the smallest 21 distances (including the query itself)
    top_21_indices = np.argsort(distances)[::-1][:21]

    # Exclude the query(not excluded for debugging purposes)
    #top_20_indices = top_21_indices[top_21_indices != query_index]

    # Create a list of index-value pairs
    index_value_pairs = np.zeros((21,2)).astype(object)
    for i, index in enumerate(top_21_indices):
        index_value_pairs[i][0] = name_from_index(index)
        index_value_pairs[i][1] = distances[index]

    return index_value_pairs




def sim_top_20_ranking():
    metadata_csv = './MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    query_indices = get_query_indices(metadata_csv)

    distance_matrices = [
        'top_100_features_euc.csv',
        'top_100_features_man.csv',
        'top_100_features_cos.csv',
        'features_euc.csv',
        'features_man.csv',
        'features_cos.csv',
    ]

    for query_index in query_indices:
        for distance_file in distance_matrices:
            distance_matrix = np.genfromtxt(f'./processed_features/{distance_file}', delimiter=',')

            top_20_recommendations = get_top_20_recommendations_ascending(query_index, distance_matrix)

            query_name = name_from_index(query_index)
            query_name = query_name.replace('"', '')
            recommendations_file = f'./recommendations/distance_sim/{query_name}_top_20_{distance_file}'
            np.savetxt(recommendations_file, np.array(top_20_recommendations), delimiter=',', fmt='%s')


def meta_matrixes():
    metas = np.genfromtxt('./MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv', delimiter=",", dtype=str,
                          skip_header=1)
    sim_matriz = np.zeros((900, 900))

    for line, i in enumerate(metas):
        i_moods = i[9][1:-1].split("; ")
        i_gen = i[11][1:-1].split("; ")
        for col, j in enumerate(metas):
            # print(i)
            # print(j)
            # input()
            value = (i[1] == j[1]) + (i[3] == j[3])

            j_moods = j[9][1:-1].split("; ")
            for k in i_moods:
                value += k in j_moods

            j_gen = j[11][1:-1].split("; ")
            for k in i_gen:
                value += k in j_gen

            sim_matriz[line, col] = value

    np.savetxt("./recommendations/meta_sim/metadataSimilarity.csv", sim_matriz, delimiter=',', fmt='%d')


def meta_top_20_ranking():
    metadata_csv = './MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    query_indices = get_query_indices(metadata_csv)

    for query_index in query_indices:
            meta_matrix = np.genfromtxt(f'./recommendations/meta_sim/metadataSimilarity.csv', delimiter=',')

            top_20_recommendations = get_top_20_recommendations_descending(query_index, meta_matrix)

            query_name = name_from_index(query_index)
            query_name = query_name.replace('"', '')
            recommendations_file = f'./recommendations/meta_sim/{query_name}_top_20.csv'
            np.savetxt(recommendations_file, np.array(top_20_recommendations), delimiter=',', fmt='%s')

def calc_precision():
    def save_file(name, query, tipo, q1, q2, q2_2: np.ndarray):
        titl = name[:name.find('_')]
        with open(os.path.join("./recommendations/comp_sim", f"{name[:-4]}_{query}_{tipo}.txt"), "w+") as file:
            file.write(f"Query = '{titl}.mp3'\n\n")
            file.write(f"Ranking: {query}, {tipo}-------------\n")
            file.write(q1.__str__().replace("\"", "") + "\n\n")

            file.write(f"Ranking: Metadata-------------\n")
            file.write(q2.__str__().replace("\"", "") + "\n\n")
            file.write("Score metadata = " + q2_2.astype(float).__str__())
            file.write("\n\n")

    def update_file(path, precisions):
        with open(path, "a") as file:
            y = lambda x: file.write(f"Precision der:  {np.asarray(x, dtype=float).__str__()} *** {sum(x)/4} \n")
            y(precisions["e"])
            y(precisions["m"])
            y(precisions["c"])

    meta_tops = sorted(os.listdir("./recommendations/meta_sim"))[:-1]
    # meta_tops = [i[:i.find("_")] for i in meta_tops]

    distanses = {}
    for i in os.listdir("./recommendations/distance_sim"):
        temp  = i[:i.find("_")]
        if temp not in distanses:
            distanses[temp] = []
        
        distanses[i[:i.find("_")]].append(i)

    precisions = {
        'e': [0, 0, 0, 0],
        'm': [0, 0, 0, 0],
        'c': [0, 0, 0, 0],
    }

    for k, i in enumerate(meta_tops):
        temp = os.path.join("./recommendations/meta_sim", i)
        a = np.genfromtxt(temp, delimiter=",", dtype=str)
        # print(i)
        # print(a[:,0])
        for j in sorted(distanses[i[:i.find("_")]]):
            temp = os.path.join("./recommendations/distance_sim", j)
            b = np.genfromtxt(temp, delimiter=",", dtype=str)
            
            # precisions[j.split("_")[-1][0]][k] += len(set(a[:,0]).intersection(set(b[:,0])))-1
            # print(len(set(a[:,0]).intersection(set(b[:,0])))-1)
            
            q = "FMrosa"
            if "top_100" in j:
                q =  "top_100"
            else:
                precisions[j.split("_")[-1][0]][k] += len(set(a[:,0]).intersection(set(b[:,0])))-1

            
            save_file(i[:i.find("_")]+".mp3", q, j.split("_")[-1][:-4], b[:,0], a[:,0], a[:,1])

    print(*map(lambda x: x, precisions["e"]))
    print(*map(lambda x: x, precisions["m"]))
    print(*map(lambda x: x, precisions["c"]))

    for i in precisions:
        precisions[i] = [j*100/20 for j in precisions[i]]

    for i in os.listdir("./recommendations/comp_sim"):
        update_file(os.path.join("./recommendations/comp_sim", i), precisions)


def main():
    #top_100_features_normalized = process_top100_features()
    #top_100_sim_matrix(top_100_features_normalized)


    # features_normalized = extractFeatures("./MER_audio_taffc_dataset/musicas")
    # all_sim_matrix(features_normalized)

    # sim_top_20_ranking()

    # meta_matrixes()
    # meta_top_20_ranking()
    calc_precision()

if __name__ == "__main__":
    main()
