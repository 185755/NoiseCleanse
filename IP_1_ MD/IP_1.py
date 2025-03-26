import numpy as np
import scipy.io.wavfile as wav

# Wczytanie pliku WAV
def read_wav(file_path):
    sample_rate, data = wav.read(file_path)
    return sample_rate, data.astype(np.float64)

# Zapis pliku WAV
def save_wav(file_path, sample_rate, data):
    wav.write(file_path, sample_rate, data.astype(np.int16))

# Implementacja algorytmu EW-LS
def ew_ls(data, order, lambda_, delta, window_size, threshold_factor):
    N = len(data)
    a = np.zeros(order)  # Współczynniki modelu AR
    P = np.eye(order) * delta  # Macierz kowariancji
    errors = np.zeros(N)
    std_dev = np.zeros(N)
    y_clean = np.copy(data)
    consecutive_noises = 0  # Licznik zakłóconych próbek z rzędu
    
    # Algorytm EW-LS i detektor zakłóceń
    for k in range(order, N):
        y = data[k]
        y_prev = data[k-order:k][::-1]  # Poprzednie próbki (w odwrotnej kolejności)
        
        # Predykcja na podstawie modelu AR
        y_pred = np.dot(a, y_prev)
        e = y - y_pred  # Błąd predykcji
        
        # Aktualizacja parametrów modelu AR za pomocą EW-LS
        K = np.dot(P, y_prev) / (lambda_ + np.dot(y_prev, np.dot(P, y_prev)))
        a = a + K * e
        P = (P - np.outer(K, np.dot(y_prev, P))) / lambda_
        
        # Detektor zakłóceń
        errors[k] = e
        if k >= window_size and len(errors[k-window_size:k]) > 1:
            std_dev[k] = np.std(errors[k-window_size:k])
        else:
            std_dev[k] = np.std(errors[:k]) if len(errors[:k]) > 1 else 0
        
        # Jeśli std_dev[k] > 0, sprawdzamy próg
        if std_dev[k] > 0 and abs(e) > threshold_factor * std_dev[k]:  # Sprawdzanie próbek przekraczających próg
            consecutive_noises += 1
            if consecutive_noises <= 3:  # Zakłócenie dla maks. 3 kolejnych próbek
                y_clean[k] = np.dot(a, y_clean[k-order:k][::-1])
            else:
                # Czwartą próbkę traktujemy jako poprawną
                consecutive_noises = 0
        else:
            consecutive_noises = 0  # Reset licznika, jeśli próbka nie jest zakłócona

    return y_clean, errors

# Funkcja do znajdowania odpowiedniego progu
def find_optimal_threshold(data, order, lambda_, delta, window_size):
    threshold_factor = 0.1  # Zaczynamy od bardzo niskiej wartości
    while threshold_factor < 10:  # Zakładamy maksymalną wartość 10
        clean_data, errors = ew_ls(data, order, lambda_, delta, window_size, threshold_factor)
        consecutive_noises = 0
        max_consecutive_noises = 0
        
        # Sprawdzanie liczby zakłóconych próbek z rzędu na całej długości utworu
        for k in range(len(data)):
            y = data[k]
            e = y - clean_data[k]
            if k >= window_size and len(errors[k-window_size:k]) > 1:
                std_dev = np.std(errors[k-window_size:k])
            else:
                std_dev = np.std(errors[:k]) if len(errors[:k]) > 1 else 0
            if std_dev > 0 and abs(e) > threshold_factor * std_dev:  # Sprawdzanie próbek przekraczających próg
                consecutive_noises += 1
                max_consecutive_noises = max(max_consecutive_noises, consecutive_noises)
            else:
                consecutive_noises = 0
        
        # Jeśli maksymalnie 3 zakłócone próbki z rzędu, zwracamy próg
        if max_consecutive_noises >= 4:
            print(f"Odpowiedni próg detekcji: {threshold_factor}")
            return threshold_factor
        
        threshold_factor += 0.1  # Podwyższanie wartości progu
    
    print("Nie znaleziono odpowiedniego progu.")
    return None

# Main function
def remove_impulse_noise(input_file_path, output_file_path):
    sample_rate, data = read_wav(input_file_path)
    order = 4
    lambda_ = 0.95
    delta = 2000
    window_size = 100
    
    threshold_factor = find_optimal_threshold(data, order, lambda_, delta, window_size)
    if threshold_factor is not None:
        clean_data, _ = ew_ls(data, order, lambda_, delta, window_size, threshold_factor)
        save_wav(output_file_path, sample_rate, clean_data)
        print(f"Zapisano odszumiony plik audio jako {output_file_path}")

# Usage example
input_file_path = '06.wav'
output_file_path = '06_odszumione.wav'
print(f"Rozpoczynam proces odszumiania pliku {input_file_path}")
remove_impulse_noise(input_file_path, output_file_path)
