%% Inicjalizacja i wczytanie pliku
clear all; close all; clc;

[data, Fs] = audioread('06.wav');
data = data+(10^-6)*randn(size(data));
N = length(data);
%% Główne parametry
% Inicjalizacja parametrów EWLS i modelu AR
r = 4;  % Rząd modelu AR
lambda = 0.95; % Czynnik zapominania
P = 10000 * eye(r); % Macierz kowariancji
theta = zeros(r, 1); % Współczynniki AR

% Parametry do wygładzania współczynników
max_theta_change = 0.05; % Maksymalna zmiana współczynnika między iteracjami

% Inicjalizacja zmiennych przechowujących wartości estymowane i błędy
theta_s = zeros(r, N);
e_residuals = zeros(1, N);

% Parametry detekcji błędów
std_window = 80; % Długość okna do obliczania lokalnego odchylenia std
threshold_factor = 3; % Maksymalne odchylenie próbki
max_consecutive_errors = 4; % Maksymalna liczba błędnych próbek z rzędu
startup_samples = 100; % Liczba próbek na ustabilizowanie estymacji

% EWLS i detekcja błędów
clean_signal = data; % Kopia sygnału do oczyszczenia
consecutive_error_count = 0; % Licznik kolejnych błędów
%% Główna pętla i zapis
for t = r+1:N
    % Tworzenie wektora regresji (przeszłe wartości sygnału)
    phi = data(t-1:-1:t-r);

    % Błąd predykcji
    e = data(t) - ((phi') * theta);
    e_residuals(t) = e;

    % Aktualizacja współczynników AR za pomocą EWLS
    k = (P * phi) / (lambda + ((phi') * P * phi));
    theta_new = theta + k * e;
    
    % Ograniczenie nagłych zmian współczynników AR
    theta = theta + max(min(theta_new - theta, max_theta_change), -max_theta_change);

    % Aktualizacja macierzy kowariancji
    P = (P - (k * (phi') * P)) / lambda;

    % Zapisujemy współczynniki do analizy
    theta_s(:, t) = theta;

    % **Detekcja zakłóceń dopiero po 100 próbkach**
    if t > startup_samples
        if t > std_window
            local_std = std(e_residuals(t-std_window:t-1)); % Lokalne odchylenie std błędu
            
            if abs(e) > threshold_factor * local_std
                consecutive_error_count = consecutive_error_count + 1;
                
                % Jeśli liczba błędnych próbek przekracza limit, pomijamy zastępowanie
                if consecutive_error_count <= max_consecutive_errors
                    clean_signal(t) = clean_signal(t-1); % Zastępujemy poprzednią wartością
                end
            else
                % Reset licznika jeśli próbka jest poprawna
                consecutive_error_count = 0;
                
                % Jeśli poprzednia próbka była błędna, zastosuj interpolację
                if t > 1 && e_residuals(t-1) > threshold_factor * local_std
                    clean_signal(t) = 0.5 * (clean_signal(t-1) + clean_signal(t+1)); % Interpolacja
                end
            end
        end
    end
end

% Zapis odszumionego sygnału
audiowrite('odszumione.wav', clean_signal, Fs);
%% Wyświetlanie parametrów i wykresów
% Wyświetlenie parametrów
fprintf('=== Parametry algorytmu ===\n');  
fprintf('Częstotliwość próbkowania: %d Hz\n', Fs);  
fprintf('Liczba próbek: %d\n', N);  
fprintf('Rząd modelu AR: %d\n', r);  
fprintf('Czynnik zapominania (lambda): %.3f\n', lambda);  
fprintf('Długość okna do obliczania odchylenia std: %d\n', std_window);  
fprintf('Próg detekcji zakłóceń: %.1f * lokalne odchylenie std\n', threshold_factor);  
fprintf('Maksymalna liczba kolejnych zakłóconych próbek: %d\n', max_consecutive_errors);
fprintf('Początkowe próbki bez detekcji: %d\n', startup_samples);
fprintf('Plik wynikowy zapisany jako: odszumione.wav\n');  

%Wartości średnie parametrów
a1_mean = mean(theta_s(1,:), "all");
fprintf('Wartość średnia a1 = %d\n', a1_mean);

a2_mean = mean(theta_s(2,:), "all");
fprintf('Wartość średnia a2 = %d\n', a2_mean);

a3_mean = mean(theta_s(3,:), "all");
fprintf('Wartość średnia a3 = %d\n', a3_mean);

a4_mean = mean(theta_s(4,:), "all");
fprintf('Wartość średnia a4 = %d\n', a4_mean);

% Wizualizacja wyników  
figure;  
subplot(3,1,1);  
plot(data);  
title('Sygnał oryginalny');  

subplot(3,1,2);  
plot(e_residuals);  
title('Błąd predykcji');  

subplot(3,1,3);  
plot(clean_signal);  
title('Sygnał oczyszczony');  

figure;
subplot(4,1,1);
plot(theta_s(1,:));
ylim([-3, 3]);
title('a1');

subplot(4,1,2);
plot(theta_s(2,:));
ylim([-3, 3]);
title('a2');

subplot(4,1,3);
plot(theta_s(3,:));
ylim([-3, 3]);
title('a3');

subplot(4,1,4);
plot(theta_s(4,:));
title('a4');

% Dodatkowy wykres - Sygnał oryginalny z dynamicznymi progami na górze i na dole
figure;
plot(data, 'b'); % Sygnał oryginalny na niebiesko
hold on;

% Obliczanie dynamicznego progu dla każdej próbki (w oparciu o błąd predykcji)
threshold_upper = zeros(1, N); % Górny próg
threshold_lower = zeros(1, N); % Dolny próg

for t = r+1:N
    if t > std_window
        % Obliczanie lokalnego odchylenia std dla ostatnich std_window próbek błędów
        local_std = std(e_residuals(t-std_window:t-1)); 
        % Dynamiczny próg na podstawie błędu predykcji
        threshold_upper(t) = threshold_factor * local_std;
        threshold_lower(t) = -threshold_factor * local_std; % Dolny próg (ujemny)
    end
end

% Rysowanie linii progu - górny próg (czerwona linia)
plot(threshold_upper, 'r--', 'LineWidth', 1.5); 
% Rysowanie linii progu - dolny próg (czerwona linia)
plot(threshold_lower, 'r--', 'LineWidth', 1.5); 

title('Sygnał oryginalny z dynamicznymi progami');
legend('Sygnał oryginalny', 'Górny próg', 'Dolny próg');
xlabel('Próbka');
ylabel('Amplituda');

