% Carregar dados de treinamento (Xt.mat, Yt.mat)
load('Xt.mat');  % Carrega Xt para a variável Xt
load('Yt.mat');  % Carrega Yt para a variável Yt


% Número de amostras e características nos dados de treinamento
[num_features_train, num_samples_train] = size(Xt);

% Número de classes nos dados de treinamento (assumindo 2 classes)
num_classes_train = size(Yt, 1);

% Transpor Xt se necessário (se as amostras estiverem nas colunas)
if num_features_train < num_samples_train
    Xt = Xt';
end

% Número de clusters desejado para os dados de treinamento
k_train = 2;

% Executar o k-means nos dados de treinamento
[idx_train, C_train] = kmeans(Xt, k_train);

% Converter índices de cluster em rótulos de cluster (1 a k_train)
cluster_labels_train = idx_train';

% Obter rótulos verdadeiros para os dados de treinamento
true_labels_train = (Yt(1, :) == 1) + 1;  % Transforma os rótulos em classe binária (1 ou 2)

% Calcular Acurácia para os dados de treinamento
accuracy_train = sum(true_labels_train == cluster_labels_train) / numel(true_labels_train);

% Calcular Matriz de Confusão para Precisão, Recall e F1-score
confusion_matrix_train = confusionmat(true_labels_train, cluster_labels_train);

% Calcular Precisão, Recall e F1-score para os dados de treinamento
precision_train = confusion_matrix_train(2,2) / (confusion_matrix_train(2,2) + confusion_matrix_train(1,2));
recall_train = confusion_matrix_train(2,2) / (confusion_matrix_train(2,2) + confusion_matrix_train(2,1));
f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);

% Exibir métricas de avaliação para os dados de treinamento
disp('Métricas de Avaliação para Dados de Treinamento:');
disp(['Acurácia: ', num2str(accuracy_train)]);
disp(['Precisão: ', num2str(precision_train)]);
disp(['Recall: ', num2str(recall_train)]);
disp(['F1-score: ', num2str(f1_score_train)]);

% Carregar dados de validação (Xv.mat, Yv.mat)
load('Xv.mat');  % Carrega Xv para a variável Xv
load('Yv.mat');  % Carrega Yv para a variável Yv

% Número de amostras e características nos dados de validação
[num_features_val, num_samples_val] = size(Xv);

% Número de classes nos dados de validação (assumindo 2 classes)
num_classes_val = size(Yv, 1);

% Transpor Xv se necessário (se as amostras estiverem nas colunas)
if num_features_val < num_samples_val
    Xv = Xv';
end

% Número de clusters desejado para os dados de validação
k_val = 2;

% Executar o k-means nos dados de validação
[idx_val, C_val] = kmeans(Xv, k_val);

% Converter índices de cluster em rótulos de cluster (1 a k_val)
cluster_labels_val = idx_val';

% Obter rótulos verdadeiros para os dados de validação
true_labels_val = (Yv(1, :) == 1) + 1;  % Transforma os rótulos em classe binária (1 ou 2)

% Calcular Acurácia para os dados de validação
accuracy_val = sum(true_labels_val == cluster_labels_val) / numel(true_labels_val);

% Calcular Matriz de Confusão para Precisão, Recall e F1-score
confusion_matrix_val = confusionmat(true_labels_val, cluster_labels_val);

% Calcular Precisão, Recall e F1-score para os dados de validação
precision_val = confusion_matrix_val(2,2) / (confusion_matrix_val(2,2) + confusion_matrix_val(1,2));
recall_val = confusion_matrix_val(2,2) / (confusion_matrix_val(2,2) + confusion_matrix_val(2,1));
f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val);

% Exibir métricas de avaliação para os dados de validação
disp('Métricas de Avaliação para Dados de Validação:');
disp(['Acurácia: ', num2str(accuracy_val)]);
disp(['Precisão: ', num2str(precision_val)]);
disp(['Recall: ', num2str(recall_val)]);
disp(['F1-score: ', num2str(f1_score_val)]);
