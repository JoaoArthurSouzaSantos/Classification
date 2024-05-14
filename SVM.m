clear; close all; clc;

%% Carregamento dos Dados de Treinamento
load('Xt.mat'); % Carrega o conjunto de dados de entrada de treinamento Xt
load('Yt.mat'); % Carrega o conjunto de dados de saída de treinamento Yt

X_train = Xt'; % Transpõe Xt para ter amostras como linhas e características como colunas
y_train = Yt'; % Transpõe Yt para ter amostras como linhas e classes como colunas

%% Carregamento dos Dados de Validação
load('Xv.mat'); % Carrega o conjunto de dados de entrada de validação Xv
load('Yv.mat'); % Carrega o conjunto de dados de saída de validação Yv

X_valid = Xv'; % Transpõe Xv para ter amostras como linhas e características como colunas
y_valid = Yv'; % Transpõe Yv para ter amostras como linhas e classes como colunas

%% Partição para Validação Cruzada (CV)
c = cvpartition(y_train(:,1),'KFold',5); % Partição em 5 folds usando a primeira coluna de y_train como classe

%% Seleção de Características
opts = statset('display','iter');
classf = @(train_data, train_labels, test_data, test_labels) ...
    sum(predict(fitcsvm(train_data, train_labels(:,1),'KernelFunction','rbf'), test_data) ~= test_labels(:,1));

[fs, history] = sequentialfs(classf, X_train, y_train(:,1), 'cv', c, 'options', opts, 'nfeatures', 2);

%% Melhores Hiperparâmetros
X_train_with_best_features = X_train(:,fs);

Md1 = fitcsvm(X_train_with_best_features, y_train(:,1), 'KernelFunction', 'rbf', 'OptimizeHyperparameters', 'auto', ...
      'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', true));

%% Avaliação Final com Dados de Validação
X_valid_with_best_features = X_valid(:,fs);
y_valid_pred = predict(Md1, X_valid_with_best_features);

% Calcular acurácia da validação
accuracy_validation = sum(y_valid_pred == y_valid(:,1)) / length(y_valid(:,1)) * 100;

disp(['Acurácia da validação: ' num2str(accuracy_validation) '%']);

%% Visualização dos Resultados
figure;
hgscatter = gscatter(X_train_with_best_features(:,1), X_train_with_best_features(:,2), y_train(:,1));
hold on;
h_sv = plot(Md1.SupportVectors(:,1), Md1.SupportVectors(:,2), 'ko', 'markersize', 8);

gscatter(X_valid_with_best_features(:,1), X_valid_with_best_features(:,2), y_valid(:,1), 'rb', 'xx');
legend('Classe 1', 'Classe 2', 'Support Vectors', 'Location', 'best');
xlabel('Feature 1');
ylabel('Feature 2');
title('Visualização dos Dados de Treinamento e Validação');

