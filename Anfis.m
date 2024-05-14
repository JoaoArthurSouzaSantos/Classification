% Carregar os dados (substitua com seus dados reais)
load('Xt.mat'); % Características de entrada de treinamento
load('Yt.mat'); % Variáveis de saída de treinamento
load('Xv.mat'); % Características de entrada de validação
load('Yv.mat'); % Variáveis de saída de validação

% Dados de treinamento
inputDataTrain = Xt;
outputDataTrain = Yt;

% Dados de validação
inputDataValidation = Xv;
outputDataValidation = Yv;

% Configurar opções para genfis
opt = genfisOptions('SubtractiveClustering');
opt.ClusterInfluenceRange = [0.25 0.5]; 

% Gerar o FIS
fis = genfis(inputDataTrain', outputDataTrain', opt);

% Converter dados de treinamento e validação em formato adequado para ANFIS
trainData = [inputDataTrain outputDataTrain];
valData = [inputDataValidation outputDataValidation];

% Definir número de épocas para treinamento ANFIS
numEpochs = 50;

% Treinar o modelo ANFIS
disp('Treinando modelo ANFIS...');
anfisModel = anfis(trainData, fis, numEpochs, valData);

disp('Treinamento concluído.');

% Avaliar o desempenho do modelo ANFIS nos dados de treinamento e validação
trainOutputs = evalfis(inputDataTrain, anfisModel);
valOutputs = evalfis(inputDataValidation, anfisModel);

% Calcular métricas de desempenho (Erro Médio Quadrático - MSE)
trainMSE = mean((outputDataTrain - trainOutputs').^2);
valMSE = mean((outputDataValidation - valOutputs').^2);

% Exibir resultados de desempenho
disp(['Erro médio quadrático (MSE) nos dados de treinamento: ' num2str(trainMSE)]);
disp(['Erro médio quadrático (MSE) nos dados de validação: ' num2str(valMSE)]);

% Plotar saídas reais vs. previstas nos dados de validação
figure;
plot(outputDataValidation(:,1), outputDataValidation(:,2), 'bo', ...
     valOutputs(:,1), valOutputs(:,2), 'rx');
xlabel('Saída Real - Variável 1');
ylabel('Saída Real - Variável 2');
title('Saídas Reais vs. Previstas (Dados de Validação)');
legend('Real', 'Previsto');
grid on;
