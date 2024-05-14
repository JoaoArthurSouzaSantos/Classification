load('Xt.mat'); % Características de entrada de treinamento
load('Yt.mat'); % Variáveis de saída de treinamento

load('Xv.mat'); % Características de entrada de validação
load('Yv.mat'); % Variáveis de saída de validação

% Definir a arquitetura da MLP
hiddenLayerSize = 16; % Número de neurônios na camada oculta
net = patternnet(hiddenLayerSize); % Criar a rede neural feedforward

% Configurar os parâmetros da rede
net.trainParam.epochs = 1000; % Número de épocas de treinamento
net.trainParam.lr = 0.001; % Taxa de aprendizado
net.trainParam.min_grad = 1e-16; % Critério de parada (gradiente mínimo)
net.trainParam.showWindow = true; % Mostrar janela de treinamento

net = train(net, Xt, Yt);

Yv_pred = net(Xv);
classes = vec2ind(Yv_pred);
classes1=vec2ind(Yv);

acc=classes-classes1;
acc1=find(acc~=0);
acuracy =(1-length(acc1)/length(Yt))*100

% Calcular True Positives (TP), False Positives (FP) e False Negatives (FN)
TP = sum(classes == 1 & classes1 == 1); % Verdadeiros Positivos
FP = sum(classes == 1 & classes1 ~= 1); % Falsos Positivos
FN = sum(classes ~= 1 & classes1 == 1); % Falsos Negativos

% Calcular Recall
recall = TP / (TP + FN);

% Calcular Precisão
precision = TP / (TP + FP);

% Calcular F1-score
F1_score = 2 * (precision * recall) / (precision + recall);

% Exibir os resultados
fprintf('Recall: %.2f\n', recall);
fprintf('Precisão: %.2f\n', precision);
fprintf('F1-score: %.2f\n', F1_score);
fprintf('Recall: %.2f\n', acuracy);