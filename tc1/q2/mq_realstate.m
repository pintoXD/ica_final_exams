clear; close all;

%Operação para leitura dos dados e organização para realização dos cálculos
real_state_read_data = dlmread("real_state_dataset.csv", ";", 1, 1);
y_real = real_state_read_data(:,end);
x_values = real_state_read_data(:,[1:end-1]);

%Fórmula para o método dos mínimos quadrados
B=inv(x_values' * x_values) * x_values' * y_real;

%Valor predito de acordo com os coeficientes obtidos
y_pred = x_values * B;

k = size(x_values)(1);

%Avaliação do erro
erro = y_real-y_pred; %Calula o erro entre cada valor predito e original
SAE = sum(abs(erro)) %Calcula a soma dos erros absolutos
MAE = abs(mean(erro))  %Calcula media dos erros absolutos
SEQ = sum(erro.^2) %Calcula a soma dos erros quadráticos
EQM = mean(erro.^2)
ymed = mean(y_real); %Calcula a media original dos valores das casas
Syy = sum((y_real-ymed).^2) %Soma dos erros para um modelo baseado em média
AIC = (length(x_values)*log(SEQ))+2*k
R2 = 1 - (SEQ/Syy) %Calcula o R^2 da predição









