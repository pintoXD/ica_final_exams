% Implementacao de um classificador nao-linear neural (ELM) treinado pelo
% metodo dos minimos quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 1: Carregar banco de dados %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
real_state_read_data = dlmread("real_state_dataset.csv", ";", 1, 1);
X=real_state_read_data(:,1:end-1);
D=real_state_read_data(:,end);

% Normaliza dados ruidosos
x_min=min(X); x_max=max(X);  % limites necessarios para normalizacao
d_min=min(D); d_max=max(D); % limites necessarios para normalizacao
X=2*(X - x_min)/(x_max-x_min)-1; % normaliza entradas entre -1 e 1
D=2*(D - d_min)/(d_max-d_min)-1;  % normaliza saidas entre -1 e 1


X=X'; D=D';

n=size(X);  % n(1)=dimensao da entrada, n(2)=numero de exemplos

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=n(1);   % Numero de variaveis de entrada
N=n(2);   % Numero de exemplos = No. de colunas da matriz X
Ntr=floor(0.8*N);  % Numero de casos para treinamento
Nts=N-Ntr;  % Numero de casos de teste

I=randperm(N);

X=[-ones(1,N);X];  % Adiciona uma linha de -1's
X=X(:,I);  % Embaralha as colunas da matriz X

D=D(I);    % Embaralha as colunas da matriz D para manter correspondencia

% Dados de treinamento (Xtr, Dtr)
Xtr=X(:,1:Ntr);  Dtr=D(:,1:Ntr);

% Dados de teste (Xts, Dts)
Xts=X(:,Ntr+1:end);  Dts=D(:,Ntr+1:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Escolha aleatoria dos pesos e %%
%%% limiares dos neuronios intermediarios %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=200;  % Numero de neuronios intermediarios

W=0.5*rand(Q,p+1);   % Determinacao da matriz da projecao aleatoria

Utr=W*Xtr;      % Parte linear da projecao na camada intermediaria
Ztr=tanh(0.5*Utr);  % Parte nao-linear (aplicacao da funcao tangente hiperbolica)
%Ztr=(1-exp(-Utr))./(1+exp(-Utr));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=Dtr*pinv(Ztr);              % Equacao mais estavel numericamente

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zts=tanh(0.5*W*Xts);
%Zts=(1-exp(-W*Xts))./(1+exp(-W*Xts));

Ypred=M*Zts; % Saida como numeros reais

Ypred_q=sign(Ypred);  % Saida quantizada para +1 ou -1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 6: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[Dts' Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);  % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=Nts-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/Nts
Perros_neg=100*Nerros_neg/Nts
Pacertos=100*Nacertos/Nts


