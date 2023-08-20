% Implementacao de um classificador linear treinado pelo metodo dos minimos
% quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;
pkg load nan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 1: Carregar banco de dados %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load ionosphere.data
[train_data, test_data, train_labels, test_labels] = load_mnist();
X=train_data;
X_labels=train_labels;
D=test_data;
D_labels=test_labels;

%X=X'; D=D';
tic();

N_x=size(X);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem

N_d=size(D);

X = reshape(X, N_x(1),N_x(2)*N_x(3)); %Transformando de uma matriz 3D para uma 2D
D = reshape(D, N_d(1),N_d(2)*N_d(3)); %Transformando de uma matriz 3D para uma 2D

X=X'; D=D'; X_labels=X_labels'; D_labels=D_labels';

num_classes = 10;
X=double(X);
D=double(D);
X_labels=double(X_labels);
D_labels=double(D_labels);

aux_pca = [X D]; %Une dados de treino e teste para fazer PCA
aux_pca_executado = execute_pca(aux_pca); %Executa PCA em cima dos dados de treino e teste

X = aux_pca_executado(:, 1:N_x(1)); %Separa os dados de treino de novo
D = aux_pca_executado(:, N_x(1) + 1:end); %Separa os dados de teste

% X_pca=execute_pca(X); %Executa PCA em cima das variáveis de entrada de treinamento
% D_pca=execute_pca(D); %Executa PCA em cima das variáveis de entrada de teste

N = size(X); %N(1) = Numero de pixels
             %N(2) = Numero de imagens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nts=size(D)(2);  % Numero de casos de teste

%I=randperm(N);

X=[-ones(1,N(2));X];  % Adiciona uma linha de -1's
D=[-ones(1,size(D)(2));D];  % Adiciona uma linha de -1's

%X=X(:,I);  % Embaralha as colunas da matriz X

%D=D(I);    % Embaralha as colunas da matriz D para manter correspondencia
sigmoid = @(valor)1./(1 + exp(-valor));
% Dados de treinamento (Xtr, Dtr)
Xtr=X;  Dtr=X_labels;
% Dtr = sigmoid(Dtr);
labels_size = size(Dtr);
aux_dtr = zeros(10, labels_size(2));
for i=1:labels_size(2),
    aux_dtr(Dtr(1, i) + 1, i) = 1;
end
Dtr = aux_dtr;

% Dados de teste (Xts, Dts)
Xts=D;  Dts=D_labels;
labels_size = size(Dts);
aux_dts = zeros(10, labels_size(2));
for i=1:labels_size(2),
    aux_dts(Dts(1, i) + 1, i) = 1;
end
Dts=aux_dts;

% Xtr=execute_pca(Xtr); %Executa PCA em cima das variáveis de entrada de treinamento
% Xts=execute_pca(Xts); %Execetura PCA em cima das variáveis de entrada de teste

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W=Dtr*pinv(Xtr);              % Solucao que usa decomposicao SVD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ypred=W*Xts;          % Saida como numeros reais

for j=1:Nts,
   [valor posicao] = max(Ypred(:, j));
   Ypred_q(j) = posicao - 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[D_labels' Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);    % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=Nts-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/Nts
Perros_neg=100*Nerros_neg/Nts
Pacertos=100*Nacertos/Nts


elapsed_time = toc()
save -text mq_com_pca_out.txt Nerros_pos Nerros_neg Nacertos Perros_pos Perros_neg Pacertos elapsed_time;