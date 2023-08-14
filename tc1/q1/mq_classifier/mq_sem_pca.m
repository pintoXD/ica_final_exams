% Implementacao de um classificador linear treinado pelo metodo dos minimos
% quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 1: Carregar banco de dados %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load ionosphere.data
[train_data, train_labels, test_data, test_labels] = load_mnist();
X=train_data;
X_labels=train_labels;
D=test_data;
D_labels=test_labels;

%X=X'; D=D';

N_x=size(X);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem

N_xlabels=size(X_labels);

X = reshape(X, N_x(1),N_x(2)*N_x(3)); %Transformando de uma matriz 3D para uma 2D
X_labels = reshape(X_labels, N_xlabels(1),N_xlabels(2)*N_xlabels(3)); %Transformando de uma matriz 3D para uma 2D

X=X'; D=D'; X_labels=X_labels'; D_labels=D_labels';

D=double(D);
D_labels=double(D_labels);

X=double(X/255.00); %Normaliza os valores dos pixels entre 0 e 1
X_labels=double(X_labels/255.00); %Normaliza os valores dos pixels entre 0 e 1

N = size(X); %N(1) = Numero de pixels
             %N(2) = Numero de imagens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%N=N(2);   % Numero de exemplos = No. de colunas da matriz X
%Ntr=floor(0.8*N);  % Numero de casos para treinamento
Nts=N(2);  % Numero de casos de teste

%I=randperm(N);

%X=[-ones(1,N);X];  % Adiciona uma linha de -1's
%X=X(:,I);  % Embaralha as colunas da matriz X

%D=D(I);    % Embaralha as colunas da matriz D para manter correspondencia

% Dados de treinamento (Xtr, Dtr)
Xtr=X;  Dtr=D;

% Dados de teste (Xts, Dts)
Xts=X_labels;  Dts=D_labels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%W=Dtr*Xtr'*inv(Xtr*Xtr');    % Equacao de livro-texto (teorica)
%W=D*X'*inv(X*X');             % Equacao de livro-texto (teorica)
W=Dtr/Xtr;                  % Solucao que usa decomposicao QR
%W=Dtr*pinv(Xtr);              % Solucao que usa decomposicao SVD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ypred=W*Xts;          % Saida como numeros reais
Ypred_q=round(Ypred);  % Saida quantizada para +1 ou -1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[Dts' Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);  % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=Nts-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/Nts
Perros_neg=100*Nerros_neg/Nts
Pacertos=100*Nacertos/Nts








