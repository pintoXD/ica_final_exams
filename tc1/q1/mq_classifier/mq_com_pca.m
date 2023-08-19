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

tic();
%X=X'; D=D';

N_x=size(X);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem

N_d=size(D);

X = reshape(X, N_x(1),N_x(2)*N_x(3)); %Transformando de uma matriz 3D para uma 2D
D = reshape(D, N_d(1),N_d(2)*N_d(3)); %Transformando de uma matriz 3D para uma 2D

X=X'; D=D'; X_labels=X_labels'; D_labels=D_labels';

X=double(X);
D=double(D);
X_labels=double(X_labels);
D_labels=double(D_labels);
num_classes = 10;

X=double(X/255.00); %Normaliza os valores dos pixels entre 0 e 1
D=double(D/255.00); %Normaliza os valores dos pixels entre 0 e 1

N = size(X); %N(1) = Numero de pixels
             %N(2) = Numero de imagens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%N=N(2);   % Numero de exemplos = No. de colunas da matriz X
%Ntr=floor(0.8*N);  % Numero de casos para treinamento
Nts=size(D)(2);  % Numero de casos de teste

%I=randperm(N);

%X=[-ones(1,N);X];  % Adiciona uma linha de -1's
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
    aux_dts(Dtr(1, i) + 1, i) = 1;
end
Dts=aux_dts;
% Dts = sigmoid(Dts);

% Aplica PCA aos dados de teste e de treino
Xtr=execute_pca(Xtr); Xts=execute_pca(Xts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%W=Dtr*Xtr'*inv(Xtr*Xtr');    % Equacao de livro-texto (teorica)
%W=D*X'*inv(X*X');             % Equacao de livro-texto (teorica)
% W=Dtr/Xtr;                  % Solucao que usa decomposicao QR
W=Dtr*pinv(Xtr);              % Solucao que usa decomposicao SVD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ypred=W*Xts;          % Saida como numeros reais

for j=1:Nts,
   [valor posicao] = max(Ypred(:, j));
   Ypred_q(j) = posicao - 1;
end

% Ypred_q=round(Ypred);  % Saida quantizada para +1 ou -1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[D_labels' Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);  % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=Nts-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/Nts
Perros_neg=100*Nerros_neg/Nts
Pacertos=100*Nacertos/Nts


elapsed_time = toc()
save -text mq_sem_pca_out.txt Nerros_pos Nerros_neg Nacertos Perros_pos Perros_neg Pacertos elapsed_time;




