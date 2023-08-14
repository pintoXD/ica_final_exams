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


D=double(1./(1 + exp(-D))); %Normaliza os valores dos resultados com uma função sigmoid entre -1 e 1;
D_labels=double(1./(1 + exp(-D_labels))); %Normaliza os valores dos resultados com uma função sigmoid entre -1 e 1;

X=double(X);
X_labels=double(X_labels);
n = size(X); %N(1) = Numero de pixels
             %N(2) = Numero de imagens

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dados de treinamento (Xtr, Dtr)
Xtr=X;  Dtr=D;

% Dados de teste (Xts, Dts)
Xts=X_labels;  Dts=D_labels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=n(1);  % dimensao do vetor de entrada
Ne=300;  % Numero de epocas de treinamento (numero de vezes que o conjunto de treinamento eh reapresentado)
alfa=0.01; % Taxa de aprendizagem

W=rand(p,1);  % Inicializacao do vetor de pesos
Ntr = n(2);

for t=1:Ne,
    Itr=randperm(Ntr);
    Xtr=Xtr(:,Itr);  % Embaralha dados a cada epoca de treinamento
    Dtr=Dtr(Itr);

    acc_erro_quad=0;  % Acumula erro quadratico por vetor em uma epoca
    for k=1:Ntr,
        aux_sigmoid = W'*Xtr(:,k);
        sigmoid = 1./(1 + exp(-aux_sigmoid));
        ypred(k)=sigmoid;  % Saida predita para k-esimo vetor de entrada
        erro(k)=Dtr(k)-ypred(k);  % erro de predicao
        W=W+alfa*erro(k)*Xtr(:,k); % Atualizacao do vetor de pesos
        acc_erro_quad=acc_erro_quad+ 0.5*erro(k)*erro(k);
    end
    erro_medio_epoca(t)=acc_erro_quad/Ntr;
end

figure; plot(erro_medio_epoca);
title('Curva de Aprendizagem');
xlabel('Epoca de treinamento');
ylabel('Erro quadratico medio por epoca');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ypred=W'*Xts;          % Saida como numeros reais
Ypred_q=sign(Ypred);  % Saida quantizada para +1 ou -1.

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








