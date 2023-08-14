% Implementacao de um classificador linear treinado pelo metodo dos minimos
% quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 1: Carregar banco de dados %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load ionosphere.data
X=ionosphere(:,1:end-1);
D=ionosphere(:,end);

X=X'; D=D';

n=size(X);  % n(1)=dimensao da entrada, n(2)=numero de exemplos

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=n(2);   % Numero de exemplos = No. de colunas da matriz X
Ntr=floor(0.8*N);  % Numero de casos para treinamento
Nts=N-Ntr;  % Numero de casos de teste

I=randperm(N);

%X=[-ones(1,N);X];  % Adiciona uma linha de -1's
X=X(:,I);  % Embaralha as colunas da matriz X

D=D(I);    % Embaralha as colunas da matriz D para manter correspondencia

% Dados de treinamento (Xtr, Dtr)
Xtr=X(:,1:Ntr);  Dtr=D(:,1:Ntr);

% Dados de teste (Xts, Dts)
Xts=X(:,Ntr+1:end);  Dts=D(:,Ntr+1:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=n(1);  % dimensao do vetor de entrada
Ne=300;  % Numero de epocas de treinamento (numero de vezes que o conjunto de treinamento eh reapresentado)
alfa=0.001; % Taxa de aprendizagem

W=rand(p,1);  % Inicializacao do vetor de pesos

for t=1:Ne,
    Itr=randperm(Ntr);
    Xtr=Xtr(:,Itr);  % Embaralha dados a cada epoca de treinamento
    Dtr=Dtr(Itr);
    
    acc_erro_quad=0;  % Acumula erro quadratico por vetor em uma epoca
    for k=1:Ntr,
        ypred(k)=sign(W'*Xtr(:,k));  % Saida predita para k-esimo vetor de entrada
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








