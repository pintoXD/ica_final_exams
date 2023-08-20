% Implementacao de um classificador nao-linear neural (ELM) treinado pelo 
% metodo dos minimos quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;

pkg load nan
pkg load statistics
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

N_x=size(X);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem

N_d=size(D);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem

N_xlabels=size(X_labels);

X = reshape(X, N_x(1),N_x(2)*N_x(3)); %Transformando de uma matriz 3D para uma 2D
D = reshape(D, N_d(1),N_d(2)*N_d(3)); %Transformando de uma matriz 3D para uma 2D


D=double(D);
D_labels=double(D_labels);
%D_labels=[ones(1,1), D_labels];
D_labels=[ones(1,1); D_labels];

X=double(X);
X_labels=double(X_labels);
%X_labels=[ones(1,1), X_labels];
X_labels=[ones(1,1); X_labels];

X=double(X/255.00); %Normaliza os valores dos pixels entre 0 e 1
D=double(D/255.00); %Normaliza os valores dos pixels entre 0 e 1

X = [ones(1, N_x(2)*N_x(3)); X];
D = [ones(1, N_d(2)*N_d(3)); D];

X=X'; D=D'; X_labels=X_labels'; D_labels=D_labels';


n=size(X);  % n(1)=dimensao da entrada, n(2)=numero de exemplos
size_dados_teste = size(D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_pixels_imagem=n(1);   % Numero de variaveis de entrada 
num_classes = 10;
N=n(2);   % Numero de exemplos = No. de colunas da matriz X
Ntr=N;  % Numero de casos para treinamento
num_test_data=size_dados_teste(2);  % Numero de casos de teste

mapa_de_classes = eye(num_classes); %Matriz diagonal 10x10 que auxilia no cálculo e rastreamento do erro
% Dados de treinamento (Xtr, Dtr)
Xtr=X;  Dtr=X_labels;
labels_size = size(Dtr);
aux_dtr = zeros(10, labels_size(2));
for i=1:labels_size(2), %Especie de one-hot enconding(?)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Escolha aleatoria dos pesos e %%
%%% limiares dos neuronios intermediarios %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigmoid = @(valor)1./(1 + exp(-valor));
       %De acordo com as heurísticas, esse é o mínimo de neurônios a se ter.
num_neuronios=1569;  % Numero de neuronios intermediarios de acordo com Kolmogorov
% num_neuronios=88;  % Numero de neuronios intermediarios de acordo com Kolmogorov


W=0.1*rand(num_pixels_imagem, num_neuronios);   % Determinacao da matriz da projecao aleatoria

% Utr=W'*Xtr;      % Parte linear da projecao na camada intermediaria
Utr=W'*Xtr;      % Parte linear da projecao na camada intermediaria
% Ztr=tanh(0.5*Utr);  % Parte nao-linear (aplicacao da funcao tangente hiperbolica)
Ztr = sigmoid(Utr); %Aplica função logistica nesse caso
%Ztr=(1-exp(-Utr))./(1+exp(-Utr));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=Dtr*pinv(Ztr);              % Equacao mais estavel numericamente
% M = sigmoid(Dtr*Ztr');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zts=tanh(0.5*W*Xts);
Zts=sigmoid(W'*Xts);
%Zts=(1-exp(-W*Xts))./(1+exp(-W*Xts));

% Ypred=M*Zts; % Saida como numeros reais
Ypred=sigmoid(M*Zts); % Saida como numeros reais

%Mapeando os maiores valores de cada coluna pra label correspondente
%Saída quantizada entre 0 e 9 inteiros
for j=1:num_test_data,
   [valor posicao] = max(Ypred(:, j));
   Ypred_q(j) = posicao - 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 6: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[Dts' Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);  % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=num_test_data-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/num_test_data
Perros_neg=100*Nerros_neg/num_test_data
Pacertos=100*Nacertos/num_test_data


