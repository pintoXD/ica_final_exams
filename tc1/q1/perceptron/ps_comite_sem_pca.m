% Implementacao de um classificador linear treinado pelo metodo dos minimos
% quadrados
%
% Autor: Guilherme Barreto
% Data: 25/01/2016

clear; clc; close all;
pkg load nan
pkg load statistics
pkg load statistics-bootstrap
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


%X=X'; D=D'; X_labels=X_labels'; D_labels=D_labels';

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

%X=[ones(N_x(2)*N_x(3), 1); X]; %Adiciona um 1 a ao primeiro elemento para servir de bias
%D=[ones(N_d(2)*N_d(3), 1) D]; %Adiciona um 1 a ao primeiro elemento para servir de bias
%D=(1./(1 + exp(-D))); %Normaliza os valores dos resultados com uma função sigmoid entre -1 e 1;
%D_labels=(1./(1 + exp(-D_labels))); %Normaliza os valores dos resultados com uma função sigmoid entre -1 e 1;


%n = size(X); %N(1) = Numero de pixels
             %N(2) = Numero de imagens

n = size(X); %N(2) = Numero de pixels
             %N(1) = Numero de imagens

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 2: Separar dados de treino/teste %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dados de treinamento (Xtr, Dtr)
Xtr=X;  Dtr=X_labels;

% Dados de teste (Xts, Dts)
Xts=D;  Dts=D_labels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=n(2);  % dimensao do vetor de entrada
Ne=5;  % Numero de epocas de treinamento (numero de vezes que o conjunto de treinamento eh reapresentado)
alfa=0.01; % Taxa de aprendizagem

num_classes = 10;
W=rand(p,num_classes);  % Inicializacao do vetor de pesos
pesos_totais_maquinas = 0;
Ntr = n(1);
%Ntr = 10;
Nts = size(Xts)(1);
num_maquinas = 5;

sigmoid = @(valor)1./(1 + exp(-valor));
mapped_labels = eye(num_classes);
tic();

for m=1:num_maquinas,

    bootrstrapped_index = boot(1:1:Ntr, 1);
    Xtr_maquina_atual = Xtr(bootrstrapped_index,:);
    Dtr_maquina_atual = Dtr(bootrstrapped_index,:);
    maquina=m
    
    for t=1:Ne,

        Epoca=t
        acc_erro_quad=0;  % Acumula erro quadratico por vetor em uma epoca
        for k=1:Ntr,

            activations = Xtr_maquina_atual(k,:) * W;  % Saida predita para k-esimo vetor de entrada
            normalized_activations = sigmoid(activations);

            erro =  mapped_labels(Dtr(k) + 1, :) - normalized_activations;  % erro de predicao
            
            W=W+alfa*Xtr_maquina_atual(k,:)'*erro; % Atualizacao do vetor de pesos
            acc_erro_quad=acc_erro_quad+0.5*sum(erro)*sum(erro);
        end
        erro_medio_epoca(t)=acc_erro_quad/Ntr;
    end
    pesos_totais_maquinas = pesos_totais_maquinas + W;
end



elapsed_time = toc ()



% figure; plot(erro_medio_epoca);
% title('Curva de Aprendizagem');
% xlabel('Epoca de treinamento');
% ylabel('Erro quadratico medio por epoca');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W_gerado_comite = pesos_totais_maquinas/num_maquinas;
Ypred=round(Xts*W_gerado_comite);        % Saida como numeros reais
num_test_data = size(Ypred)(1);

%Mapeando os maiores valores de cada linha pra label correspondente
for i=1:num_test_data,
   [probabilidade algarismo_identificado] = max(Ypred(i, :));
   Ypred_q(i) = algarismo_identificado - 1;
end


%Ypred_q=Ypred;  % Saida quantizada para 0 ou 1 usando uma sigmoide.
%Ypred_q=1./(1 + exp(-Ypred));  % Saida quantizada para 0 ou 1 usando uma sigmoide.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 5: Determinar as taxas de acerto/erro %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resultados=[Dts Ypred_q'];          % Saida desejada e predita lado-a-lado
Erros=Resultados(:,1)-Resultados(:,2);  % Coluna 1 - Coluna 2

Nerros_pos=length(find(Erros>0))
Nerros_neg=length(find(Erros<0))
Nacertos=Nts-(Nerros_pos+Nerros_neg)

Perros_pos=100*Nerros_pos/Nts
Perros_neg=100*Nerros_neg/Nts
Pacertos=100*Nacertos/Nts








