% Implementacao de um classificador linear treinado pelo metodo dos minimos
% quadrados
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


aux_pca = [X; D]; %Une dados de treino e teste para fazer PCA
aux_pca_executado = execute_pca(aux_pca'); %Executa PCA em cima dos dados de treino e teste

X = aux_pca_executado(:, 1:N_x(1))'; %Separa os dados de treino de novo
D = aux_pca_executado(:, N_x(1) + 1:end)'; %Separa os dados de teste

size_X_pca = size(X);
size_D_pca = size(D);

X = [ones(1, size_X_pca(2)); X];
D = [ones(1, size_D_pca(2)); D];

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


% %Aplica PCA nos dados de teste e treino
% pca_Xtr=execute_pca(Xtr'); pca_Xts=execute_pca(Xts');
% Xtr = pca_Xtr'; Xts = pca_Xts';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 3: Estimar os parametros do classificador (pesos e limiares) %%
%%% pelo metodo dos minimos quadrados (classificador sem camada oculta)%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=n(2);  % dimensao do vetor de entrada
numero_de_epocas=500;  % Numero de epocas de treinamento (numero de vezes que o conjunto de treinamento eh reapresentado)
alfa=0.01; % Taxa de aprendizagem

num_classes = 10;
W=rand(p,num_classes);  % Inicializacao do vetor de pesos
Ntr = n(1);
%Ntr = 10;
Nts = size(Xts)(1);

sigmoid = @(valor)1./(1 + exp(-valor));
mapped_labels = eye(num_classes);

tic();
for t=1:numero_de_epocas,
    %Itr=randperm(Ntr);
    %Xtr=Xtr(:,Itr);  % Embaralha dados a cada epoca de treinamento
    %Dtr=Dtr(Itr);
    Epoca=t
    acc_erro_quad=0;  % Acumula erro quadratico por vetor em uma epoca
    for k=1:Ntr,
        %aux_sigmoid = W'*Xtr(:,k);
        %sigmoid = 1./(1 + exp(-aux_sigmoid));
        %ypred(k)=sigmoid;  % Saida predita para k-esimo vetor de entrada
        %ypred(k)=(W'*Xtr(:,k));  % Saida predita para k-esimo vetor de entrada
        activations(k, :) = Xtr(k,:) * W;  % Saida predita para k-esimo vetor de entrada
        normalized_activations(k, :) = sigmoid(activations(k, :));

        erro(k, :) =  mapped_labels(Dtr(k) + 1, :) - normalized_activations(k,:);  % erro de predicao
        %erro(k)=Dtr(k)-ypred(k);  % erro de predicao
        W=W+alfa*Xtr(k,:)'*erro(k,:); % Atualizacao do vetor de pesos
        acc_erro_quad=acc_erro_quad+0.5*sum(erro(k,:))*sum(erro(k,:));
    end
    erro_medio_epoca(t)=acc_erro_quad/Ntr;
end
elapsed_time = toc ()
figure; plot(erro_medio_epoca);
title('Curva de Aprendizagem');
xlabel('Epoca de treinamento');
ylabel('Erro quadratico medio por epoca');
print("ps_com_pca_aprendizado.png");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Passo 4: Determinar predicoes da classe dos vetores de teste %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ypred=round(Xts*W);        % Saida como numeros reais
num_test_data = size(Ypred)(1);

%Mapeando os maiores valores de cada linha pra label correspondente
for i=1:num_test_data,
   [valor posicao] = max(Ypred(i, :));
   Ypred_q(i) = posicao - 1;
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



save -text ps_com_pca_20_08.txt numero_de_epocas Nerros_pos Nerros_neg Nacertos Perros_pos Perros_neg Pacertos elapsed_time;




