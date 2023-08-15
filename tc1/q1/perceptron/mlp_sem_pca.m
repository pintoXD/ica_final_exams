% Implementacao da rede MLP canonica (backpropagation com fator de momento)
% Usando as funcoes built-in (internas) do matlab
%
% Exemplo para disciplina de ICA
% Autor: Guilherme de A. Barreto
% Date: 16/05/2023

%
% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc; close all
pkg load nan
pkg load statistics


% Carrega DADOS
%=================
%dados=load('derm_input.txt');  % Vetores (padroes) de entrada
%alvos=load('derm_target.txt'); % Saidas desejadas correspondentes
[train_data, test_data, train_labels, test_labels] = load_mnist();
%Converte os dados de uint8 para double e salva em variaveis proprias
dados_treino = double(train_data);
alvos_treino = double(train_labels);
dados_teste  = double(test_data);
alvos_teste  = double(test_labels);


% ===============================
% Faz uma prepação dos dados incluindo operações de reshape e normalização de dados de entrada
% ===============================
N_dados_treino=size(dados_treino);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem
N_dados_teste=size(dados_teste);  % N(1)=Numero de imagens, N(2)=dimensão X da i-esima imagem,
            % N(3)=dimensão Y da i-esima imagem


dados_treino = reshape(dados_treino, N_dados_treino(1),N_dados_treino(2)*N_dados_treino(3)); %Transformando de uma matriz 3D para uma 2D
dados_teste = reshape(dados_teste, N_dados_teste(1),N_dados_teste(2)*N_dados_teste(3)); %Transformando de uma matriz 3D para uma 2D



dados_treino=(dados_treino/255.00); %Normaliza os valores dos pixels entre 0 e 1
dados_teste=(dados_teste/255.00); %Normaliza os valores dos pixels entre 0 e 1

%Adiciona uma linha de 1s para servir de bias dos conjuntos de dados
dados_treino = [ones(1, N_dados_treino(2)*N_dados_treino(3)); dados_treino];
dados_teste = [ones(1, N_dados_teste(2)*N_dados_teste(3)); dados_teste];

%Adiciona uma linha de 1's para servir de correspondente ao bias adicionado aos dados de treino
alvos_treino=[ones(1,1); alvos_treino];
alvos_teste=[ones(1,1); alvos_teste];

N_dados_treino=size(dados_treino);  % N(1)=Numero de imagens, N(2)=Número de pixels da i-ésima imagem
N_dados_teste=size(dados_teste);  % N(1)=Numero de imagens, N(2)=Número de pixels da i-ésima imagem
N_alvos_treino=size(alvos_treino);
N_alvos_teste=size(alvos_teste);

% DEFINE ARQUITETURA DA REDE
%===========================
num_classes = 10;
num_imagens_treino = N_dados_treino(1);
num_pixels_imagem = N_dados_treino(2);
num_imagens_teste = N_dados_teste(1);
num_alvos_teste = N_alvos_teste(1);
mapa_de_classes = eye(num_classes); %Matriz diagonal 10x10 que auxilia no cálculo e rastreamento do erro
Ne = 10; % No. de epocas de treinamento
Nr = 1;   % No. de rodadas de treinamento/teste
Nh = 28;   % No. de neuronios na camada oculta
No = num_classes;   % No. de neuronios na camada de saida

eta=0.05;   % Passo de aprendizagem
mom=0.75;  % Fator de momento

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino
ptst=1-ptrn; % Porcentagem usada para teste

for r=1:Nr,

    Repeticao=r,

    # I=randperm(ColD);
    # Dn=Dn(:,I);
    # alvos=alvos(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

    # J=floor(ptrn*ColD);

    % Vetores para treinamento e saidas desejadas correspondentes
    # P = Dn(:,1:J); T1 = alvos(:,1:J);
    # [lP Ntrain]=size(P);   % Tamanho da matriz de vetores de treinamento

    # % Vetores para teste e saidas desejadas correspondentes
    # Q = Dn(:,J+1:end); T2 = alvos(:,J+1:end);
    # [lQ Ntest]=size(Q);   % Tamanho da matriz de vetores de teste

    % Inicia matrizes de pesos
    WW=0.1*rand(num_pixels_imagem, Nh);   % Pesos entrada -> camada oculta
    WW_old=WW;              % Necessario para termo de momento

    MM=0.01*rand(Nh,num_classes);   % Pesos camada oculta -> camada de saida
    MM_old = MM;            % Necessario para termo de momento

    %%% ETAPA DE TREINAMENTO
    for t=1:Ne,   % Inicio do loop de epocas

        # I=randperm(Ntrain); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento e saidas desejadas

        EQ=0;
        for tt=1:num_imagens_treino,   % Inicia LOOP de epocas de treinamento
            % CAMADA OCULTA
            # X=[+1; P(:,tt)];      % Constroi vetor de entrada
            X=dados_treino(tt, :);      % Constroi vetor de entrada
            Ui = X * WW;          % Ativacao (net) dos neuronios da camada oculta
            Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

            % CAMADA DE SAIDA
            Y=Yi;                 % Constroi vetor de entrada DESTA CAMADA
            Uk = Y * MM;          % Ativacao (net) dos neuronios da camada de saida
            Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)

            % CALCULO DO ERRO
            classe_mapeada = mapa_de_classes(alvos_treino(tt) + 1, :); %Especie de one-hot enconding
            Ek = classe_mapeada - Ok;     % erro entre a saida desejada e a saida da rede
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA


            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = Ok.*(1 - Ok) + 0.05;  % derivada da sigmoide logistica (camada de saida)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)

            Di = Yi.*(1 - Yi) + 0.05; % derivada da sigmoide logistica (camada oculta)
            DDi = Di.*(DDk * (MM)');    % gradiente local (camada oculta)

            % AJUSTE DOS PESOS - CAMADA DE SAIDA
            MM_aux=MM;
            MM = MM + eta*Y'*DDk + mom*(MM - MM_old);
            MM_old=MM_aux;

            % AJUSTE DOS PESOS - CAMADA OCULTA
            WW_aux=WW;
            WW = WW + eta*X'*DDi + mom*(WW - WW_old);
            WW_old=WW_aux;
        end   % Fim de uma epoca

        % MEDIA DO ERRO QUADRATICO P/ EPOCA
        EQMepoca(t)=EQ/num_imagens_treino;
    end   % Fim do loop de treinamento

    EQMtrain{r}=EQMepoca;   % Salva curva de aprendizagem para a r-esima repeticao

    %% ETAPA DE GENERALIZACAO  %%%
    OUT=[];
    EQMtested=0;
    for tt=1:num_imagens_teste,
        % CAMADA OCULTA
        X=dados_teste(tt, :); % Constroi vetor de entrada com adicao da entrada x0=1
        Ui = X*WW;           % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA
        Y=Yi;           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = Y*MM;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OUT=[OUT; Ok];    % Armazena saidas da rede

        classe_mapeada = mapa_de_classes(alvos_teste(tt) + 1, :); %Especie de one-hot enconding
        Ek = classe_mapeada - Ok;           % erro entre a saida desejada e a saida da rede

        % ERRO QUADRATICO GLOBAL (todos os neuronios) POR VETOR DE ENTRADA
        EQMtested = EQMtested + 0.5*sum(Ek.^2);
    end

    EQMtest(r)=EQMtested/num_imagens_teste;

    % CALCULA TAXA DE ACERTO
    count_OK=0;  % Contador de acertos
    for t=1:num_imagens_teste,
        [probabilidade algarismo_identificado] = max(OUT(t, :));
        
        if alvos_teste(t,:)==(algarismo_identificado - 1),   % Conta acerto se os dois indices coincidem
            count_OK=count_OK+1;
        end
    end

    Tx_OK(r)=100*(count_OK/num_imagens_teste); % Taxa de acerto global
end

[Tx_OK_max r_max]=max(Tx_OK); % Encontra rodada que gerou maior Tx_OK e armazena em r_max
[Tx_OK_min r_min]=min(Tx_OK); % Encontra rodada que gerou menor Tx_OK e armazena em r_min
Tx_OK_media=mean(Tx_OK); % Exibe media da taxa de acerto para as Nr rodadas
Tx_OK_desvio=std(Tx_OK); % Exibe desvio-padrao da taxa de acerto para as Nr rodadas
Tx_OK_mediana=median(Tx_OK);  % Exibe mediana da taxa de acerto dentre as Nr rodadas

STATS=[Tx_OK_media Tx_OK_desvio Tx_OK_min Tx_OK_max Tx_OK_mediana]

% Graficos
figure;
plot(1:Ne,EQMtrain{r_max},'linewidth',2); xlabel('Epocas');
ylabel('Erro Medio Quadratico');
title('Curva de Aprendizagem para Melhor Caso')
grid, set(gca,"fontsize", 12)

figure;
plot(1:Ne,EQMtrain{r_min},'linewidth',2); xlabel('Epocas');
ylabel('Erro Medio Quadratico');
title('Curva de Aprendizagem para Pior Caso')
grid, set(gca,"fontsize", 12)

# figure; boxplot(Tx_OK,'linewidth',2);
# title('Boxplot da taxa de acerto para Nr rodadas')
# set(gca,"fontsize", 12)

# figure; histfit(Tx_OK);
# set(gca, "fontsize", 12)
# title('Histograma da taxa de acerto para Nr rodadas')


