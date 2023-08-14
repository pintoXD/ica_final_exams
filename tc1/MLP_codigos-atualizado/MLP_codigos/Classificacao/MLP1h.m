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
pkg load statistics


% Carrega DADOS
%=================
%dados=load('derm_input.txt');  % Vetores (padroes) de entrada
%alvos=load('derm_target.txt'); % Saidas desejadas correspondentes

dados=load('wine_input.txt');  % Vetores (padroes) de entrada
alvos=load('wine_target.txt'); % Saidas desejadas correspondentes

% Embaralha vetores de entrada e saidas desejadas
[LinD ColD]=size(dados);

% Normaliza componentes para media zero e variancia unitaria
for i=1:LinD,
	mi=mean(dados(i,:));  % Media das linhas
  di=std(dados(i,:));   % desvio-padrao das linhas
	dados(i,:)= (dados(i,:) - mi)./di;
end
Dn=dados;

% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 150; % No. de epocas de treinamento
Nr = 10;   % No. de rodadas de treinamento/teste
Nh = 15;   % No. de neuronios na camada oculta
No = length(alvos(:,1));   % No. de neuronios na camada de saida

eta=0.05;   % Passo de aprendizagem
mom=0.75;  % Fator de momento

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino
ptst=1-ptrn; % Porcentagem usada para teste

for r=1:Nr,

    Repeticao=r,

    I=randperm(ColD);
    Dn=Dn(:,I);
    alvos=alvos(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

    J=floor(ptrn*ColD);

    % Vetores para treinamento e saidas desejadas correspondentes
    P = Dn(:,1:J); T1 = alvos(:,1:J);
    [lP Ntrain]=size(P);   % Tamanho da matriz de vetores de treinamento

    % Vetores para teste e saidas desejadas correspondentes
    Q = Dn(:,J+1:end); T2 = alvos(:,J+1:end);
    [lQ Ntest]=size(Q);   % Tamanho da matriz de vetores de teste

    % Inicia matrizes de pesos
    WW=0.01*rand(Nh,lP+1);   % Pesos entrada -> camada oculta
    WW_old=WW;              % Necessario para termo de momento

    MM=0.01*rand(No,Nh+1);   % Pesos camada oculta -> camada de saida
    MM_old = MM;            % Necessario para termo de momento

    %%% ETAPA DE TREINAMENTO
    for t=1:Ne,   % Inicio do loop de epocas

        I=randperm(Ntrain); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento e saidas desejadas

        EQ=0;
        for tt=1:Ntrain,   % Inicia LOOP de epocas de treinamento
            % CAMADA OCULTA
            X=[+1; P(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
            Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
            Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

            % CAMADA DE SAIDA
            Y=[+1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
            Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
            Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)

            % CALCULO DO ERRO
            Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA


            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = Ok.*(1 - Ok) + 0.05;  % derivada da sigmoide logistica (camada de saida)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)

            Di = Yi.*(1 - Yi) + 0.05; % derivada da sigmoide logistica (camada oculta)
            DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

            % AJUSTE DOS PESOS - CAMADA DE SAIDA
            MM_aux=MM;
            MM = MM + eta*DDk*Y' + mom*(MM - MM_old);
            MM_old=MM_aux;

            % AJUSTE DOS PESOS - CAMADA OCULTA
            WW_aux=WW;
            WW = WW + eta*DDi*X' + mom*(WW - WW_old);
            WW_old=WW_aux;
        end   % Fim de uma epoca

        % MEDIA DO ERRO QUADRATICO P/ EPOCA
        EQMepoca(t)=EQ/Ntrain;
    end   % Fim do loop de treinamento

    EQMtrain{r}=EQMepoca;   % Salva curva de aprendizagem para a r-esima repeticao

    %% ETAPA DE GENERALIZACAO  %%%
    OUT=[];
    EQMtested=0;
    for tt=1:Ntest,
        % CAMADA OCULTA
        X=[+1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW*X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA
        Y=[+1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM*Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OUT=[OUT Ok];    % Armazena saidas da rede

        Ek = T2(:,tt) - Ok;   % erro entre a saida desejada e a saida da rede

        % ERRO QUADRATICO GLOBAL (todos os neuronios) POR VETOR DE ENTRADA
        EQMtested = EQMtested + 0.5*sum(Ek.^2);
    end

    EQMtest(r)=EQMtested/Ntest;

    % CALCULA TAXA DE ACERTO
    count_OK=0;  % Contador de acertos
    for t=1:Ntest,
        [T2max iT2max]=max(T2(:,t));  % Indice da saida desejada de maior valor
        [OUT_max iOUT_max]=max(OUT(:,t)); % Indice do neuronio cuja saida eh a maior
        if iT2max==iOUT_max,   % Conta acerto se os dois indices coincidem
            count_OK=count_OK+1;
        end
    end

    Tx_OK(r)=100*(count_OK/Ntest); % Taxa de acerto global
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

figure; boxplot(Tx_OK,'linewidth',2);
title('Boxplot da taxa de acerto para Nr rodadas')
set(gca,"fontsize", 12)

figure; histfit(Tx_OK);
set(gca, "fontsize", 12)
title('Histograma da taxa de acerto para Nr rodadas')


