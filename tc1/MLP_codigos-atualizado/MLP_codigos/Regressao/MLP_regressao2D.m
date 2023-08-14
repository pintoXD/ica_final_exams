% Implementacao da rede MLP canonica (backpropagation com gradiente descendente e termo de momento)
%
% Autor: Guilherme de A. Barreto
% Date: 16/05/2017
%
% Objetivo verificar a ocorrencia de overfitting

% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc; close all;

% Carrega DADOS
%=================
X=load('robot2D.dat');   % Dados robo planar (2D)

c=size(X);
I=randperm(c(1));
X=X(I,:);   % Embaralha as linhas do conjunto de dados

ptrain=0.8;  % Porcentagem de dados para treinamento
s=floor(ptrain*c(1));  % Ponto de corte para dados treinamento/teste

Xtrain=X(1:s,:);        % Dados de treinamento
Ntrain=length(Xtrain);  % No. dados de treinamento

Xtest=X(s+1:end,:);     % Dados de teste
Ntest=length(Xtest);    % No. dados de teste

% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 500; % No. de epocas de treinamento
Nr = 1;   % No. de rodadas de treinamento/teste
Nh = 20;   % No. de neuronios na camada oculta
No = 2;   % No. de neuronios na camada de saida

eta=0.01;   % Passo de aprendizagem
mom=0;  % Fator de momento

% Inicia matrizes de pesos
WW=0.1*rand(Nh,3);   % Pesos entrada -> camada oculta
WW_old=WW;              % Necessario para termo de momento

MM=0.1*rand(No,Nh+1);   % Pesos camada oculta -> camada de saida
MM_old = MM;            % Necessario para termo de momento

%%% ETAPA DE TREINAMENTO
for t=1:Ne,

    Epoca=t

    % Embaralha vetores de entrada e saidas desejadas
    I=randperm(Ntrain);
    Xtrain=Xtrain(I,:);

    EQ=0;
    for tt=1:Ntrain,   % Inicia LOOP de epocas de treinamento
        % CAMADA OCULTA
        x =[+1; Xtrain(tt,1:2)'];      % Constroi vetor de entrada com adicao da entrada x0=+1
        Ui = WW * x;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 2./(1+exp(-Ui)) - 1; % Saida entre [-1,1] (funcao tangente hiperbolica)

        % CAMADA DE SAIDA
        Y=[+1; Yi];               % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=+1
        Uk = MM * Y;              % Ativacao (net) dos neuronios da camada de saida
        Ok = Uk;                  % Neuronios de Saida com funcao de ativacao linear

        % CALCULO DO VETOR DE ERROS
        Ek = Xtrain(tt,3:4)' - Ok;     % erro entre a saida desejada e a saida da rede

        EQ = EQ + 0.5*sum(Ek.^2);    % soma dos erros quadratico para todos os neuronios por VETOR DE ENTRADA

        %%% CALCULO DOS GRADIENTES LOCAIS
        Dk = 1;             % Neuronios de saida lineares, logo derivada da funcao de ativacao igual a 1
        DDk = Ek.*Dk;       % gradiente local = proprio erro (camada de saida)

        Di = 1 - Yi.^2; % derivada da sigmoide logistica (camada oculta)
        DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

        % AJUSTE DOS PESOS - CAMADA DE SAIDA
        MM_aux=MM;
        MM = MM + eta*DDk*Y' + mom*(MM - MM_old);
        MM_old=MM_aux;

        % AJUSTE DOS PESOS - CAMADA OCULTA
        WW_aux=WW;
        WW = WW + eta*DDi*x' + mom*(WW - WW_old);
        WW_old=WW_aux;
    end   % Fim de uma epoca

    % MEDIA DO ERRO QUADRATICO P/ EPOCA
    EQMtrain(t)=EQ/Ntrain;
end   % Fim do loop de treinamento


% VERIFICACAO DE REDUNDANCIA COM A REDE JAH TREINADA
% USA DADOS DE TREINAMENTO, MAS NAO ALTERA OS PESOS
EQM_trained=0;
HID1=[];
OUT1=[];
SSE=[0;0];
SSY=[0;0];
MED=mean(Xtrain(:,3:4));   % Valores medios das 2 saidas
for tt=1:Ntrain,
    % CAMADA OCULTA
        x=[+1; Xtrain(tt,1:2)'];           % Constroi vetor de entrada com adicao da entrada x0=+1
        Ui = WW * x;               % Ativacao (net) dos neuronios da camada oculta
        Yi = 2./(1+exp(-Ui)) - 1;  % Saida entre [-1,1] (funcao tangente hiperbolica)
        HID1=[HID1 Yi];            % Armazena saida dos neuronios ocultos

        % CAMADA DE SAIDA
        Y=[+1; Yi];                % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=+1
        Uk = MM * Y;               % Ativacao (net) dos neuronios da camada de saida
        Ok = Uk;                   % Saida linear

        OUT1=[OUT1 Ok];        % Armazena saida dos neuronios de saida

        Ek = Xtrain(tt,3:4)' - Ok;           % erro entre a saida desejada e a saida da rede
        EQM_trained = EQM_trained + 0.5*sum(Ek.^2);     % soma do erro quadratico para todos os neuronios por VETOR DE ENTRADA

        SSE=SSE + Ek.^2;  % Soma dos erros quadraticos por neuronio de saida
        SSY=SSY + (Xtrain(tt,3:4)' - MED').^2;  % Soma dos desvios quadráticos das saidas
end

R2= 1 - SSE./SSY;

% MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TREINAMENTO)
EQM_trained=EQM_trained/Ntrain;

Ch=cov(HID1');  % Matriz de covariancia das saidas dos neuronios da camada OCULTA
Av=eig(Ch);     % Autovalores da matriz Ch
Rc=1/cond(Ch)    % Razao entre menor e maior autovalor da matriz Ch

%I=1; Plam=100*(sum(Av(end-I:end))/sum(Av));


%% ETAPA DE GENERALIZACAO  %%%
RESULTS=[];
EQMtest=0;
for tt=1:Ntest,
    % CAMADA OCULTA
    x=[+1; Xtest(tt,1:2)'];            % Constroi vetor de entrada com adicao da entrada x0=+1
    Ui = WW * x;              % Ativacao (net) dos neuronios da camada oculta
    Yi = 2./(1+exp(-Ui)) - 1; % Saida entre [-1,1] (funcao tangente hiperbolica)

    % CAMADA DE SAIDA
    Y=[+1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=+1
    Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
    Ok = Uk;              % Saida linear

    RESULTS=[RESULTS; Xtest(tt,3:4) Ok'];       % Armazena as saidas desejadas e a estimadas pela rede

    Ek = Xtest(tt,3:4)' - Ok;           % erro entre a saida desejada e a saida da rede
    EQMtest = EQMtest + 0.5*sum(Ek.^2);
end

EQMtest=EQMtest/Ntest

% Graficos
figure;
plot(1:Ne,EQMtrain,'linewidth',2); xlabel('Epocas'); ylabel('Erro Medio Quadratico');
grid, set(gca,"fontsize", 12)

figure;
plot(RESULTS(:,1),RESULTS(:,2),'ro','linewidth',2,'markersize',10)
hold on; plot(RESULTS(:,3),RESULTS(:,4),'b+','linewidth',2,'markersize',10);
xlabel('\theta_1'); ylabel('\theta_2');
legend('True coordinate','Estimated coordinate')
et(gca,"fontsize", 12)

