% Implementacao da rede MLP canonica (backpropagation com gradiente descendente)
% Usando as funcoes built-in (internas) do matlab
%
% Objetivo: demonstrar a problematica do OVERFITTING a dados ruidosos
%
% Nota: o mapeamento a ser aprendido eh estatico e unidimensional, logo
%       a rede neural terá uma entrada e uma saida apenas. O unico parametro
%       variavel eh o numero de neuronios ocultos.
%
% Exemplo para disciplina de ICA
% Autor: Guilherme de A. Barreto
% Date: 16/05/2023
%
%
% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc; %close all;

% Carrega DADOS
%=================
D=load('noisy1d.dat');        % observacoes ruidosas da funcao
Real=load('function1d.dat');  % funcao deterministica (i.e. sem ruido)

% pares (entrada, saida) = (x,d)
x=D(:,1); d=D(:,2);

% Normaliza dados ruidosos
x_min=min(x); x_max=max(x);  % limites necessarios para normalizacao
d_min=min(d); d_max=max(d); % limites necessarios para normalizacao
x=2*(x - x_min)/(x_max-x_min)-1; % normaliza entradas entre -1 e 1
d=2*(d - d_min)/(d_max-d_min)-1;  % normaliza saidas entre -1 e 1

% Normaliza dados da funcao real (deterministica)
Rx=2*(Real(:,1) - x_min)/(x_max-x_min)-1; % normaliza entradas entre -1 e 1
Ry=2*(Real(:,2) - d_min)/(d_max-d_min)-1; % normaliza entradas entre -1 e 1

[LinD ColD]=size(D);

% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 500; % No. de epocas de treinamento
Ni = 1;   % No. de unidades de entrada
Nh = 50;   % No. de neuronios na camada oculta
No = 1;   % No. de neuronios na camada de saida

eta=0.1;   % Passo de aprendizagem

% Inicia matrizes de pesos
WW=rand(Nh,Ni+1);   % Pesos entrada -> camada oculta
MM=rand(No,Nh+1);   % Pesos camada oculta -> camada de saida

WW=2*WW-1;
MM=2*MM-1;

%%% ETAPA DE TREINAMENTO
for t=1:Ne,

    Epoca=t

    % Embaralha vetores de treinamento e saidas desejadas
    I=randperm(LinD);
    x=x(I); d=d(I);

    EQ=0;
    for tt=1:LinD,   % Inicia LOOP de epocas de treinamento
        % CAMADA OCULTA
        X=[+1; x(tt)];        % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta

        aux=exp(-Ui);
        %Zi = 1./(1+exp(-Ui)); % Saida entre [0,1] (sigmoide logistica)
        Zi = (1-aux)./(1+aux); % Saida entre [-1,1] (tangente hiperbolica)

        % CAMADA DE SAIDA
        Z=[+1; Zi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Ak = MM * Z;          % Ativacao (net) dos neuronios da camada de saida

        aux=exp(-Ak);
        %Yk = Ak;	      % Saida linear = funcao identidade
        %Yk = 1./(1+aux); % Saida entre [0,1] (sigmoide logistica)
        Yk = (1-aux)./(1+aux); % Saida entre [-1,1] (tangente hiperbolica)

        % CALCULO DO ERRO
        Ek = d(tt) - Yk;           % erro entre a saida desejada e a saida da rede
        EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

        %%% CALCULO DOS GRADIENTES LOCAIS
        %Dk = 1;		    % derivada da funcao identidade (camada de saida)
        %Dk = Yk.*(1 - Yk);  % derivada da sigmoide logistica (camada de saida)
        Dk = 0.5*(1 - Yk.*Yk);  % derivada da tangente hiperbolica (camada de saida)
        DDk = Ek.*Dk;       % gradiente local (camada de saida)

        %Di = Zi.*(1 - Zi); % derivada da sigmoide logistica (camada oculta)
        Di = 0.5*(1 - Zi.*Zi); % derivada da tangente hiperbolica (camada oculta)
        DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)


        MM = MM + eta*(DDk)*Z'; % AJUSTE DOS PESOS - CAMADA DE SAIDA
        WW = WW + eta*(DDi)*X';  % AJUSTE DOS PESOS - CAMADA OCULTA

    end   % Fim de uma epoca de treinamento

    EQM(t)=EQ/LinD; % MEDIA DO ERRO QUADRATICO P/ EPOCA

end   % Fim do loop de treinamento



%% ETAPA DE GENERALIZACAO  %%%
xmin=min(x);
xmax=max(x);
xn=xmin:0.01:xmax;
LinD=length(xn);
EQ=0;
for tt=1:LinD,   % Inicia LOOP de epocas de treinamento
        % CAMADA OCULTA
        X=[+1; xn(tt)];       % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        %Zi = 1./(1+exp(-Ui)); % Saida entre [0,1] (sigmoide logistica)
        aux = exp(-Ui);
        Zi = (1-aux)./(1+aux); % Saida entre [-1,1] (tangente hiperbolica)

        % CAMADA DE SAIDA
        Z=[+1; Zi];              % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Ak = MM * Z;             % Ativacao (net) dos neuronios da camada de saida
	      %Yk(tt) = Ak;	         % Saida linear = funcao identidade
        %Yk(tt) = 1/(1+exp(-Ak)); % Saida entre [0,1] (sigmoide logistica)
        aux = exp(-Ak);
        Yk(tt) = (1-aux)./(1+aux); % Saida entre [-1,1] (tangente hieprbolica)
end

% Plota curva de aprendizagem
figure; plot(EQM,'linewidth',3);
xlabel('Epocas'); ylabel('Erro Medio Quadratico');
grid, set(gca,"fontsize", 12)

% Plota dados reais versus dados preditos
figure; plot(x,d,'ro','markersize',8,'linewidth',3,xn,Yk,'b-','markersize',8,'linewidth',3);
hold on; plot(Rx,Ry,'g*','markersize',8,'linewidth',3); hold off

