% Implementacao do algoritmo Particle Swarm Optimization (PSO)
% para encontrar o minimo da funcao Ackley 1D
%
%       f(x) = -20exp(-0.2|x|)-exp(cos(2*pi*x))+20+exp(1),  -5 < x < 5.
%
% Autor: Guilherme A. Barreto
% Data: 02/09/2021

clear; clc; close all;
pkg load statistics
load aerogerador.dat
v_aero=aerogerador(:,1);
P=aerogerador(:,2);

%%% Gráfico da função de Ackley
limites=[-6 10];   % Limites do intervalo de busca para funcao de Ackely
%x_ackley=limites(1):0.1:limites(2);  % Dominio de x discretizado em incrementos de 0.1
k = 5;


%f_ackley=ackley1D(x_ackley);

%%% Parametros do PSO
Ng=200;       % Numero de iteracoes
Np=80;       % numero de particulas
c1=2.05;     % Constante de aceleracao 1
c2=c1;	     % Constante de aceleracao 2
w=0.6;       % Constante de inercia

v=zeros(Np,1);   % Velocidades iniciais das particulas
%x=unifrnd(limites(1),limites(2),Np,1); 	% Posicoes iniciais das particulas

%b_aerogerador é a matriz de coeficientes do polinomio que tenta aprovciar
%a potência do aerogerador. A mtrix é do tamanho Np x K+1 onde Np é o número
%de partículas e k+1 é o grau do polinômio mais 1. Esse mais 1 corresponde
%ao valor independente do polinômio. Por exemplo, se P = ax + b. o +1 representa
%os valores que b pode assumir.
b_aerogerador=unifrnd(limites(1), limites(2), Np, k+1);
SEQ  = pso_sea(b_aerogerador, v_aero, P);

Fcand = SEQ;

b_best=b_aerogerador;    % Faz posicoes iniciais = melhores posicoes
Fbest=Fcand;  % Faz aptidoes iniciais = melhores aptidoes

% Encontra melhor solucao no enxame
[Fmin I]=min(Fbest);
g_best=b_best(I, :);  % Melhor posição corrente no enxame

%%% Roda algoritmo por Ng iteracoes
for t=1:Ng,

    iteracao=t;
    %printf("Iteration %d\n", iteracao);
    %%% Plota melhores solucoes ate o momento sobre a curva da funcao de ackley
    %{
    if ~mod(iteracao,5),
       figure;
       plot(x_ackley,f_ackley,'k-','linewidth',3,x_best,Fbest,'ro','markersize',10,'linewidth',2);
       xlabel('iteracao');
       ylabel('Funcao objetivo');
       pause
    end
    %}
    %% Atualiza velocidades das particulas

    Vcog = rand(Np,1).*(b_best - b_aerogerador); %%componente cognitiva
    Vsoc = rand(Np,1).*(g_best - b_aerogerador); %%componente social

    v = w*v + c1*Vcog + c2*Vsoc;  % atualiza velocidades
    b_aerogerador = b_aerogerador + v; 	% atualiza posicoes (i.e. gera novas solucoes candidatas)

    % Verifica se existem solucoes candidatas fora dos limites
    I=find(b_aerogerador < limites(1)); b_aerogerador(I)=limites(1);   % fora do limite inferior
    I=find(b_aerogerador > limites(2)); b_aerogerador(I)=limites(2);   % fora do limite superior

    %Fcand=ackley1D(x);  % Avalia solucoes candidatas
    SEQ = pso_sea(b_aerogerador, v_aero, P);
    Fcand = SEQ;
    %% Avalia solucoes candidatas (problema minimizacao)
    DF=Fcand-Fbest;
    I_better=find(DF<=0);  % Particulas que melhoraram posicoes
    I_worse =find(DF>0);  % Particluas que pioraram posicoes

    % Atualiza posicoes das particulas que melhoraram performance
    b_best(I_better)=b_aerogerador(I_better);
    Fbest(I_better)=Fcand(I_better);

    % Mantem posicoes das particulas que pioraram performance
    b_best(I_worse)=b_best(I_worse);
    Fbest(I_worse)=Fbest(I_worse);

    % Encontra melhor solucao corrente no enxame
    [F_gbest I]=min(Fbest);

    g_best=b_best(I, :);  % Melhor posição corrente no enxame

    aptidao(t)=F_gbest;

end

%[b_best Fbest]
%b_best
%Fbest
g_best, F_gbest

%%% Plota melhor solucao apos Ng iteracoes sobre a curva da funcao de ackley
%plot(x_ackley,f_ackley,'k-','linewidth',3,x_best,Fbest,'ro','markersize',10,'linewidth',2);
%plot(x_best,Fbest,'ro','linewidth',3,'markersize',10);
plot(v_aero, polyval(g_best, v_aero));
hold off

figure; plot(aptidao,'linewidth',3);
xlabel('Iteration');
ylabel('Fitness');
