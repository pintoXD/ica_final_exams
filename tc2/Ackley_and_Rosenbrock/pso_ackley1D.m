% Implementacao do algoritmo Particle Swarm Optimization (PSO)
% para encontrar o minimo da funcao Ackley 1D
%
%       f(x) = -20exp(-0.2|x|)-exp(cos(2*pi*x))+20+exp(1),  -5 < x < 5.
%
% Autor: Guilherme A. Barreto
% Data: 02/09/2021

clear; clc; close all;

%%% Gráfico da função de Ackley
limites=[-5 5];   % Limites do intervalo de busca para funcao de Ackely
x_ackley=limites(1):0.1:limites(2);  % Dominio de x discretizado em incrementos de 0.1
f_ackley=ackley1D(x_ackley);

%%% Parametros do PSO
Ng=50;       % Numero de iteracoes
Np=20;       % numero de particulas
c1=2.05;     % Constante de aceleracao 1
c2=c1;	     % Constante de aceleracao 2
w=0.6;       % Constante de inercia

v=zeros(Np,1);   % Velocidades iniciais das particulas
x=unifrnd(limites(1),limites(2),Np,1); 	% Posicoes iniciais das particulas
Fcand=ackley1D(x);  % Avalia solucoes iniciais

x_best=x;    % Faz posicoes iniciais = melhores posicoes
Fbest=Fcand;  % Faz aptidoes iniciais = melhores aptidoes

% Encontra melhor solucao no enxame
size(Fbest)
[Fmin I]=min(Fbest)
g_best=x_best(I)  % Melhor posição corrente no enxame

pause(30);
%%% Roda algoritmo por Ng iteracoes
for t=1:Ng,

    iteracao=t;
    printf("Iteration %d\n", iteracao);
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

    Vcog = rand(Np,1).*(x_best - x); %%componente cognitiva
    Vsoc = rand(Np,1).*(g_best - x); %%componente social

    v = w*v + c1*Vcog + c2*Vsoc;  % atualiza velocidades
    x = x + v; 	% atualiza posicoes (i.e. gera novas solucoes candidatas)

    % Verifica se existem solucoes candidatas fora dos limites
    I=find(x < limites(1)); x(I)=limites(1);   % fora do limite inferior
    I=find(x > limites(2)); x(I)=limites(2);   % fora do limite superior

    Fcand=ackley1D(x);  % Avalia solucoes candidatas

    %% Avalia solucoes candidatas (problema minimizacao)
    DF=Fcand-Fbest;
    I_better=find(DF<=0);  % Particulas que melhoraram posicoes
    I_worse =find(DF>0);  % Particluas que pioraram posicoes

    % Atualiza posicoes das particulas que melhoraram performance
    x_best(I_better)=x(I_better);
    Fbest(I_better)=Fcand(I_better);

    % Mantem posicoes das particulas que pioraram performance
    x_best(I_worse)=x_best(I_worse);
    Fbest(I_worse)=Fbest(I_worse);

    % Encontra melhor solucao corrente no enxame
    [F_gbest I]=min(Fbest);

    g_best=x_best(I);  % Melhor posição corrente no enxame

    aptidao(t)=F_gbest;

end

[x_best Fbest]
g_best, F_gbest

%%% Plota melhor solucao apos Ng iteracoes sobre a curva da funcao de ackley
plot(x_ackley,f_ackley,'k-','linewidth',3,x_best,Fbest,'ro','markersize',10,'linewidth',2);
%plot(x_best,Fbest,'ro','linewidth',3,'markersize',10);
hold off

figure; plot(aptidao,'linewidth',3);
xlabel('Iteration');
ylabel('Fitness');
