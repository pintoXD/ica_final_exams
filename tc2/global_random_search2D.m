% Implementacao do algoritmo de busca aleatoria GLOBAL (GRS, global random search)
% para encontrar o maximo de uma funcao de 2 variaveis
%
%     funcao 1:  f(x,y)=(x-20)^2 + (y-20)^2 + 50;  0 <= x,y <= 40
%
%     funcao 2:  f(x,y)=60x+100y-1.5*x^2-1.5*y^2-xy, 0 <= x,y <=60;
%
% Autor: Guilherme A. Barreto
% Data: 22/08/2020

clear; clc; close all;

%%% Parametros do AG
Ng=500;   % Numero de iteracoes

funcao=2;  % Escolhe funcao-objetivo

% Escolhe limites do dominio de x
if funcao==1,
	limites=[0 40];   % Limites do intervalo para funcao: F=x^2
else    
	limites=[0 60];   % Limites do intervalo de busca para funcao: F=x*sin(10*pi*x)+1;
end

% Gera solucao inicial dentro do dominio de Y
x_best=unifrnd(limites(1),limites(2));  % na 1a. iteracao a melhor solucao para x coincide com a solucao candidata
y_best=unifrnd(limites(1),limites(2));  % na 1a. iteracao a melhor solucao para y coincide com a solucao candidata

Fbest=func_objetivo2D(x_best,y_best,funcao);  % Avalia solucao inicial

figure(1); plot(x_best,y_best,'r*','markersize',12); hold on;
plot(x_best,y_best,'ro','markersize',8,'linewidth',4);
title('espaco de busca');
axis([0 60 0 60]); grid

%%% Roda GRS por Ng iteracoes
for t=2:Ng,
    iteracao=t
    
    x_cand(t)=unifrnd(limites(1),limites(2));      % Gera solucao candidata para X
    y_cand(t)=unifrnd(limites(1),limites(2));	% Gera solucao candidata para Y

    Fcand=func_objetivo2D(x_cand(t),y_cand(t),funcao);  % Avalia solucao candidata
    
    if Fcand>Fbest,
        x_best=x_cand(t);    % Se [x_cand   y_cand] produzem melhor resultado que [x_best  y_best], viram "melhor solucao ate o momento"
	      y_best=y_cand(t);
        Fbest=Fcand;
    end
    
    %plot(x_cand,y_cand,'b*','markersize',12,'linewidth',4);
    %plot(x_best,y_best,'ro','markersize',8,'linewidth',4);
    
    %pause(0.05)

    aptidao(t)=Fbest;

end

hold off

[x_best  y_best], Fbest

figure(2); plot(aptidao,'linewidth',4);
xlabel('Iteracao'); ylabel('y=f(x1,x2)');
set(gca, "fontsize", 14)


