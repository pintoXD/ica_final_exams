function F=rosenbrock2D(x,y)
% 
% Funcao objetivo Rosenbrock 2D (2 variaveis: x e y)
%
% Entradas:
%	(x,y): coordenada a ser avaliada
%       funcao: f(x,y) = (1-x)^2+100(y-x^2)^2; 
%
% Saida:
%	F: valor da funcao-objetivo em (x,y)
%
% Data: 03/12/2017
% Autor: Guilherme Barreto

F=(1-x).^2 + 100*(y-x.^2).^2; 

