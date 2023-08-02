function F=ackley2D(x,y)
% 
% Funcao objetivo Ackley 2D (2 variaveis: x e y)
%
% Entradas:
%	(x,y): coordenada a ser avaliada
%       funcao: f(x,y) = -20*exp(-0.2*sqrt(0.5*(x^2+y^2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+20+exp(1);
%
% Saida:
%	F: valor da funcao-objetivo em (x,y)
%
% Data: 03/12/2017
% Autor: Guilherme Barreto

F=-20*exp(  -0.2*sqrt( 0.5*(x.^2+y.^2) )  ) - exp(   0.5*( cos(2*pi.*x) + cos(2*pi.*y) )    )+20+exp(1);

