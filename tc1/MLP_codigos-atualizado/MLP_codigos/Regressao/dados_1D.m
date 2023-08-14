% Codigo para gerar funcao ruidosa 1-D para comprovar overfitting
%
% Data: 17/10/2007
% Autor: Guilherme A. Barreto

clear all; close all;

%% Gera grafico da funcao sem ruido
x=0:0.01:1;
for t=1:length(x),
	y(t) = 0.5 + 0.4*sin(2*pi*x(t));
end
F=[x' y'];

%% Gera grafico da funcao com ruido
u=0:0.05:1;
%u=x;
for t=1:length(u),
	h(t) = 0.5 + 0.4*sin(2*pi*u(t)) + 0.05*randn + 0.05*t;
end

X=[u' h'];

% Graficos
figure(1); 
plot(x,y,'r-',u,h,'bo'); 
xlabel('x'); ylabel('y');

save -ascii noisy1d.dat X
save -ascii function1d.dat F  

