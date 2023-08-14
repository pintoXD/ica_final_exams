% Codigo para gerar funcao ruidosa 1-D para comprovar overfitting
%
% Data:07/08/2020
% Autor: Guilherme A. Barreto

clear all; close all;

%% Gera grafico da funcao sem ruido
x=0:0.01:1;
for t=1:length(x),
	y(t) = 0.5 + 1*x(t) + 0.4*sin(2*pi*x(t));
end
F=[x' y'];

%% Gera grafico da funcao com ruido
incr=0.08;
u=0:incr:1;
for t=1:length(u),
	h(t) = 0.5 + 1*u(t) + 0.4*sin(2*pi*u(t)) + 0.2*randn;
end

X=[u' h'];

% Graficos
figure; 
plot(x,y,'r-@',u,h,'bo','markersize',8,'linewidth',3); 
xlabel('x'); ylabel('y');

save -ascii noisy1d.dat X
save -ascii function1d.dat F  

