load aerogerador.dat
v=aerogerador(:,1);
P=aerogerador(:,2);

k=1;

figure; plot(v, P, 'bo');
title(sprintf("K=%d", k));
xlabel('Velocidade do vento [m/s]');
ylabel('Potencia gerada [kwatts]');

disp("Waiting...");
pause(2);



B = polyfit(v, P, k);
y = P;
ypred=polyval(B,v);
erro=y-ypred;
SEQ = sum(erro.^2);

ymed=mean(y);
Syy=sum((y-ymed).^2);

R2 = 1 - (SEQ/Syy);
disp("R2 is: "); disp(R2);

SEQdiv = SEQ/(length(v) - (k+1));
Syydiv = Syy/(length(v) - 1);
R2adj = 1 - (SEQdiv/Syydiv);
disp("R2adj is: "); disp(R2adj);

AIC = (length(v)*log(SEQ))+2*k;
disp("AIC is: "); disp(AIC);

vv=min(v):0.1:max(v);
vv=vv';

ypred2=polyval(B,vv);

hold on; plot(vv, ypred2, 'r-'); hold off;




