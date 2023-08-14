
// Display mode
//mode(0);

// Display warning for floating point exception
//ieee(1);

// Implementacao da rede MLP canonica (backpropagation com gradiente descendente)
// Usando as funcoes built-in (internas) do matlab
// 
// Exemplo para disciplina de ICA
// Autor: Guilherme de A. Barreto
// Date: 06/07/2006

// 
// X = Vetor de entrada
// d = saida desejada (escalar)
// W = Matriz de pesos Entrada -> Camada Oculta
// M = Matriz de Pesos Camada Oculta -> Camada saida
// eta = taxa de aprendizagem
// alfa = fator de momento

clear;
clc;

// Carrega DADOS
//=================
loadmatfile("derm_input.txt");
loadmatfile("derm_target.txt");

// ! L.23: mtlb(derm_input) can be replaced by derm_input() or derm_input whether derm_input is an M-file or not
dados = mtlb(derm_input);
// Vetores (padroes) de entrada
// ! L.24: mtlb(derm_target) can be replaced by derm_target() or derm_target whether derm_target is an M-file or not
alvos = mtlb(derm_target);
// Saidas desejadas correspondentes

clear("derm_input");
// Libera espaco em memoria
clear("derm_target");

// Embaralha vetores de entrada e saidas desejadas
[LinD,ColD] = size(mtlb_double(dados));

// Normaliza componetes para media zero e variancia unitaria
for i = 1:LinD
  mi = mean(dados(i,:),"m");
  // Media das linhas
  // !! L.35: Matlab function std not yet converted, original calling sequence used
  di = std(dados(i,:));
  // desvio-padrao das linhas 
  dados(i,:) = mtlb_s(mtlb_double(dados(i,:)),mi) ./mtlb_double(di);
end;
Dn = dados;

//I=randperm(ColD);
//Dn=Dn(:,I);
//alvos=alvos(:,I);   ;// Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

// Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn = 0.8;
// Porcentagem usada para treino
ptst = 1-ptrn;
// Porcentagem usada para teste

J = floor(ptrn*ColD);

// Vetores para treinamento e saidas desejadas correspondentes
P = Dn(:,1:J);
T1 = alvos(:,1:J);
[lP,cP] = size(P);
// Tamanho da matriz de vetores de treinamento

// Vetores para teste e saidas desejadas correspondentes
Q = Dn(:,J+1:$);
T2 = alvos(:,J+1:$);
[lQ,cQ] = size(Q);
// Tamanho da matriz de vetores de teste


// DEFINE ARQUITETURA DA REDE
//===========================
Ne = 100;
// No. de epocas de treinamento
Nr = 1;
// No. de rodadas de treinamento/teste
Nh = 10;
// No. de neuronios na camada oculta
No = 6;
// No. de neuronios na camada de saida

eta = 0.01;
// Passo de aprendizagem
mom = 0.75;
// Fator de momento

// Inicia matrizes de pesos
WW = 0.1*rand(Nh,lP+1);
// Pesos entrada -> camada oculta
WW_old = WW;
// Necessario para termo de momento

MM = 0.1*rand(No,Nh+1);
// Pesos camada oculta -> camada de saida
MM_old = MM;
// Necessario para termo de momento

//%% ETAPA DE TREINAMENTO
for t = 1:Ne

  Epoca = t,

  // !! L.81: Matlab function randperm not yet converted, original calling sequence used
  I = randperm(cP);
  P = P(:,I);
  T1 = T1(:,I);
  // Embaralha vetores de treinamento e saidas desejadas

  EQ = 0;
  for tt = 1:cP // Inicia LOOP de epocas de treinamento
   // CAMADA OCULTA
   X = [-1;P(:,tt)]; // Constroi vetor de entrada com adicao da entrada x0=-1
   Ui = WW*X; // Ativacao (net) dos neuronios da camada oculta
   Yi = 1 ./(1+exp(-Ui)); // Saida entre [0,1] (funcao logistica)
  
   // CAMADA DE SAIDA 
   Y = [-1;Yi]; // Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
   Uk = MM*Y; // Ativacao (net) dos neuronios da camada de saida
   Ok = 1 ./(1+exp(-Uk)); // Saida entre [0,1] (funcao logistica)
  
   // CALCULO DO ERRO 
   Ek = mtlb_double(T1(:,tt))-Ok; // erro entre a saida desejada e a saida da rede
   EQ = mtlb_a(EQ,0.5*sum(Ek .^2,"m")); // soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA
  
  
   //%% CALCULO DOS GRADIENTES LOCAIS
   Dk = Ok .*(1-Ok); // derivada da sigmoide logistica (camada de saida)
   DDk = Ek .*Dk; // gradiente local (camada de saida)
  
   Di = Yi .*(1-Yi); // derivada da sigmoide logistica (camada oculta)
   DDi = Di .*((MM(:,2:$))'*DDk); // gradiente local (camada oculta)
  
   // AJUSTE DOS PESOS - CAMADA DE SAIDA
   MM_aux = MM;
   MM = mtlb_a(mtlb_a(MM,(eta*DDk)*Y'),mom*mtlb_s(MM,MM_old));
   MM_old = MM_aux;
  
   // AJUSTE DOS PESOS - CAMADA OCULTA
   WW_aux = WW;
   WW = mtlb_a(mtlb_a(WW,(eta*DDi)*X'),mom*mtlb_s(WW,WW_old));
   WW_old = WW_aux;
  end;
  // Fim de uma epoca

  // MEDIA DO ERRO QUADRATICO P/ EPOCA
  EQM(1,t) = matrix(EQ/cP,1,-1);
end;
// Fim do loop de treinamento


// VERIFICACAO DE REDUNDANCIA COM A REDE JAH TREINADA
// USA DADOS DE TREINAMENTO, MAS NAO ALTERA OS PESOS
EQ1 = 0;
HID1 = [];
OUT1 = [];
for tt = 1:cP
  // CAMADA OCULTA
  X = [-1;P(:,tt)];
  // Constroi vetor de entrada com adicao da entrada x0=-1
  Ui = WW*X;
  // Ativacao (net) dos neuronios da camada oculta
  Yi = 1 ./(1+exp(-Ui));
  // Saida entre [0,1] (funcao logistica)
  HID1 = [HID1,Yi];
  // Armazena saida dos neuronios ocultos

  // CAMADA DE SAIDA 
  Y = [-1;Yi];
  // Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
  Uk = MM*Y;
  // Ativacao (net) dos neuronios da camada de saida
  Ok = 1 ./(1+exp(-Uk));
  // Saida entre [0,1] (funcao logistica)
  OUT1 = [OUT1,Ok];
  // Armazena saida dos neuronios de saida

  Ek = mtlb_double(T1(:,tt))-Ok;
  // erro entre a saida desejada e a saida da rede
  EQ1 = mtlb_a(EQ1,0.5*sum(Ek .^2,"m"));
  // soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

  Dk = Ok .*(1-Ok);
  // derivada da sigmoide logistica (neuronios de saida)
  DDk = Ek .*Dk;
  // gradiente local (neuronios de saida)

  // Gradiente local da camada oculta
  Di = Yi .*(1-Yi);
  // derivada da sigmoide logistica (neuronios ocultos)        
  DDi = Di .*((MM(:,2:$))'*DDk);
  // gradiente local (neuronios ocultos) 
end;

// MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TREINAMENTO)
EQM1 = EQ1/cP;

Ch = mtlb_cov(HID1');
// Matriz de covariancia das saidas dos neuronios da camada OCULTA
Av = spec(mtlb_double(Ch));
// Autovalores da matriz Ch
Rc = 1/mtlb_cond(Ch);
// Razao entre menor e maior autovalor da matriz Ch

//I=1; Plam=100*(sum(Av(end-I:end))/sum(Av));


//% ETAPA DE GENERALIZACAO  %%%
EQ2 = 0;
HID2 = [];
OUT2 = [];
for tt = 1:cQ
  // CAMADA OCULTA
  X = [-1;Q(:,tt)];
  // Constroi vetor de entrada com adicao da entrada x0=-1
  Ui = WW*X;
  // Ativacao (net) dos neuronios da camada oculta
  Yi = 1 ./(1+exp(-Ui));
  // Saida entre [0,1] (funcao logistica)
  HID2 = [HID2,Yi];
  // Armazena saida dos neuronios ocultos

  // CAMADA DE SAIDA 
  Y = [-1;Yi];
  // Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
  Uk = MM*Y;
  // Ativacao (net) dos neuronios da camada de saida
  Ok = 1 ./(1+exp(-Uk));
  // Saida entre [0,1] (funcao logistica)
  OUT2 = [OUT2,Ok];
  // Armazena saida da rede

  // Gradiente local da camada de saida
  Ek = mtlb_double(T2(:,tt))-Ok;
  // erro entre a saida desejada e a saida da rede
  Dk = Ok .*(1-Ok);
  // derivada da sigmoide logistica                
  DDk = Ek .*Dk;
  // gradiente local igual ao erro x derivada da funcao de ativacao

  // ERRO QUADRATICO GLOBAL (todos os neuronios) POR VETOR DE ENTRADA
  EQ2 = mtlb_a(EQ2,0.5*sum(Ek .^2,"m"));

  // Gradiente local da camada oculta
  Di = Yi .*(1-Yi);
  // derivada da sigmoide logistica                
  DDi = Di .*((MM(:,2:$))'*DDk);
end;

// MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TREINAMENTO)
EQM2 = EQ2/cQ;


// CALCULA TAXA DE ACERTO
count_OK = 0;
// Contador de acertos
for t = 1:cQ
  [T2max,iT2max] = mtlb_max(mtlb_double(T2(:,t)),"m");
  // Indice da saida desejada de maior valor
  [OUT2_max,iOUT2_max] = mtlb_max(OUT2(:,t),"m");
  // Indice do neuronio cuja saida eh a maior
  if mtlb_logic(iT2max,"==",iOUT2_max) then // Conta acerto se os dois indices coincidem 
   count_OK = count_OK+1;
  end;
end;

// Taxa de acerto global
Tx_OK = 100*(count_OK/cQ)
