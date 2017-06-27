clear all; close all;
N=10;
d=28;
D=d*d;

% make some fake training data:
% x=1.0*(randn(D,N)>0);

% MNIST data:

path =['../theano/logs_CSS/LR0.001M0.0BS10NS100RS0DATA0MF0_1127AM_June27_2017/'...
    'TRAIN_IMAGES.dat'];

train_images = load(path);

x= train_images';

[D, N] = size(x);

disp(D)
disp(N)

% try to learn this:
W = 0.00000001*randn(D,D); W=0.5*(W+W');
W = W - diag(diag(W));
b = 0*randn(D,1);
Winit=W;

S=100;
% uniform importance distribution:
%kappa=1/(S*0.5^D);
logkappa = -(log(S)+D*log(0.5));

%Sample= 1.0*(randn(D,S)>0);

Nloops=1500;
for loop=1:Nloops
epsilon=(1/(1+loop/100))*0.001/N;
%epsilon=(1/(1+loop/100))*0.001/N; 
    disp(loop);
    gradW=zeros(D,D);
    gradb=zeros(D,1);
    
   logE=zeros(N+S,1);
    for m=1:N
        logE(m)= logEnergy(x(:,m),W,b);
    end
    Sample= 1.0*(randn(D,S)>0);
    for s=1:S
        logE(N+s)=logkappa + logEnergy(Sample(:,s),W,b);
    end
    
   logE=logE-max(logE);

   ptilde=exp(logE);
   ptilde=ptilde./sum(ptilde);
    
    for n=1:N
        gradW=gradW+(1-N*ptilde(n)).*(x(:,n)*x(:,n)');
        gradb=gradb+(1-N*ptilde(n)).*x(:,n);
    end
    for s=1:S
        gradW=gradW - N*ptilde(N+s).*(Sample(:,s)*Sample(:,s)');
        gradb=gradb - N*ptilde(N+s)*Sample(:,s);
    end
    
    disp(sum(ptilde(1:N)))
    gradW = gradW - diag(diag(gradW));
    W=W+epsilon*gradW; 
    b=b+epsilon*gradb;
    plot(ptilde(1:N)); drawnow
end



