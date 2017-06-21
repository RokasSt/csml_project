clear all; close all;
N=10;
d=15;
D=d*d;

% make some fake training data:
x=1.0*(randn(D,N)>0);

% try to learn this:
W=0.00000001*randn(D,D); W=0.5*(W+W');
b=0*randn(D,1);
Winit=W;

S=100;
% uniform importance distribution:
kappa=1/(S*0.5^D);

%Sample= 1.0*(randn(D,S)>0);

Nloops=1500;
for loop=1:Nloops
epsilon=(1/(1+loop/100))*0.01/N;
    loop
    gradW=zeros(D,D);
    gradb=zeros(D,1);
    
    tildeZ = 0;
    for m=1:N
        tildeZ = tildeZ + Energy(x(:,m),W,b);
    end
    Sample= 1.0*(randn(D,S)>0);
    for s=1:S
        tildeZ = tildeZ + kappa*Energy(Sample(:,s),W,b);
    end
    
    for m=1:N
        ptilde(m) = Energy(x(:,m),W,b)/tildeZ;
    end
    
    for s=1:S
        ptilde(N+s) = kappa*Energy(Sample(:,s),W,b)/tildeZ;
    end
    
    
    for n=1:N
        gradW=gradW+(1-N*ptilde(n)).*(x(:,n)*x(:,n)');
        gradb=gradb+(1-N*ptilde(n)).*x(:,n);
    end
    for s=1:S
        gradW=gradW - N*ptilde(N+s).*(Sample(:,s)*Sample(:,s)');
        gradb=gradb - N*ptilde(N+s)*Sample(:,s);
    end
    
    W=W+epsilon*gradW;
    b=b+epsilon*gradb;
    plot(ptilde(1:N)); drawnow
end


