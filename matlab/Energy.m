function E=Energy(x,W,b)
%W=0.5*(W+W');
E=exp(x'*W'*x +x'*b);