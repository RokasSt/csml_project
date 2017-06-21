function E=logEnergy(x,W,b)
%W=0.5*(W+W');
E=(x'*W'*x +x'*b);