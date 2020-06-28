function [J,Y] = costFunc(y, H)
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
K = size(H,2);
Y = ComputeY(y, K);

cost = 0;

for i = 1:m
  for k = 1:K
    cost = cost + (Y(i,k)* log(H(i,k))) + ((1-Y(i,k)) * log(1 - H(i,k)));
  endfor
endfor

J = -(1/m)*cost;

end

function Y = ComputeY(y, K)
  m = size(y,1);
  Y = zeros(m, K);
  for i = 1: m
    YVec = zeros(K,1);
    %construct y[0,0,..1] vector
    for k = 1 : K
      if y(i) == k
        YVec(k) = 1;
        break;
      endif     
    endfor
    
    Y(i, :) = YVec';
  endfor
endfunction
