function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

rowsize = size(z)(1,1);
columnsize = size(z)(1,2);

for i= 1: rowsize
  for j=1:columnsize
    g(i,j) = 1 / (1+ e^-z(i,j));  
  endfor
endfor



% =============================================================

end


