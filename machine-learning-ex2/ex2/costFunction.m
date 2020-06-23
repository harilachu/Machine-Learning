function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = ComputeCost(X, y, theta);

n = size(X)(2);
for c = 1:n
  grad(c) = (1/m)*GD(X, y, theta, c);
endfor


% =============================================================

end

function G = GD(X, y, theta, j)
  m = length(y);
  G = 0;
  for i = 1:m
    z = computeThetaX(X(i,:), theta);
    G = G + (sigmoid(z) - y(i))*X(i,j)
  endfor
endfunction

function D = ComputeCost(X, y, theta)
  m = length(y);
  error = 0;
  for i = 1:m;
      z = computeThetaX(X(i,:), theta);
      error =  error + (y(i)*log(sigmoid(z))) + ((1-y(i))*log(1-sigmoid(z))); 
  end
  D = -(1/m) * error;
endfunction

function z = computeThetaX(setX1, theta)
  setX1 = setX1';
  z = theta' * setX1; 
endfunction
