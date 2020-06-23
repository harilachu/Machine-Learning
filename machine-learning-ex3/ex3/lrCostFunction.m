function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

JTemp = ComputeCost(X, y, theta);
J = JTemp + regTheta(lambda, m, theta);


% =============================================================

grad = (1/m) .* (ComputeGrad(X, y, theta) + regGrad(lambda, m, theta));
grad = grad(:);

end

function G = ComputeGrad(X, y, theta)
  Z = computeThetaX(X, theta);
  grad = X' * (sigmoid(Z) - y);
  G = grad;
endfunction

function rGrad = regGrad(lambda, m, theta)
  temp = theta;
  temp(1) = 0;
  rGrad = lambda.*temp;
endfunction

function D = ComputeCost(X, y, theta)
  m = length(y);
  Z = computeThetaX(X, theta);
  cost = 0;
  cost = cost + ((y' * (log(sigmoid(Z)))) + (1.-y') * (log(1.-sigmoid(Z))));
  D = -(1/m)*cost;
endfunction

function rTheta = regTheta(lambda, m, theta)
  temp = theta;
  temp(1) = 0; % for regularizing make theta0 as 0.
  rTheta = (lambda/(2*m)) * sum(temp.^2);
endfunction

function Z = computeThetaX(X, theta)
  Z = X * theta; 
endfunction