function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate cost function


cost = 0;
for i = 1:m
  hxi =  X(i, :) * theta;
  cost = cost + (hxi - y(i))^2;
endfor

cost = cost/(2*m);
thetaj = theta(2:end, :);
cost = cost + (lambda/(2*m)) * (sum(thetaj(:).^2));
J = cost;


temptheta = theta;
temptheta(1) = 0;

for j = 1:size(theta,1)
  for i = 1:m
    hxi =  X(i, :) * theta;
    temp = (hxi - y(i))*X(i,j);
    grad(j) = grad(j) + temp;  
  endfor
  grad(j) = (grad(j)/m) + (lambda/m)*temptheta(j);
endfor

% =========================================================================

grad = grad(:);

end
