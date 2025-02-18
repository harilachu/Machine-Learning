function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

[regJ, regGrad] = costFunction(theta, X, y);
n = size(X)(2);
ltheta = 0;

for j = 2 : n
 ltheta = ltheta + (theta(j)^2);
endfor

regJ = regJ + (lambda/(2*m)) * ltheta;

J = regJ;

grad(1) = regGrad(1);
grad(2:size(grad)(1),:) = regGrad(2:size(regGrad)(1),:)+((lambda/m).*theta(2:size(theta)(1),:));

% =============================================================

end
