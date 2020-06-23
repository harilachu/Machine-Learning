function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    finaltheta = [];
    for i=1:length(theta)
      finaltheta = [finaltheta; (theta(i) - alpha * ComputeD(X, y, theta, i, m))]
    endfor
    %a = theta(1) - alpha * ComputeD(X, y, theta, 1, m);
    %b = theta(2) - alpha * ComputeD(X, y, theta, 2, m);
    
    theta = finaltheta;
    %theta(1) = a;
    %theta(2) = b;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end

function D = ComputeD(X, y, theta, j, m)
  error = 0;
  for i = 1:m;
      h = computeHX(X(i,:), theta);
      error =  error + ((h - y(i)) * X(i, j));
  end
  D = (1/m) * error;
end

function h = computeHX(setX1, theta)
  setX1 = setX1';
  h = theta' * setX1; 
endfunction