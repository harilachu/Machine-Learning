function [H,A] = ComputeHX(Theta1, Theta2, X)

H = zeros(size(X, 1), 1);

m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(size(X,1),1) X];
A = ComputeThetaX(X, Theta1');
%A = max(A', [], 2);
A = [ones(size(A,1),1) A];
HM = ComputeThetaX(A, Theta2');
H = HM;

end

function A = ComputeThetaX(X, theta)
  A = sigmoid(X * theta);
endfunction
