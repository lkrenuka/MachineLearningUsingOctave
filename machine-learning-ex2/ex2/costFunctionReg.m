function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
lt = size(X);
ln= lt(2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%[J1, grad1] = costFunction(theta, X(2:m,:), y(2:m,:));
[J1, grad1] = costFunction(theta, X, y);
theta_sqr = zeros(ln-1,1);


for i=2:ln,
  theta_sqr(i) = theta(i)^2;
end
J = J1 + (lambda/(2*m))* (sum(theta_sqr));

grad = grad1;
grad(2:ln) = grad1(2:ln) + ((lambda/m)*theta(2:ln));


% =============================================================

end
