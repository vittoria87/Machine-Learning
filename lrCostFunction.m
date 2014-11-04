function [J, grad] = lrCostFunction(theta, X, y, lambda)

%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
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

J = (1/m) * (-sum(y .* log(sigmoid(X * theta))) - sum((1 - y) .* log(1 - sigmoid(X * theta)))) + (lambda/(2*m)) * sum([0 ; theta(2:end)] .^ 2);
grad = (1/m) * X' * (sigmoid(X * theta) - y) + (lambda/m) * [0 ; theta(2:end)] ; 


% =============================================================

grad = grad(:);
end
