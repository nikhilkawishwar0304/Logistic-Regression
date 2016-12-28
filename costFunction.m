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

% Gradient can be calculated using formula  1/m sum(h(x) - y) * x 
h = sigmoid(X * theta);
diffTerm = h - y;
grad = diffTerm' * X;

% Calculating Cost Fuction = -1/m sum(ylog(h(x)) + (1-y)log(1-h(x)))

% Calculating the first term ylog(h(x))
logTerm = log(h);
FirstTerm = y' * logTerm;

% Calculating the second term (1-y)log(1-h(x))
negY = 1 - y;
negH = 1 - h;
logTerm = log(negH);
SecTerm = negY' * logTerm;

% Collating first term and second term
Res = FirstTerm + SecTerm;

% Average over the number of training examples
J = Res/m;
J = -1 * J;

end
