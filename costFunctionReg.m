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

% Gradient can be calculated using formula  1/m sum(h(x) - y) * x 
h = sigmoid(X * theta);
diffTerm = h - y;
grad = diffTerm' * X;
grad = grad / m;

% Regularized gradient will have another term lambda/ m * theta
for i = 2:size(grad,2)
  grad(1,i) = grad(1,i) + (lambda/m * theta(i,1));

% Calculating Cost Fuction = -1/m sum(ylog(h(x)) + (1-y)log(1-h(x)))

% Calculating the first term ylog(h(x))
logTerm = log(h);
FirstTerm = y' * logTerm;

% Calculating the second term (1-y)log(1-h(x))
negY = 1 - y;
negH = 1 - h;
logTerm = log(negH);
SecTerm = negY' * logTerm;

% Calculating the regularization term 

for i = 2 : size(theta,1)
  thetasqr(i,1) = theta(i,1)^2;
  end
ThirdTerm = thetasqr * (lambda /(2*m));
ThirdTerm = sum(ThirdTerm);

% Collating first term and second term
Res = FirstTerm + SecTerm;

% Average over the number of training examples
J = Res/m;
J = -1 * J;
J = J + ThirdTerm;

end
