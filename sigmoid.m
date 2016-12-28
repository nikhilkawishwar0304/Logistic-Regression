function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% Storing the exponent term in a variables
expTerm = exp(-z);

% Preparing the denominator term 1 + exp(-z)
denomTerm = 1 + expTerm;

% Element wise division of denominator to get 1 / 1 + exp(-z)
g = 1./denomTerm;

end
