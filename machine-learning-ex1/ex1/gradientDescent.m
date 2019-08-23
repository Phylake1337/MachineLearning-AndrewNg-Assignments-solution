function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

H = X*theta; 
SE = (H - y);
z1= sum (SE);
s2 = X(:,2);
z2= sum(SE.*s2);
th1 = theta(1,1)- ((alpha/m)*z1);
th2 = theta(2,1)- ((alpha/m)*z2);
theta = [th1;th2]
   
J_history(iter) = computeCost(X, y, theta);
   
end
