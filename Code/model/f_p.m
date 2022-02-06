function dydt = f_p(t,y,Trans,dim,par)
%the Aliev Panfilov EP model

dydt = zeros(2*dim,1);
u = y(1:dim);
v = y(dim+1:dim*2);
k = 8;
e = 0.01;
%a = 0.15;a
%  e0 = 0.22;
%  w1 = 0.2;
%  w2 = 0.3;

dydt(1:dim) = Trans*u + k*u.*(1-u).*(u-par)-u.*v;
dydt(dim+1:dim*2) = -e*(k*u.*(u-par - 1) + v);

% dydt(1:dim) = Trans*u - k*u.*(u-par).*(u-1) - u.*v;
% dydt(dim+1:dim*2) = (e0 + w1*v./(u + w2)).*(-v - k*u.*(u-par-1));
% also no sign of "limit-cycle"
% standard a = 0.08;

