function [varargout] = testGp(hyp, mean, cov, lik, x, post, xs)
%xs=gpuArray(xs); %this
alpha = post.alpha; L = post.L; sW = post.sW;invLt = post.invLt;
nz = true(size(alpha,1),1);
kss = feval(cov{:}, hyp.cov, xs, 'diag');              % self-variance
Ks = feval(cov{:}, hyp.cov, x(nz,:), xs);        % avoid computation
Fmu = 0 + Ks'*full(alpha(nz,:));        % conditional mean fs|f                            % predictive means
V  = invLt*((sW).*Ks);
%V=L'\((sW).*Ks);
Fs2 = kss - sum(V.*V,1)';                       % predictive variances
%Fmu=gather(Fmu); % this
%Fs2=gather(Fs2);% this
varargout = {Fmu, Fs2};        % assign output arguments



% %new
% xs=gpuArray(xs);
% alpha = post.alpha; L = post.L; sW = post.sW;
% nz = true(size(alpha,1),1);
% Ks = feval(cov{:}, hyp.cov, x(nz,:), xs);        % avoid computation
% Fmu = 0 + Ks'*full(alpha(nz,:));        % conditional mean fs|f
%  Fmu=gather(Fmu);
% varargout = {Fmu, 0};        % assign output arguments