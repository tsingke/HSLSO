function h = normalize_history(history,maxiter)
%NORMALIZE_HISTORY Force convergence history to a maxiter-by-1 vector.
h = history(:);
if isempty(h), h=nan(maxiter,1); return; end
if numel(h)>=maxiter
    h=h(1:maxiter);
else
    h=[h; repmat(h(end),maxiter-numel(h),1)];
end
end
