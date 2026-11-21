function out = msa_seed_context(action,value)
%MSA_SEED_CONTEXT Process-local seed-profile context used by Viterbi.
% This avoids changing every legacy optimizer's public function signature.
persistent ctx
if nargin<1, action='get'; end
switch lower(action)
    case 'set'
        if nargin<2, error('msa_seed_context(''set'',value) requires value.'); end
        ctx=value; out=ctx;
    case 'get'
        out=ctx;
    case 'clear'
        ctx=[]; out=[];
    otherwise
        error('Unknown msa_seed_context action: %s',action);
end
end
