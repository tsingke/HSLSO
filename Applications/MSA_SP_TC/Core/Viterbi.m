function aligned = Viterbi(sequence, a, lengthdata, A, B, K)
%VITERBI Standard global profile-HMM Viterbi decoder (v4).
%
% States
% ------
%   M_k : emits one nucleotide, advances sequence and profile
%   I_k : emits one nucleotide, advances sequence only, k=0..K
%   D_k : silent, advances profile only
%
% Topology
% --------
%   Begin -> M1 / I0 / D1
%   I0    -> M1 / I0 / D1
%   for k=1..K-1, each of M_k/I_k/D_k -> M_{k+1}/I_k/D_{k+1}
%   at k=K, each of M_K/I_K/D_K -> I_K / End
%
% Parameter count remains exactly 17*K+7:
%   transitions: 3 + 3 + 9*(K-1) + 6 = 9*K+3
%   emissions  : 4*K + 4*(K+1)       = 8*K+4
%
% A and B retain the legacy reshape sizes only to preserve every optimizer's
% public interface.  Their flattened values are interpreted according to
% the standard topology above.
%
% Candidate genes are converted to probabilities by perturbing a
% reference-free Profile-HMM prior estimated from a coarse pre-alignment
% built only from the input sequences. The benchmark reference is never used.  No BAliBASE reference is used in decoding or optimization.

if nargin ~= 6
    error('Viterbi expects sequence,a,lengthdata,A,B,K.');
end
if numel(sequence) ~= lengthdata || numel(a) ~= lengthdata
    error('lengthdata does not match sequence/a length.');
end
if size(A,1) ~= 3*K+1 || size(A,2) ~= 3
    error('A must have size (3*K+1)-by-3.');
end
if size(B,1) ~= 2*K+1 || size(B,2) ~= 4
    error('B must have size (2*K+1)-by-4.');
end

[p, alphabet] = build_parameters(sequence,A(:),B(:),K);

matchChar = repmat('-',lengthdata,K);
insertions = cell(lengthdata,K+1);
for sidx=1:lengthdata
    for z=1:K+1, insertions{sidx,z}=''; end
end

for sidx = 1:lengthdata
    obs = upper(sequence(sidx).Sequence);
    if length(obs) ~= a(sidx)
        error('Sequence %d length mismatch.',sidx);
    end
    code = encode_dna(obs,alphabet);
    [mchar,ins] = decode_one(obs,code,p,K);
    matchChar(sidx,:) = mchar;
    insertions(sidx,:) = ins;
end

% Merge sequence-to-profile paths into one rectangular MSA.  Match states
% define homologous consensus columns.  Residues emitted by the same I_k
% state are left-justified inside that insertion block; remaining positions
% are padded with gaps.  Insert states themselves are non-consensus states.
maxIns = zeros(1,K+1);
for z=1:K+1
    for sidx=1:lengthdata
        maxIns(z)=max(maxIns(z),length(insertions{sidx,z}));
    end
end

W = K + sum(maxIns);
aligned = repmat('-',lengthdata,W);
col=1;
if maxIns(1)>0
    for sidx=1:lengthdata
        str=insertions{sidx,1};
        if ~isempty(str)
            aligned(sidx,col:col+length(str)-1)=str;
        end
    end
    col=col+maxIns(1);
end
for k=1:K
    aligned(:,col)=matchChar(:,k); col=col+1;
    if maxIns(k+1)>0
        for sidx=1:lengthdata
            str=insertions{sidx,k+1};
            if ~isempty(str)
                aligned(sidx,col:col+length(str)-1)=str;
            end
        end
        col=col+maxIns(k+1);
    end
end

% Remove all-gap columns only.  This cannot alter residue-residue pairing.
aligned = aligned(:,any(aligned~='-',1));

% Strong invariant: decoding may add gaps but may never add, delete, or
% reorder nucleotides.
for sidx=1:lengthdata
    ungapped = aligned(sidx,aligned(sidx,:)~='-');
    if ~strcmp(upper(ungapped),upper(sequence(sidx).Sequence))
        error('Decoder residue-conservation invariant failed for sequence %d.',sidx);
    end
end
end

function [mchar,ins] = decode_one(obs,code,p,K)
n=length(obs);
negInf=-inf;

% Trace predecessor codes: 1=M, 2=I, 3=D, 4=Begin, 5=I0.
psiM=zeros(n,K,'uint8');
psiI=zeros(n,K+1,'uint8'); % col 1 is I0, col k+1 is I_k
psiD=zeros(n+1,K,'uint8'); % row i+1 is D after i residues consumed

% Scores after i=0 residues.
prevM=-inf(1,K);
prevI=-inf(1,K);
prevI0=negInf;

% Initial deletion chain B->D1->D2->... at i=0.
base=-inf(1,K); basePred=zeros(1,K,'uint8');
base(1)=p.logBegin(3); basePred(1)=4;
if K>1
    c=squeeze(p.logInternal(:,3,3))'; % D_k -> D_{k+1}
    P=[0 cumsum(c)];
    z=base-P;
    [mx,startIdx]=prefix_cummax(z);
    prevD=P+mx;
    for k=1:K
        if startIdx(k)==k, psiD(1,k)=basePred(k); else, psiD(1,k)=3; end
    end
else
    prevD=base;
    psiD(1,1)=4;
end

% For i=0, base(k>1) must include a non-D start if one exists; none does,
% so the prefix construction above correctly produces the pure D chain.

for i=1:n
    x=code(i);

    % I0: leading insertion loop.
    if i==1
        cand=[p.logBegin(2), negInf];
        pred=[4 5];
    else
        cand=[negInf, prevI0+p.logI0(2)];
        pred=[4 5];
    end
    [best,q]=max(cand);
    curI0=best+p.logEmitI(1,x);
    psiI(i,1)=uint8(pred(q));

    % M1 boundary: B->M1 only for i=1; otherwise I0->M1.
    curM=-inf(1,K);
    if i==1
        cand=[p.logBegin(1), negInf]; pred=[4 5];
    else
        cand=[negInf, prevI0+p.logI0(1)]; pred=[4 5];
    end
    [best,q]=max(cand);
    curM(1)=best+p.logEmitM(1,x);
    psiM(i,1)=uint8(pred(q));

    % M_k, k>=2, vectorized over profile positions.
    if K>1
        tM=squeeze(p.logInternal(:,1,1))';
        tI=squeeze(p.logInternal(:,2,1))';
        tD=squeeze(p.logInternal(:,3,1))';
        vals=[prevM(1:K-1)+tM; prevI(1:K-1)+tI; prevD(1:K-1)+tD];
        [best,pred]=max(vals,[],1);
        curM(2:K)=best+p.logEmitM(2:K,x)';
        psiM(i,2:K)=uint8(pred);
    end

    % I_k, k=1..K.  At k=K use terminal insertion transitions.
    curI=-inf(1,K);
    if K>1
        tM=[squeeze(p.logInternal(:,1,2))' p.logTerminal(1,1)];
        tI=[squeeze(p.logInternal(:,2,2))' p.logTerminal(2,1)];
        tD=[squeeze(p.logInternal(:,3,2))' p.logTerminal(3,1)];
    else
        tM=p.logTerminal(1,1); tI=p.logTerminal(2,1); tD=p.logTerminal(3,1);
    end
    vals=[prevM+tM; prevI+tI; prevD+tD];
    [best,pred]=max(vals,[],1);
    curI=best+p.logEmitI(2:K+1,x)';
    psiI(i,2:K+1)=uint8(pred);

    % D row.  base(k) is the best NON-D entrance to D_k; D->D chains
    % are then evaluated exactly with a max-plus prefix scan.
    base=-inf(1,K); basePred=zeros(1,K,'uint8');
    base(1)=curI0+p.logI0(3); basePred(1)=5;
    if K>1
        md=curM(1:K-1)+squeeze(p.logInternal(:,1,3))';
        id=curI(1:K-1)+squeeze(p.logInternal(:,2,3))';
        chooseI=id>md;
        base(2:K)=md;
        base(2:K)=max(md,id);
        bp=ones(1,K-1,'uint8'); bp(chooseI)=2;
        basePred(2:K)=bp;

        c=squeeze(p.logInternal(:,3,3))';
        P=[0 cumsum(c)];
        z=base-P;
        [mx,startIdx]=prefix_cummax(z);
        curD=P+mx;
        for k=1:K
            if startIdx(k)==k
                psiD(i+1,k)=basePred(k);
            else
                psiD(i+1,k)=3;
            end
        end
    else
        curD=base;
        psiD(i+1,1)=5;
    end

    prevM=curM; prevI=curI; prevD=curD; prevI0=curI0;
end

% Global end after all residues and all K profile positions are consumed.
terminal=[prevM(K)+p.logTerminal(1,2), ...
          prevI(K)+p.logTerminal(2,2), ...
          prevD(K)+p.logTerminal(3,2)];
[best,state]=max(terminal);
if ~isfinite(best)
    error('No finite profile-HMM path for sequence of length %d.',n);
end

mchar=repmat('-',1,K);
ins=cell(1,K+1); for z=1:K+1, ins{z}=''; end

i=n; k=K; state=uint8(state);
steps=0; maxSteps=4*(n+K+10);
while state~=4
    steps=steps+1;
    if steps>maxSteps, error('Traceback exceeded safety bound.'); end
    switch state
        case 1 % M_k
            if i<1 || k<1, error('Invalid M traceback coordinate.'); end
            mchar(k)=obs(i);
            pred=psiM(i,k);
            i=i-1; k=k-1; state=pred;
        case 2 % I_k, k>=1
            if i<1 || k<1, error('Invalid I traceback coordinate.'); end
            ins{k+1}=[obs(i) ins{k+1}]; %#ok<AGROW>
            pred=psiI(i,k+1);
            i=i-1; state=pred;
        case 3 % D_k
            if k<1, error('Invalid D traceback coordinate.'); end
            pred=psiD(i+1,k);
            k=k-1; state=pred;
        case 5 % I0
            if i<1 || k~=0, error('Invalid I0 traceback coordinate.'); end
            ins{1}=[obs(i) ins{1}]; %#ok<AGROW>
            pred=psiI(i,1);
            i=i-1; state=pred;
        otherwise
            error('Unknown traceback state %d.',state);
    end
    if state==0, error('Broken traceback.'); end
end
if i~=0 || k~=0
    error('Traceback reached Begin at (%d,%d), expected (0,0).',i,k);
end
end

function [mx,idx] = prefix_cummax(z)
%PREFIX_CUMMAX Compatibility replacement for the two-output form of CUMMAX.
% Returns, for every prefix z(1:k), the maximum value and the index of
% its first occurrence.  This works on MATLAB releases where CUMMAX only
% supports one output.
if isempty(z)
    mx=z;
    idx=zeros(size(z));
    return;
end
mx=zeros(size(z));
idx=zeros(size(z));
mx(1)=z(1);
idx(1)=1;
for ii=2:numel(z)
    if z(ii)>mx(ii-1)
        mx(ii)=z(ii);
        idx(ii)=ii;
    else
        mx(ii)=mx(ii-1);
        idx(ii)=idx(ii-1);
    end
end
end

function [p,alphabet] = build_parameters(sequence,tgene,egene,K)
s=profile_hmm_settings();
alphabet='AGCT';
tgene=min(max(tgene(:),0),1);
egene=min(max(egene(:),0),1);
if numel(tgene)~=9*K+3 || numel(egene)~=8*K+4
    error('Parameter vector length mismatch for K=%d.',K);
end

% Prefer the reference-free seed profile prepared from the input-only
% pre-alignment-informed Profile-HMM. If no context is installed, retain the
% original v4 unaligned-sequence prior as a compatibility fallback.
ctx=[];
if exist('msa_seed_context','file')==2
    ctx=msa_seed_context('get');
end
if ~isempty(ctx) && isfield(ctx,'K') && ctx.K==K
    mPrior=ctx.emitM;
    iPrior=ctx.emitI;
    bPrior=ctx.begin;
    i0Prior=ctx.i0;
    intPrior=ctx.internal;
    termPrior=ctx.terminal;
    T=ctx.PerturbationTemperature;
else
    pc=s.emissionPseudoCount;
    mc=pc*ones(K,4);
    gc=pc*ones(1,4);
    for r=1:numel(sequence)
        seq=upper(sequence(r).Sequence);
        c=encode_dna(seq,alphabet);
        n=length(c);
        if n==1
            pos=1;
        else
            pos=1+round((0:n-1)*(K-1)/(n-1));
        end
        for q=1:n
            mc(pos(q),c(q))=mc(pos(q),c(q))+1;
            gc(c(q))=gc(c(q))+1;
        end
    end
    mPrior=normalize_rows(mc);
    iPrior=repmat(gc/sum(gc),K+1,1);
    bPrior=[0.90 0.05 0.05];
    i0Prior=[0.72 0.23 0.05];
    intPrior=zeros(K-1,3,3);
    for k=1:K-1
        intPrior(k,1,:)=reshape([0.90 0.05 0.05],1,1,3);
        intPrior(k,2,:)=reshape([0.70 0.25 0.05],1,1,3);
        intPrior(k,3,:)=reshape([0.70 0.05 0.25],1,1,3);
    end
    termPrior=[0.05 0.95; 0.25 0.75; 0.05 0.95];
    T=s.perturbationTemperature;
end

idx=1;
[p.begin,idx]=take_group(tgene,idx,bPrior,T);
[p.i0,idx]=take_group(tgene,idx,i0Prior,T);
p.internal=zeros(K-1,3,3);
for k=1:K-1
    for src=1:3
        prior=squeeze(intPrior(k,src,:))';
        [v,idx]=take_group(tgene,idx,prior,T);
        p.internal(k,src,:)=reshape(v,1,1,3);
    end
end
p.terminal=zeros(3,2);
for src=1:3
    [v,idx]=take_group(tgene,idx,termPrior(src,:),T);
    p.terminal(src,:)=v;
end
if idx~=numel(tgene)+1, error('Transition parser mismatch.'); end

p.emitM=zeros(K,4);
for k=1:K
    [p.emitM(k,:),idxE]=take_group(egene,1+4*(k-1),mPrior(k,:),T); %#ok<ASGLU>
end
base=4*K;
p.emitI=zeros(K+1,4);
for k=0:K
    start=base+1+4*k;
    [p.emitI(k+1,:),~]=take_group(egene,start,iPrior(k+1,:),T);
end

p.logBegin=log(max(p.begin,realmin('double')));
p.logI0=log(max(p.i0,realmin('double')));
p.logInternal=log(max(p.internal,realmin('double')));
p.logTerminal=log(max(p.terminal,realmin('double')));
p.logEmitM=log(max(p.emitM,realmin('double')));
p.logEmitI=log(max(p.emitI,realmin('double')));
end

function [prob,next] = take_group(gene,start,prior,T)
len=numel(prior);
g=gene(start:start+len-1)';
w=prior(:)'.*exp(T*(2*g-1));
if ~all(isfinite(w)) || sum(w)<=0, w=prior(:)'; end
prob=w/sum(w);
next=start+len;
end

function X=normalize_rows(X)
rs=sum(X,2); X=X./rs;
end

function code=encode_dna(seq,alphabet)
code=zeros(1,length(seq));
for i=1:length(seq)
    q=find(alphabet==seq(i),1);
    if isempty(q), error('Unsupported nucleotide ''%s''.',seq(i)); end
    code(i)=q;
end
end
