function model = build_seed_profile(aligned,cfg)
%BUILD_SEED_PROFILE Estimate a standard profile-HMM prior from a seed MSA.
% The seed MSA must be derived only from the unaligned input sequences.

if nargin<2, cfg=struct(); end
if isfield(cfg,'hmmMatchOccupancy'), occThr=cfg.hmmMatchOccupancy;
elseif isfield(cfg,'seedMatchOccupancy'), occThr=cfg.seedMatchOccupancy; else, occThr=0.5; end
if isfield(cfg,'hmmTransitionPseudoCount'), tpc=cfg.hmmTransitionPseudoCount;
elseif isfield(cfg,'seedTransitionPseudoCount'), tpc=cfg.seedTransitionPseudoCount; else, tpc=0.5; end
if isfield(cfg,'hmmEmissionPseudoCount'), epc=cfg.hmmEmissionPseudoCount;
elseif isfield(cfg,'seedEmissionPseudoCount'), epc=cfg.seedEmissionPseudoCount; else, epc=0.5; end
if isfield(cfg,'hmmPerturbationTemperature'), temp=cfg.hmmPerturbationTemperature;
elseif isfield(cfg,'seedPerturbationTemperature'), temp=cfg.seedPerturbationTemperature; else, temp=0.75; end

[n,W]=size(aligned);
occ=sum(aligned~='-',1)/n;
matchMask=occ>=occThr;
K=sum(matchMask);
if K<1, error('Seed profile contains no Match states at occupancy threshold %.3f.',occThr); end

% Broad topology priors serve only as pseudocount proportions.
bBase=[0.90 0.05 0.05];
i0Base=[0.72 0.23 0.05];
intBase=zeros(K-1,3,3);
for k=1:K-1
    intBase(k,1,:)=reshape([0.90 0.05 0.05],1,1,3);
    intBase(k,2,:)=reshape([0.70 0.25 0.05],1,1,3);
    intBase(k,3,:)=reshape([0.70 0.05 0.25],1,1,3);
end
termBase=[0.05 0.95; 0.25 0.75; 0.05 0.95];

bc=tpc*bBase; i0c=tpc*i0Base; intc=tpc*intBase; termc=tpc*termBase;
mc=epc*ones(K,4); ic=epc*ones(K+1,4);
alphabet='AGCT';

for r=1:n
    types=zeros(1,2*W+K+4,'uint8'); ks=zeros(size(types)); ne=0;
    k=0;
    for c=1:W
        ch=upper(aligned(r,c));
        if matchMask(c)
            k=k+1;
            ne=ne+1; ks(ne)=k;
            if ch=='-'
                types(ne)=3; % D_k
            else
                types(ne)=1; % M_k
                q=find(alphabet==ch,1); if isempty(q), error('Unsupported nucleotide %s.',ch); end
                mc(k,q)=mc(k,q)+1;
            end
        elseif ch~='-'
            ne=ne+1; types(ne)=2; ks(ne)=k; % I_k, including I_0/I_K
            q=find(alphabet==ch,1); if isempty(q), error('Unsupported nucleotide %s.',ch); end
            ic(k+1,q)=ic(k+1,q)+1;
        end
    end
    types=types(1:ne); ks=ks(1:ne);
    if isempty(types), error('Empty path in seed profile.'); end

    % Begin -> first state.
    [bc,i0c,intc,termc]=add_transition(bc,i0c,intc,termc,0,0,types(1),ks(1),K);
    for z=1:ne-1
        [bc,i0c,intc,termc]=add_transition(bc,i0c,intc,termc,types(z),ks(z),types(z+1),ks(z+1),K);
    end
    % Last state -> End (target type 4).
    [bc,i0c,intc,termc]=add_transition(bc,i0c,intc,termc,types(end),ks(end),4,K+1,K);
end

model=struct();
model.K=K;
model.MatchMask=matchMask;
model.Occupancy=occ;
model.begin=normalize1(bc);
model.i0=normalize1(i0c);
model.internal=zeros(K-1,3,3);
for k=1:K-1
    for src=1:3
        model.internal(k,src,:)=reshape(normalize1(squeeze(intc(k,src,:))'),1,1,3);
    end
end
model.terminal=zeros(3,2);
for src=1:3, model.terminal(src,:)=normalize1(termc(src,:)); end
model.emitM=normalize_rows(mc);
model.emitI=normalize_rows(ic);
model.PerturbationTemperature=temp;
model.MatchOccupancyThreshold=occThr;
model.Source='UPGMA_identity_progressive_prealignment_informed_ProfileHMM_prior';
end

function [bc,i0c,intc,termc]=add_transition(bc,i0c,intc,termc,st,sk,tt,tk,K)
% st: 0 Begin, 1 M, 2 I, 3 D. tt=4 denotes End.
if st==0
    if tt==1 && tk==1, q=1;
    elseif tt==2 && tk==0, q=2;
    elseif tt==3 && tk==1, q=3;
    else, error('Illegal Begin transition.'); end
    bc(q)=bc(q)+1; return;
end
if st==2 && sk==0
    if tt==1 && tk==1, q=1;
    elseif tt==2 && tk==0, q=2;
    elseif tt==3 && tk==1, q=3;
    else, error('Illegal I0 transition.'); end
    i0c(q)=i0c(q)+1; return;
end
if sk<K
    src=st;
    if tt==1 && tk==sk+1, q=1;
    elseif tt==2 && tk==sk, q=2;
    elseif tt==3 && tk==sk+1, q=3;
    else, error('Illegal internal transition from state %d at k=%d.',st,sk); end
    intc(sk,src,q)=intc(sk,src,q)+1; return;
end
% k=K terminal transitions: source -> I_K or End.
if sk==K
    src=st;
    if tt==2 && tk==K, q=1;
    elseif tt==4, q=2;
    else, error('Illegal terminal transition from state %d at k=%d.',st,sk); end
    termc(src,q)=termc(src,q)+1; return;
end
error('Unrecognized seed-profile transition.');
end

function x=normalize1(x)
s=sum(x); if s<=0, error('Cannot normalize zero vector.'); end; x=x/s;
end
function X=normalize_rows(X)
rs=sum(X,2); X=X./rs;
end
