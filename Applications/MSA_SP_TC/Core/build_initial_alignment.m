function [aligned, info] = build_initial_alignment(sequence,cfg)
%BUILD_INITIAL_ALIGNMENT UPGMA-guided identity progressive protein pre-alignment.
%
% This is a deliberately lightweight, reference-free pre-alignment used
% ONLY to initialize the nucleotide Profile-HMM. The procedure is:
%   DNA -> protein translation -> pairwise identity distances -> UPGMA guide
%   tree -> progressive cluster-profile alignment using residue identity
%   scoring -> codon projection back to DNA.
%
% No BAliBASE reference alignment is used here, and no iterative refinement
% is performed during the pre-alignment stage.

if nargin < 2, cfg=struct(); end
if isfield(cfg,'preAlignmentProteinMatch'), matchScore=cfg.preAlignmentProteinMatch; else, matchScore=1; end
if isfield(cfg,'preAlignmentProteinMismatch'), mismatchScore=cfg.preAlignmentProteinMismatch; else, mismatchScore=-1; end
if isfield(cfg,'preAlignmentProteinGapOpen'), gapOpen=cfg.preAlignmentProteinGapOpen; else, gapOpen=-4; end
if isfield(cfg,'preAlignmentProteinGapExtend'), gapExtend=cfg.preAlignmentProteinGapExtend; else, gapExtend=-0.5; end

n=numel(sequence);
if n<2, error('At least two sequences are required for MSA.'); end
nuc=cell(n,1); prot=cell(n,1);
for i=1:n
    nuc{i}=upper(sequence(i).Sequence);
    if mod(length(nuc{i}),3)~=0
        error('Sequence %d length (%d) is not divisible by 3; codon projection is not applicable.',i,length(nuc{i}));
    end
    prot{i}=translate_standard_dna(nuc{i});
end

% Pairwise protein identity distances used only to construct the UPGMA tree.
pairIdentity=eye(n);
pairDistance=zeros(n,n);
pairScore=zeros(n,n);
for i=1:n-1
    for j=i+1:n
        [ai,aj,sc]=global_affine_identity(prot{i},prot{j},matchScore,mismatchScore,gapOpen,gapExtend);
        mask=(ai~='-' & aj~='-');
        if any(mask)
            pid=sum(ai(mask)==aj(mask))/sum(mask);
        else
            pid=0;
        end
        pairIdentity(i,j)=pid; pairIdentity(j,i)=pid;
        pairDistance(i,j)=1-pid; pairDistance(j,i)=1-pid;
        pairScore(i,j)=sc; pairScore(j,i)=sc;
    end
end

% UPGMA guide tree. Each active cluster stores a protein alignment profile
% and the original sequence indices corresponding to its rows.
maxNodes=2*n-1;
profiles=cell(maxNodes,1);
leaves=cell(maxNodes,1);
for i=1:n
    profiles{i}=prot{i};
    leaves{i}=i;
end
active=1:n;
mergeHistory=zeros(n-1,4); % nodeA,nodeB,sizeA,sizeB
nextNode=n+1;

for step=1:n-1
    bestDist=inf; bestA=0; bestB=0; bestTie=[inf inf];
    for ia=1:numel(active)-1
        ca=active(ia);
        for ib=ia+1:numel(active)
            cb=active(ib);
            la=leaves{ca}; lb=leaves{cb};
            vals=zeros(1,numel(la)*numel(lb)); z=0;
            for p=1:numel(la)
                for q=1:numel(lb)
                    z=z+1; vals(z)=pairDistance(la(p),lb(q));
                end
            end
            d=mean(vals);
            tie=[min(la) min(lb)];
            if d<bestDist-1e-12 || (abs(d-bestDist)<=1e-12 && lex_less(tie,bestTie))
                bestDist=d; bestA=ca; bestB=cb; bestTie=tie;
            end
        end
    end

    pa=profiles{bestA}; pb=profiles{bestB};
    profiles{nextNode}=merge_profiles_by_consensus(pa,pb,matchScore,mismatchScore,gapOpen,gapExtend);
    leaves{nextNode}=[leaves{bestA} leaves{bestB}];
    mergeHistory(step,:)=[bestA bestB numel(leaves{bestA}) numel(leaves{bestB})];
    active(active==bestA | active==bestB)=[];
    active(end+1)=nextNode; %#ok<AGROW>
    nextNode=nextNode+1;
end

rootNode=active(1);
protProfile=profiles{rootNode};
rootLeaves=leaves{rootNode};
if ischar(protProfile) && size(protProfile,1)==1 && n==1
    protAligned=protProfile;
else
    protAligned=repmat('-',n,size(protProfile,2));
    for r=1:numel(rootLeaves)
        protAligned(rootLeaves(r),:)=protProfile(r,:);
    end
end

% Remove any all-gap protein columns (normally none).
protAligned=protAligned(:,any(protAligned~='-',1));
Waa=size(protAligned,2);

% Codon projection: each protein alignment column becomes three DNA columns.
aligned=repmat('-',n,3*Waa);
for i=1:n
    q=0;
    for c=1:Waa
        cols=3*(c-1)+(1:3);
        if protAligned(i,c)~='-'
            q=q+1;
            aligned(i,cols)=nuc{i}(3*(q-1)+(1:3));
        end
    end
    if q~=length(prot{i}), error('Codon projection lost residues for sequence %d.',i); end
    if ~strcmp(aligned(i,aligned(i,:)~='-'),nuc{i})
        error('Pre-alignment residue conservation failed for sequence %d.',i);
    end
end
aligned=aligned(:,any(aligned~='-',1));

info=struct();
info.Method='UPGMA_identity_progressive_protein_prealignment_for_ProfileHMM_initialization';
info.GuideTree='UPGMA_from_pairwise_protein_identity_distance';
info.ProgressiveMerge='cluster_consensus_identity_alignment_no_refinement';
info.ProteinMatch=matchScore;
info.ProteinMismatch=mismatchScore;
info.ProteinGapOpen=gapOpen;
info.ProteinGapExtend=gapExtend;
info.ProteinAlignmentWidth=Waa;
info.NucleotideAlignmentWidth=size(aligned,2);
info.PairIdentityMatrix=pairIdentity;
info.PairDistanceMatrix=pairDistance;
info.PairScoreMatrix=pairScore;
info.MergeHistory=mergeHistory;
end

function tf=lex_less(a,b)
if a(1)<b(1), tf=true;
elseif a(1)>b(1), tf=false;
else, tf=a(2)<b(2); end
end

function merged=merge_profiles_by_consensus(pa,pb,matchScore,mismatchScore,gapOpen,gapExtend)
% Align cluster consensuses, then propagate the new gaps to all profile rows.
if isvector(pa), pa=reshape(pa,1,[]); end
if isvector(pb), pb=reshape(pb,1,[]); end
ca=profile_consensus(pa);
cb=profile_consensus(pb);
[ga,gb]=global_affine_identity(ca,cb,matchScore,mismatchScore,gapOpen,gapExtend);
W=length(ga);
na=size(pa,1); nb=size(pb,1);
newA=repmat('-',na,W); newB=repmat('-',nb,W);
ia=0; ib=0;
for z=1:W
    if ga(z)~='-'
        ia=ia+1;
        newA(:,z)=pa(:,ia);
    end
    if gb(z)~='-'
        ib=ib+1;
        newB(:,z)=pb(:,ib);
    end
end
if ia~=size(pa,2) || ib~=size(pb,2), error('Progressive profile merge lost columns.'); end
merged=[newA;newB];
merged=merged(:,any(merged~='-',1));
end

function c=profile_consensus(P)
% One representative residue per profile column; gaps are ignored.
W=size(P,2); c=repmat('X',1,W);
order='ARNDCQEGHILKMFPSTWYV*X';
for j=1:W
    col=P(:,j); col=col(col~='-');
    if isempty(col), error('All-gap column found inside a profile.'); end
    bestCount=-1; bestAA='X'; bestRank=inf;
    u=unique(col(:)');
    for k=1:numel(u)
        aa=u(k); cnt=sum(col==aa);
        rank=find(order==aa,1); if isempty(rank), rank=100+double(aa); end
        if cnt>bestCount || (cnt==bestCount && rank<bestRank)
            bestCount=cnt; bestAA=aa; bestRank=rank;
        end
    end
    c(j)=bestAA;
end
end

function aa=translate_standard_dna(seq)
seq=upper(seq); aa=repmat('X',1,length(seq)/3);
for i=1:length(aa)
    codon=seq(3*(i-1)+(1:3));
    switch codon
        case {'TTT','TTC'}, aa(i)='F';
        case {'TTA','TTG','CTT','CTC','CTA','CTG'}, aa(i)='L';
        case {'ATT','ATC','ATA'}, aa(i)='I';
        case 'ATG', aa(i)='M';
        case {'GTT','GTC','GTA','GTG'}, aa(i)='V';
        case {'TCT','TCC','TCA','TCG','AGT','AGC'}, aa(i)='S';
        case {'CCT','CCC','CCA','CCG'}, aa(i)='P';
        case {'ACT','ACC','ACA','ACG'}, aa(i)='T';
        case {'GCT','GCC','GCA','GCG'}, aa(i)='A';
        case {'TAT','TAC'}, aa(i)='Y';
        case {'TAA','TAG','TGA'}, aa(i)='*';
        case {'CAT','CAC'}, aa(i)='H';
        case {'CAA','CAG'}, aa(i)='Q';
        case {'AAT','AAC'}, aa(i)='N';
        case {'AAA','AAG'}, aa(i)='K';
        case {'GAT','GAC'}, aa(i)='D';
        case {'GAA','GAG'}, aa(i)='E';
        case {'TGT','TGC'}, aa(i)='C';
        case 'TGG', aa(i)='W';
        case {'CGT','CGC','CGA','CGG','AGA','AGG'}, aa(i)='R';
        case {'GGT','GGC','GGA','GGG'}, aa(i)='G';
        otherwise, error('Unsupported/ambiguous codon %s.',codon);
    end
end
end

function [ga,gb,best]=global_affine_identity(a,b,matchScore,mismatchScore,gapOpen,gapExtend)
% Needleman-Wunsch global alignment with affine gaps and identity scoring.
m=length(a); n=length(b); neg=-inf;
M=neg*ones(m+1,n+1); X=M; Y=M;
pM=zeros(m+1,n+1,'uint8'); pX=pM; pY=pM;
M(1,1)=0;
for i=1:m
    if i==1, X(i+1,1)=gapOpen; pX(i+1,1)=1;
    else, X(i+1,1)=X(i,1)+gapExtend; pX(i+1,1)=2; end
end
for j=1:n
    if j==1, Y(1,j+1)=gapOpen; pY(1,j+1)=1;
    else, Y(1,j+1)=Y(1,j)+gapExtend; pY(1,j+1)=3; end
end
for i=1:m
    for j=1:n
        if a(i)==b(j), s=matchScore; else, s=mismatchScore; end
        [v,st]=max([M(i,j),X(i,j),Y(i,j)]);
        M(i+1,j+1)=v+s; pM(i+1,j+1)=uint8(st);
        [v,st]=max([M(i,j+1)+gapOpen,X(i,j+1)+gapExtend,Y(i,j+1)+gapOpen]);
        X(i+1,j+1)=v; pX(i+1,j+1)=uint8(st);
        [v,st]=max([M(i+1,j)+gapOpen,X(i+1,j)+gapOpen,Y(i+1,j)+gapExtend]);
        Y(i+1,j+1)=v; pY(i+1,j+1)=uint8(st);
    end
end
[best,state]=max([M(m+1,n+1),X(m+1,n+1),Y(m+1,n+1)]);
i=m; j=n; ca=cell(1,m+n+4); cb=ca; z=0;
while i>0 || j>0
    z=z+1;
    switch state
        case 1
            if i<1 || j<1, error('Invalid M traceback.'); end
            ca{z}=a(i); cb{z}=b(j); state=double(pM(i+1,j+1)); i=i-1; j=j-1;
        case 2
            if i<1, error('Invalid X traceback.'); end
            ca{z}=a(i); cb{z}='-'; state=double(pX(i+1,j+1)); i=i-1;
        case 3
            if j<1, error('Invalid Y traceback.'); end
            ca{z}='-'; cb{z}=b(j); state=double(pY(i+1,j+1)); j=j-1;
        otherwise
            error('Broken affine traceback.');
    end
end
ca=ca(1:z); cb=cb(1:z);
ga=fliplr([ca{:}]); gb=fliplr([cb{:}]);
end
