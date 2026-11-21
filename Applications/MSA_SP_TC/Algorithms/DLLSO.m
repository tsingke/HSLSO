function [gbestx,bestever,gbesthistory] = DLLSO(sequence,a,lengthdata,L,maxiter,dimension,opts)
%DLLSO MSA adaptation of the dynamic level-based learning algorithm.
% Maximizes internal SPS. Adaptive roulette level selection is preserved.
if nargin<7, opts=struct(); end
[popsize,MaxFEs]=msa_level_baseline_setup(maxiter,opts);
phi=0.4;
S=[4,6,8,10,20,50];
% L2 must sample two *distinct* particles from L1; at 50 particles NL=50
% would make L1 a singleton. Discard infeasible choices (documented).
S=S(S<=floor(popsize/2));
if isempty(S), error('DLLSO requires popsize >= 8.'); end
R=ones(1,numel(S));
p=rand(popsize,dimension);p(1,:)=0.5;
v=zeros(popsize,dimension);
fitnessx=zeros(popsize,1);
feBest=zeros(MaxFEs,1);FEs=0;bestever=-inf;gbestx=[];
for i=1:popsize
    v(i,:)=rand(1,dimension); % Source DLLSO initial velocity in [0,1].
    fitnessx(i)=fitness(sequence,a,lengthdata,L,p(i,:));FEs=FEs+1;
    if fitnessx(i)>bestever,bestever=fitnessx(i);gbestx=p(i,:);end
    feBest(FEs)=bestever;
end
while FEs<MaxFEs
    pr=exp(7*(R-max(R)));pr=pr/sum(pr); % numerically stable softmax
    action=find(rand<=cumsum(pr),1,'first');
    if isempty(action),action=numel(S);end
    NL=S(action); oldBest=bestever;
    [~,rankIndex]=sort(fitnessx,'descend');
    ls=floor(popsize/NL);levels=cell(1,NL);
    for lev=1:NL-1, levels{lev}=rankIndex((lev-1)*ls+1:lev*ls);end
    levels{NL}=rankIndex((NL-1)*ls+1:end);
    for lev=NL:-1:2
        members=levels{lev};
        for jj=1:numel(members)
            if FEs>=MaxFEs,break;end
            k=members(jj);
            if lev==2
                pick=randperm(numel(levels{1}),2);
                e1=levels{1}(pick(1));e2=levels{1}(pick(2));
                if fitnessx(e2)>fitnessx(e1),tmp=e1;e1=e2;e2=tmp;end
            else
                chosen=sort(randperm(lev-1,2));
                pool1=levels{chosen(1)};pool2=levels{chosen(2)};
                e1=pool1(randi(numel(pool1)));
                e2=pool2(randi(numel(pool2)));
            end
            v(k,:)=rand(1,dimension).*v(k,:) ...
                +rand(1,dimension).*(p(e1,:)-p(k,:)) ...
                +phi*rand(1,dimension).*(p(e2,:)-p(k,:));
            p(k,:)=min(max(p(k,:)+v(k,:),0),1);
            fitnessx(k)=fitness(sequence,a,lengthdata,L,p(k,:));FEs=FEs+1;
            if fitnessx(k)>bestever,bestever=fitnessx(k);gbestx=p(k,:);end
            feBest(FEs)=bestever;
        end
        if FEs>=MaxFEs,break;end
    end
    % Relative improvement reward of selected level count (source Eq. 8).
    R(action)=abs(oldBest-bestever)/max(abs(oldBest),eps);
end
gbesthistory=msa_level_baseline_history(feBest,maxiter);
end
