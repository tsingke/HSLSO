function [gbestx,bestever,gbesthistory] = CSO(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% Competitive Swarm Optimizer, MSA adaptation with exact FE budget.
if nargin<7 || isempty(MaxFEs), error('CSO requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
popsize=50; phi=0.2; xmin=0; xmax=1; vmin=0; vmax=1;
if MaxFEs<popsize || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end
p=zeros(popsize,dimension); v=zeros(popsize,dimension); fitnessx=zeros(popsize,1);
feBest=nan(MaxFEs,1); FEs=0;
for i=1:popsize
    if i==1, p(i,:)=0.5*ones(1,dimension);
    else, p(i,:)=xmin+(xmax-xmin).*rand(1,dimension); end
    v(i,:)=vmin+(vmax-vmin).*rand(1,dimension);
    fitnessx(i)=fitness(sequence,a,lengthdata,L,p(i,:));
    FEs=FEs+1;
    if i==1, bestever=fitnessx(i); gbestx=p(i,:);
    elseif fitnessx(i)>bestever, bestever=fitnessx(i); gbestx=p(i,:); end
    feBest(FEs)=bestever;
end

gen=0;
while FEs<MaxFEs
    gen=gen+1;
    rlist=randperm(popsize);
    rpairs=[rlist(1:ceil(popsize/2));rlist(floor(popsize/2)+1:popsize)]';
    center=ones(ceil(popsize/2),1)*mean(p,1);
    mask=fitnessx(rpairs(:,1))<fitnessx(rpairs(:,2));
    losers=zeros(ceil(popsize/2),1); winners=losers;
    for k=1:ceil(popsize/2)
        if mask(k)==0
            losers(k)=rpairs(k,2); winners(k)=rpairs(k,1);
        else
            losers(k)=rpairs(k,1); winners(k)=rpairs(k,2);
        end
    end
    for k=1:ceil(popsize/2)
        if FEs>=MaxFEs, break; end
        li=losers(k);
        rr=rand(3,dimension);
        v(li,:)=rr(1,:).*v(li,:) ...
            +rr(2,:).*(p(winners(k),:)-p(li,:)) ...
            +phi*rr(3,:).*(center(k,:)-p(li,:));
        p(li,:)=p(li,:)+v(li,:);
        p(li,:)=max(p(li,:),xmin); p(li,:)=min(p(li,:),xmax);
        fitnessx(li)=fitness(sequence,a,lengthdata,L,p(li,:));
        FEs=FEs+1;
        if fitnessx(li)>=bestever, bestever=fitnessx(li); gbestx=p(li,:); end
        feBest(FEs)=bestever;
    end
    if mod(gen,25)==0 || FEs>=MaxFEs
        fprintf('CSO: generation=%d, FEs=%d/%d, best SPS=%e\n',gen,FEs,MaxFEs,bestever);
    end
end
gbesthistory=msa_level_baseline_history(feBest,historyPoints);
gbestx=gbestx(:)';
end
