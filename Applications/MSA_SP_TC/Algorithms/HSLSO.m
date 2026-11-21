function [gbestx,bestever,gbesthistory] = HSLSO(sequence,aa,lengthdata,L,historyPoints,dimension,MaxFEs)
% HSLSO: MSA implementation consistent with the manuscript's Eq. (6)-(7).
% The total FE budget includes all population-initialization evaluations.
% Eq. (6): shrinking superior-level range based on FEs/MaxFEs.
% Eq. (7): level-wise update probability based on FEs/MaxFEs.

if nargin < 7 || isempty(MaxFEs)
    error('HSLSO requires explicit MaxFEs as the seventh input.');
end
if nargin < 6 || isempty(dimension) || dimension < 1 || dimension ~= floor(dimension)
    error('dimension must be a positive integer.');
end
if nargin < 5 || isempty(historyPoints), historyPoints=200; end

popsize=50;
NL=10;
if mod(popsize,NL)~=0
    error('HSLSO requires popsize divisible by NL.');
end
if MaxFEs < popsize || MaxFEs ~= floor(MaxFEs)
    error('MaxFEs must be an integer >= popsize (%d).',popsize);
end
m=popsize/NL;
phi=0.3;
xmin=0;
xmax=1;
vmin=0;
vmax=1;
PLinit=(1:NL)/NL;
PLfinal=1-PLinit;

p=zeros(popsize,dimension);
v=zeros(popsize,dimension);
fitnessx=zeros(popsize,1);
pbest=zeros(popsize,dimension);
pbestfitness=zeros(popsize,1);
feBest=nan(MaxFEs,1);
FEs=0;

% Initialization: every fitness call consumes one FE.
for i=1:popsize
    if i==1
        p(i,:)=0.5*ones(1,dimension);
    else
        p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    end
    v(i,:)=vmin+(vmax-vmin).*rand(1,dimension);
    fitnessx(i)=fitness(sequence,aa,lengthdata,L,p(i,:));
    FEs=FEs+1;
    feBest(FEs)=max([fitnessx(1:i); -inf]);
end

pbest=p;
pbestfitness=fitnessx;
[bestever,bestID]=max(fitnessx);
gbestx=p(bestID,:);
feBest(1:popsize)=cummax(fitnessx(:));

sweep=0;
noProgressSweeps=0;
while FEs<MaxFEs
    sweep=sweep+1;
    FEsAtSweepStart=FEs;

    % Rank current particles from best to worst and form NL equal levels.
    [fitnessx,rank]=sort(fitnessx,'descend');
    p=p(rank,:);
    v=v(rank,:);
    pbest=pbest(rank,:);
    pbestfitness=pbestfitness(rank);

    for level=2:NL
        if FEs>=MaxFEs, break; end

        % Eq. (7): update probability uses the consumed FE ratio.
        progress=min(FEs/MaxFEs,1);
        PL=PLinit(level)+progress*(PLfinal(level)-PLinit(level));

        for q=1:m
            if FEs>=MaxFEs, break; end
            if rand<=PL
                i=(level-1)*m+q;

                if level==2
                    % Both exemplars are independently sampled from level 1.
                    for j=1:dimension
                        donorA=randi(m);
                        donorB=randi(m);
                        learna(j)=pbest(donorA,j); %#ok<AGROW>
                        learnb(j)=pbest(donorB,j); %#ok<AGROW>
                    end
                else
                    % Eq. (6): shrink the admissible superior-level range.
                    progress=min(FEs/MaxFEs,1);
                    sourceRange=max(2,ceil((level-1)*(1-progress^2)));
                    sourceRange=min(sourceRange,level-1);
                    selectedLevels=randperm(sourceRange,2);
                    sourceLevelA=min(selectedLevels);
                    sourceLevelB=max(selectedLevels);

                    % Dimension-wise donor sampling within the two selected levels.
                    for j=1:dimension
                        donorA=(sourceLevelA-1)*m+randi(m);
                        donorB=(sourceLevelB-1)*m+randi(m);
                        learna(j)=pbest(donorA,j); %#ok<AGROW>
                        learnb(j)=pbest(donorB,j); %#ok<AGROW>
                    end
                end

                v(i,:)=rand(1,dimension).*v(i,:) ...
                    +rand(1,dimension).*(learna-p(i,:)) ...
                    +phi.*rand(1,dimension).*(learnb-p(i,:));
                p(i,:)=p(i,:)+v(i,:);
                p(i,:)=max(p(i,:),xmin);
                p(i,:)=min(p(i,:),xmax);

                fitnessx(i)=fitness(sequence,aa,lengthdata,L,p(i,:));
                FEs=FEs+1;

                if fitnessx(i)>pbestfitness(i)
                    pbestfitness(i)=fitnessx(i);
                    pbest(i,:)=p(i,:);
                end
                if fitnessx(i)>bestever
                    bestever=fitnessx(i);
                    gbestx=p(i,:);
                end
                feBest(FEs)=bestever;
            end
        end
    end

    if FEs==FEsAtSweepStart
        noProgressSweeps=noProgressSweeps+1;
    else
        noProgressSweeps=0;
    end
    if noProgressSweeps>10000
        error('HSLSO did not consume any FEs for 10000 consecutive sweeps.');
    end

    if mod(sweep,25)==0 || FEs>=MaxFEs
        fprintf('HSLSO: sweep=%d, FEs=%d/%d, best SPS=%e\n',sweep,FEs,MaxFEs,bestever);
    end
end

gbesthistory=msa_level_baseline_history(feBest,historyPoints);
gbestx=gbestx(:)';
end
