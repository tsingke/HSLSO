function [gbestx,BestValue,BestCost] =EAPSO(sequence,a,lengthdata,L,maxiter,dimension)
% (fhd,nPop,nVar,VarMin,VarMax,MaxIt,X)
popsize=50;
xmax=1;
xmin=0;
FEs=0;
MaxFEs=3e6;
k=a;

nPop=popsize;
nVar=dimension;
VarMin=xmin.*ones(1,nVar);
VarMax=xmax.*ones(1,nVar);
MaxIt=maxiter;

% MaxIt=floor(MaxIt/nPop*2);
% 使用结构体
VarSize=[1 nVar];  
empty_particle.Position=[]; % 粒子位置
empty_particle.Cost=[]; % 粒子适应度
empty_particle.Velocity=[]; % 粒子速度
empty_particle.Best.Position=[]; % 个体最优位置
empty_particle.Best.Cost=[]; % 个体最优适应度
particle=repmat(empty_particle,nPop,1); % npop行，1列
GlobalBest.Cost=eps;

% 种群初始化
for i=1:nPop
    particle(i).Position=VarMin+(VarMax-VarMin).*rand(1,nVar);% 位置
    particle(i).Velocity=zeros(VarSize);% 速度
    particle(i).Cost= fitness(sequence,k,lengthdata,L,particle(i).Position);% 适应度
    FEs=FEs+1;
    particle(i).Best.Position=particle(i).Position;% 个体最优位置
    particle(i).Best.Cost=particle(i).Cost;% 个体最优适应度
    if particle(i).Best.Cost>GlobalBest.Cost
        GlobalBest=particle(i).Best;% 全局最优
    end
end
pcount=2;% 存储Bt，保存Xt中所有粒子的一些有前途的历史最佳解决方案
for i=1:nPop
    cosp(i)=particle(i).Cost; % 个体最优位置
end
[~,inp]=sort(cosp);% 个体最优位置适应度排序
PART(1)=particle(inp(1)).Best;% 选择两个最好的粒子放入At和Bt中
PART(2)=particle(inp(2)).Best;
NP=nPop;
BestCost(i) = GlobalBest.Cost;

GART(1)=GlobalBest;% Ct，保存一些有前途的全局最佳解决方案而构建的，其最大长度也等于种群大小
gcount=1;% Ct
VelMax=(VarMax-VarMin);% 速度最大值
VelMin=-VelMax;% 速度最小值
it=2;
while it <= maxiter 
    
    for i=1:nPop
        cosp(i)=particle(i).Best.Cost;% 个体最优位置
    end
    [~,ind]=sort(cosp);
    WINNER=ind(1:nPop/2);% 胜者
    LOSER= ind(nPop/2+1:nPop);% 败者
    for i=1:nPop/2
        a=randperm(length(PART),1);
        b=randperm(length(PART),1);
        while a==b
            a=randperm(length(PART),1);
            b=randperm(length(PART),1);
        end
        if PART(a).Cost<PART(b).Cost
            a=b;
        end
        if length(GART)>1
            c=randperm(length(GART),1);
            d=randperm(length(GART),1);
            while c==d
                c=randperm(length(GART),1);
                d=randperm(length(GART),1);
            end
            if GART(c).Cost<GART(d).Cost
                b=d;
            else
                b=c;
            end
        else
            b=1;
        end
        c=randperm(nPop/2,1);
        q=randperm(nPop/2,1);
        while c==q
            c=randperm(nPop/2,1);
            q=randperm(nPop/2,1);
        end
        if  particle(WINNER(q)).Best.Cost>particle(WINNER(c)).Best.Cost
            c=q;
        end
        w=rand(1,nVar);
        MM=particle(LOSER(i)).Position;
        F1=rand(1,nVar);
        F2=rand(1,nVar);
        %% 算法3
        if cosp(LOSER(i))>mean(cosp(LOSER))
            if PART(a).Cost>GART(b).Cost && PART(a).Cost>particle(WINNER(c)).Best.Cost 
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif PART(a).Cost<GART(b).Cost && GART(b).Cost>particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(GART(b).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif  PART(a).Cost<particle(WINNER(c)).Best.Cost &&  GART(b).Cost<particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(particle(WINNER(c)).Best.Position-MM)+F2.*(GlobalBest.Position-MM);
            end
        else
             if PART(a).Cost<GART(b).Cost && PART(a).Cost<particle(WINNER(c)).Best.Cost 
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(GART(b).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(LOSER(i)).Position);
             elseif PART(a).Cost>GART(b).Cost && GART(b).Cost<particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(LOSER(i)).Position);
             elseif PART(a).Cost>particle(WINNER(c)).Best.Cost && GART(b).Cost>particle(WINNER(c)).Best.Cost
                 particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(GART(b).Position-particle(LOSER(i)).Position); 
             end
        end
        particle(LOSER(i)).Velocity = max(particle(LOSER(i)).Velocity,VelMin);% 速度边界控制
        particle(LOSER(i)).Velocity = min(particle(LOSER(i)).Velocity,VelMax);
        particle(LOSER(i)).Position = particle(LOSER(i)).Position + particle(LOSER(i)).Velocity;
        IsOutside=(particle(LOSER(i)).Position<VarMin | particle(LOSER(i)).Position>VarMax);% 位置边界控制
        particle(LOSER(i)).Velocity(IsOutside)=-particle(LOSER(i)).Velocity(IsOutside);
        particle(LOSER(i)).Position = max(particle(LOSER(i)).Position,VarMin);
        particle(LOSER(i)).Position = min(particle(LOSER(i)).Position,VarMax);
        particle(LOSER(i)).Cost = fitness(sequence,k,lengthdata,L,particle(i).Position);
        FEs=FEs+1;
        % 更新Bt
        if particle(LOSER(i)).Cost>particle(LOSER(i)).Best.Cost
            particle(LOSER(i)).Best.Position=particle(LOSER(i)).Position;
            particle(LOSER(i)).Best.Cost=particle(LOSER(i)).Cost;
            pcount=pcount+1;
            if pcount<=NP
                PART(pcount).Position=particle(LOSER(i)).Position;
                PART(pcount).Cost=particle(LOSER(i)).Cost;
            else
                a=randperm(NP,1);
                b=randperm(NP,1);
                while a==b
                    a=randperm(NP,1);
                    b=randperm(NP,1);
                end
                if PART(a).Cost>PART(b).Cost
                    if particle(LOSER(i)).Cost>PART(b).Cost
                        PART(b).Cost=particle(LOSER(i)).Cost;
                        PART(b).Position=particle(LOSER(i)).Position;
                    end
                else
                    if particle(LOSER(i)).Cost>PART(a).Cost
                        PART(a).Cost=particle(LOSER(i)).Cost;
                        PART(a).Position=particle(LOSER(i)).Position;
                    end
                end
            end
        end
        if particle(LOSER(i)).Best.Cost>GlobalBest.Cost
            GlobalBest=particle(LOSER(i)).Best;
        end
    end
    % 更新Ct
    gcount=gcount+1;
    if gcount<=NP
        GART(gcount).Cost= GlobalBest.Cost;
        GART(gcount).Position= GlobalBest.Position;
    else
        a=randperm(length(GART),1);
        b=randperm(length(GART),1);
        while a==b
            a=randperm(length(GART),1);
            b=randperm(length(GART),1);
        end
        if GART(b).Cost<GART(a).Cost
            a=b;
        end
        if  GlobalBest.Cost<GART(a).Cost
            GART(a).Cost= GlobalBest.Cost;
            GART(a).Position= GlobalBest.Position;
        end   
    end
    BestValue=GlobalBest.Cost;
    gbestx=GlobalBest.Position;

    BestCost(it) = GlobalBest.Cost;

        fprintf("EAPSO 第%d代，最佳适应度 = %e\n",it,BestValue);
    
    if FEs >= MaxFEs
        break;
    end
    it = it+1;
end
end


