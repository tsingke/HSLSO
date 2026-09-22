function [gbestx,gbestfitness,gbesthistory]=D_CSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation, FuncId,VisualSwitch)
MaxFEs = 3e6;
FEs = 0;
num_initial = 500;
num_vari = dimension;
varphi = 0.1; %100维0,500维0.1和0.05,1000维0.15和0.1
func_num = FuncId;
benchmark_func = fCalculation;

sample_x = lhsdesign(num_initial, num_vari).*(xmax - xmin) + xmin.*ones(num_initial, num_vari);
for i = 1:num_initial
    sample_y(i,:) = benchmark_func(sample_x(i,:)',func_num);
end
FEs = FEs + num_initial;
v = zeros(num_initial,num_vari);  %v初始化为0
fmin = min(sample_y);
gbesthistory(1:FEs) = fmin;

while FEs <= MaxFEs

    % 根据种群熵来划分pb和pw,种群熵的计算与MATLAB中entropy计算熵类似
    n_bin = num_initial;    %n_bin表示区间数
    res = rescale(sample_y);    % rescale将数组的条目缩放到区间 [0,1]
    p = imhist(res,n_bin);  % rescale将数组的条目缩放到区间 [0,1]
    p(p==0) = [];   % 除去p中的0
    p = p ./ numel(sample_y);% 正则化p使得sum(p)为1,numel计算数组中元素的数目

    % 为了E最大为1,所以用logn(p),而不是log2(p)
    n_nozero = size(p,1);
    E = -sum(p.*(log(p)./log(n_nozero)));
    
    % 划分pb和pw
    [sample_y,ind]= sort(sample_y); %按升序对 A 的元素进行排序
    sample_x = sample_x(ind,:);
    v = v(ind,:);
    mean_x = mean(sample_x);
    d = 0.45;   %100维0.25,500维0.35,1000维0.45
    if E > 1 - d
        mb = -(num_initial / d) * E + num_initial / d;
    else
        mb = num_initial;
    end
    mb = ceil(mb);
    mw = num_initial - mb;
    pb = sample_x(1:mb,:);  %越小越好
    pby = sample_y(1:mb,:);
    pbv = v(1:mb,:);
    pw = sample_x((mb + 1):num_initial,:);
    pwv = v((mb + 1):num_initial,:);

    next_samplex = [];
    next_sampley = [];
    next_v = [];
    % 更新pw
    for i = 1:size(pw,1)
        if size(pb,1) == 0  %pb有为空的问题
            xj = sample_x(1,:);   %当前最小值
        else
            xj = pb(randi(size(pb,1)),:);
        end
        pwv(i,:) = rand(1, num_vari) .* pwv(i,:) + rand(1, num_vari) .* (xj - pw(i,:)) + varphi .* rand(1, num_vari) .* (mean_x - pw(i,:));
        xi = pw(i,:) + pwv(i,:);
        xi(xi > xmax) = xmax;
        xi(xi < xmin) = xmin;% 范围检查
        if FEs <= MaxFEs
            yi = benchmark_func(xi',func_num);
            if yi < fmin
                fmin = yi;
                gbestx = xi;
            end
            FEs = FEs + 1;
            gbesthistory(FEs) = fmin;
            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                fprintf("DCSO算法,第%d次评价，最佳适应度 = %e\n",FEs,fmin);
            end
            next_samplex = [next_samplex;xi];   %加到下一次迭代中
            next_sampley = [next_sampley;yi];
            next_v = [next_v;pwv(i,:)];
        else
            break;
        end
    end

    % 更新pb
    while ~isempty(pb)
        if size(pb,1) == 1
            next_samplex = [next_samplex;pb];
            next_sampley = [next_sampley;pby];
            next_v = [next_v;pbv];
            pb = [];
            pby = [];
            pbv = [];
        else
            k1 = randi(size(pb, 1));    %随机抽取两个点
            k2 = randi(size(pb, 1));
            while k2 == k1
                k2 = randi(size(pb, 1));
            end
            if pby(k1,1) < pby(k2,1)
                add = k1;
                update = k2;
            else
                add = k2;
                update = k1;
            end
            next_samplex = [next_samplex;pb(add,:)]; %add直接加入
            next_sampley = [next_sampley;pby(add,:)];
            next_v = [next_v;pbv(add,:)];
            pbv(update,:) = rand(1, num_vari) .* pbv(update,:) + rand(1, num_vari) .* (pb(add,:) - pb(update,:));    %update需要更新
            xl = pb(update,:) + pbv(update,:);
            xl(xl > xmax) = xmax;
            xl(xl < xmin) = xmin;% 范围检查
            if FEs <= MaxFEs
                yl = benchmark_func(xl',func_num);
                if yl < fmin
                    fmin = yl;
                    gbestx = xl;
                end
                FEs = FEs + 1;
                gbesthistory(FEs) = fmin;
                if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                    fprintf("DCSO算法,第%d次评价，最佳适应度 = %e\n",FEs,fmin);
                end
                next_samplex = [next_samplex;xl];   %加到下一次迭代中
                next_sampley = [next_sampley;yl];
                next_v = [next_v;pbv(update,:)];
            else
                break;
            end

            % 从pb中去掉k1和k2,从后面删除,防止后面删除的数索引发生变化
            if k1 > k2
                first = k1;
                second = k2;
            else
                first = k2;
                second = k1;
            end
            pb(first, :) = [];
            pby(first, :) = [];
            pbv(first, :) = [];
            pb(second, :) = [];
            pby(second, :) = [];
            pbv(second, :) = [];
        end
    end
    
    % 更新sample_x,v和sample_y
    if size(next_samplex,1) ~= num_initial   %代表程序结束
        break;
    else
        sample_x = next_samplex;
        sample_y = next_sampley;
        v = next_v;
    end
    gbestfitness = fmin;
end
if FEs<MaxFEs
    gbesthistory(FEs+1:MaxFEs)=gbestfitness;
else
    if FEs>MaxFEs
        gbesthistory(MaxFEs+1:end)=[];
    end
end
end