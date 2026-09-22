function rounded_num = bankerRound(num, dec_place)
% 此算法为银行家舍入算法
% 注：matlab默认保留四位小数

% x为输入浮点数
% n为保留小数位数

% 将x扩大到要保留的小数位数的整数倍
factor = 10^dec_place;
num = num * factor;

% 获取待舍位数的左侧数字
left_num = floor(num/10);

% 获得待舍位数
round_num = mod(num,10);

% 判断待舍位数左侧数字的奇偶性
if mod(left_num,2)==0
    % 左侧数字为偶数，采用"五舍六入"策略
    if round_num<5
        rounded_num = floor(num/10)/factor;
    else
        rounded_num = ceil(num/10)/factor;
    end
else
    % 左侧数字为奇数，采用"四舍六入五考虑"策略
    if round_num<5
        rounded_num = floor(num/10)/factor;
    elseif round_num>5
        rounded_num = ceil(num/10)/factor;
    else
        % 如果待舍位数为5，则将结果末位保证是偶数
        if mod(floor(num/factor),2)==0
            rounded_num = floor(num/10)/factor;
        else
            rounded_num = ceil(num/10)/factor;
        end
    end
end
end

