function [xmax,xmin,vmax,vmin,Dim] = get_ranges(suite,FuncId,D)
global initial_flag
initial_flag = 0;
Dim = D;

switch upper(suite)
    case 'CEC2010'
        a = [1,4,7,8,9,12,13,14,17,18,19,20];
        b = [2,5,10,15];
        c = [3,6,11,16];
        if ismember(FuncId,a)
            xmax = 100; xmin = -100; vmax = 100; vmin = -100;
        elseif ismember(FuncId,b)
            xmax = 5; xmin = -5; vmax = 5; vmin = -25;
        elseif ismember(FuncId,c)
            xmax = 32; xmin = -32; vmax = 32; vmin = -32;
        else
            error('Invalid CEC2010 function id.');
        end
    case 'CEC2013'
        a = [1,4,7,8,11,12,13,14,15];
        b = [2,5,9];
        c = [3,6,10];
        if ismember(FuncId,a)
            xmax = 100; xmin = -100; vmax = 100; vmin = -100;
        elseif ismember(FuncId,b)
            xmax = 5; xmin = -5; vmax = 5; vmin = -25;
        elseif ismember(FuncId,c)
            xmax = 32; xmin = -32; vmax = 32; vmin = -32;
        else
            error('Invalid CEC2013 function id.');
        end
        if FuncId == 13 || FuncId == 14
            Dim = 905;
        elseif FuncId == 15
            Dim = 1000;
        end
    otherwise
        error('Unknown suite.');
end
end
