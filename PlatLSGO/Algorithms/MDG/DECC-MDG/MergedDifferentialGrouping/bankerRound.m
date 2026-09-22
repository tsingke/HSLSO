function rounded_num = bankerRound(num, dec_place)
% This is banker's rounding
% Note: MATLAB keeps four decimal places by default

% x is the input floating-point number
% n is the number of decimal places to keep

% Scale x up to an integer multiple of the number of decimal places to keep
factor = 10^dec_place;
num = num * factor;

% Get the digit to the left of the digit to be rounded off
left_num = floor(num/10);

% Get the digit to be rounded off
round_num = mod(num,10);

% Determine the parity of the digit to the left of the digit to be rounded off
if mod(left_num,2)==0
    % The left digit is even, so the "5 down, 6 up" strategy is used
    if round_num<5
        rounded_num = floor(num/10)/factor;
    else
        rounded_num = ceil(num/10)/factor;
    end
else
    % The left digit is odd, so the "4 down, 6 up, 5 considered" strategy is used
    if round_num<5
        rounded_num = floor(num/10)/factor;
    elseif round_num>5
        rounded_num = ceil(num/10)/factor;
    else
        % If the digit to be rounded off is 5, make the last digit of the result even
        if mod(floor(num/factor),2)==0
            rounded_num = floor(num/10)/factor;
        else
            rounded_num = ceil(num/10)/factor;
        end
    end
end
end

