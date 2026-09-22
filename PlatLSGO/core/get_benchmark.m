function f = get_benchmark(suite)
if strcmpi(suite,'CEC2010')
    f = @benchmark_func_2010_LSGO;
elseif strcmpi(suite,'CEC2013')
    f = @benchmark_func_2013_LSGO;
else
    error('Unknown suite.');
end
end
