function value = msa_fitness_counter(action)
%MSA_FITNESS_COUNTER Track real objective calls *during* optimizer search.
% Other decoding and final validation are not charged to the optimizer.
persistent enabled count
if isempty(enabled), enabled=false; count=0; end
switch lower(action)
    case 'reset'
        count=0;enabled=true;
    case 'increment'
        if enabled,count=count+1;end
    case 'get'
        % Read without altering state.
    case 'off'
        enabled=false;
    otherwise
        error('Unknown msa_fitness_counter action: %s',action);
end
value=count;
end
