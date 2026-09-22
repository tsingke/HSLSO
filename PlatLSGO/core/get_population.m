function n = get_population(name)
switch upper(name)
    case 'HSLSO', n = 400;
    case 'TPCSO', n = 500;
    case 'HCLPSO', n = 500;
    case 'SCLDPSO', n = 500;
    case 'DCSO', n = 500;
    case 'WGA', n = 120;
    case 'CSO', n = 500;
    case 'SLPSO', n = 200;
    case 'APSO_DEE', n = 100;
    case 'EAPSO', n = 100;
    case 'CCOS', n = 100;
    case 'DECC_MDG', n = 50;
    case 'LLSO', n = 500;
    case 'DLLSO', n = 500;
    case 'RLLPSO', n = 500;
    case 'AHLSO', n = 500;
    case 'PCLSO', n = 200;
    case 'DPCLSO', n = 600;
    case 'RCIPSO', n = 900;
    case 'DECC_DG2', n = 50;
    case 'MOS', n = 400;
    case 'MLSHADE_SPA', n = 250;
    otherwise, error('Unknown algorithm.');
end
end
