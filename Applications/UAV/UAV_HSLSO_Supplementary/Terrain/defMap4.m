function [X, Y, Z] = defMap4(mapRange)
    x = linspace(0, mapRange(1), 100);
    y = linspace(0, mapRange(2), 100);
    [X, Y] = meshgrid(x, y);

    numPeaks = 20;
    Z = zeros(size(X));

    for i = 1:numPeaks
        peakHeight = 50 + rand() * 120;
        peakWidth  = 30 + rand() * 80;
        peakX = rand() * mapRange(1);
        peakY = rand() * mapRange(2);

        Z = Z + peakHeight * exp(-((X - peakX).^2 + (Y - peakY).^2) / peakWidth^2);
    end

    noiseLevel = 10;
    Z = Z + noiseLevel * rand(size(Z)) - noiseLevel / 2;
end
