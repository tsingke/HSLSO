function visualizeTerrainContour(X, Y, Z)
    figure;
    contourf(X, Y, Z, 40, 'LineColor', 'none');
    colormap(turbo);
    colorbar;
    title('Contour Map of Terrain');
    xlabel('X');
    ylabel('Y');
    axis equal tight;
end
