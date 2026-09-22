function visualizeTerrain3D(X, Y, Z)
    figure;
    surf(X, Y, Z, Z, 'EdgeColor', 'none');
    colormap(turbo);
    colorbar;
    title('3D Terrain Visualization');
    xlabel('X');
    ylabel('Y');
    zlabel('Height Z');
    view(45, 45);
    shading interp;
    axis tight;
    grid on;
end
